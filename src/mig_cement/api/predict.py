"""Inference. Loads the persisted model and serves forecasts over the same
feature builder used in training.

Two notes on how this differs from the original scaffold, both because the model
that actually won is not the one the scaffold anticipated:

1. **Not recursive.** Every feature is derived from the pour schedule, which is
   known up to 4 weeks ahead, so all horizons are predicted in one pass. There is
   no feeding of week *n* back in as an input to week *n+1*, and therefore no
   error compounding - accuracy is flat across the 8-week horizon.

2. **Site grain, not site x cement type.** Each of the 30 sites handles all three
   cement grades; the model aggregates over them and takes `site_id`, `region` and
   `behavior` as the categorical features. Responses report cement_type as "ALL".

Feature construction is delegated to `features.build` - the same function the
training pipeline calls. Nothing in this file reimplements it. That is the
train/serve skew failure described in WORKFLOW.md section 2.3.
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from mig_cement.config import settings
from mig_cement.data import load, preprocess
from mig_cement.features.build import FEATURES, build_weekly_panel
from mig_cement.inventory import simulate as inv

MODEL_FILE = "rf_demand_forecaster.joblib"
META_FILE = "rf_demand_forecaster_meta.json"

# Interval half-width. The uncertainty that matters over an 8-week horizon is the
# model's realised hold-out error, not the spread between trees, so intervals come
# from each site's own sigma. 1.96 -> nominal 95%.
INTERVAL_Z = 1.96
FALLBACK_SIGMA = 30.53  # estate hold-out RMSE, used only if the sigma file is absent


class ModelNotLoaded(RuntimeError):
    """Raised when a request arrives before the artefact is available."""


class Forecaster:
    """Holds the loaded model, its metadata and the feature panel.

    The panel is built once at startup, not per request: rebuilding it means
    re-reading the database and recomputing the schedule features, which takes
    seconds. `refresh()` rebuilds it on demand.
    """

    def __init__(self) -> None:
        self.model = None
        self.feature_columns: list[str] = []
        self.model_version: str = "unloaded"
        self.metadata: dict = {}
        self._panel: pd.DataFrame | None = None
        self._daily: pd.DataFrame | None = None
        self._sigma: pd.Series | None = None

    # --- lifecycle --------------------------------------------------------- #
    def load(self) -> None:
        """Load the artefact and warm the panel. Called once at app startup."""
        model_path = settings.models_dir / MODEL_FILE
        if not model_path.exists():
            raise FileNotFoundError(
                f"{model_path} not found - run `python -m mig_cement.pipeline`")

        self.model = joblib.load(model_path)

        meta_path = settings.models_dir / META_FILE
        self.metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        self.feature_columns = self.metadata.get("features", FEATURES)
        self.model_version = self.metadata.get("created_utc", "unknown")

        if self.feature_columns != FEATURES:
            raise ValueError(
                "feature order in the artefact does not match features.build.FEATURES "
                f"({self.feature_columns} vs {FEATURES}) - retrain before serving")

        self.refresh()

    def refresh(self) -> None:
        """Rebuild the panel from the database. Use after new data lands."""
        clean, _ = preprocess.build_clean_panel(load.load_panel(), mode="cap_deliveries")
        clean["date"] = pd.to_datetime(clean["date"])
        self._daily = clean
        self._panel, _ = build_weekly_panel(clean)

        path = settings.processed_dir / "per_site_sigma.parquet"
        self._sigma = (pd.read_parquet(path).sigma_weekly if path.exists()
                       else pd.Series(FALLBACK_SIGMA, index=self._panel.site_id.unique()))

    @property
    def is_loaded(self) -> bool:
        return self.model is not None and self._panel is not None

    def _require(self) -> None:
        if not self.is_loaded:
            raise ModelNotLoaded("model artefact not loaded")

    # --- the servable window ------------------------------------------------ #
    def servable(self) -> pd.DataFrame:
        """Weeks the model has not been trained on.

        Forecasting a week the model was fitted on would report memorised values as
        predictions, so the served window starts strictly after the training cutoff.
        """
        self._require()
        cutoff = pd.Timestamp(self.metadata.get("trained_on", {}).get("to")
                              or settings.val_end)
        return self._panel[self._panel.date > cutoff]

    def sites(self) -> list[str]:
        self._require()
        return sorted(self._panel.site_id.unique())

    # --- prediction --------------------------------------------------------- #
    def _predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Demand cannot be negative, so predictions are clipped at zero - the same
        clip the training pipeline applies when it scores."""
        return np.clip(self.model.predict(frame[self.feature_columns]), 0, None)

    def _with_intervals(self, frame: pd.DataFrame) -> pd.DataFrame:
        pred = self._predict(frame)
        sigma = frame.site_id.map(self._sigma).fillna(FALLBACK_SIGMA).to_numpy()
        margin = INTERVAL_Z * sigma
        return pd.DataFrame({
            "site_id": frame.site_id.to_numpy(),
            "date": pd.to_datetime(frame.date).to_numpy(),
            "predicted_tonnes": pred.round(2),
            "lower": np.clip(pred - margin, 0, None).round(2),
            "upper": (pred + margin).round(2),
            "horizon_week": frame.groupby("site_id").cumcount().to_numpy() + 1,
        })

    def predict_batch(self, site_ids: list[str] | None = None,
                      horizon_weeks: int = 8) -> pd.DataFrame:
        """Forecast one site, several, or the whole estate in a single pass.

        `site_ids=None` means all 30. The dashboard overview needs the estate at
        once, and 30 sequential calls would each re-enter the model for eight rows.
        """
        self._require()
        window = self.servable()
        if site_ids:
            unknown = sorted(set(site_ids) - set(window.site_id))
            if unknown:
                raise KeyError(", ".join(unknown))
            window = window[window.site_id.isin(site_ids)]
        rows = (window.sort_values(["site_id", "date"])
                      .groupby("site_id", group_keys=False).head(horizon_weeks))
        return self._with_intervals(rows)

    # --- inventory ---------------------------------------------------------- #
    def simulate(self, site_ids: list[str] | None = None,
                 region: str | None = None,
                 horizon_weeks: int = 8) -> tuple[pd.DataFrame, pd.Series]:
        """Run the daily (s, S) simulation over the forecast window.

        Returns `(alerts, summary)`. The summary is scored the same way the
        notebook scores it, warm-up week excluded.
        """
        self._require()
        region_of = self._panel.groupby("site_id").region.first()

        if region:
            in_region = set(region_of[region_of == region].index)
            site_ids = sorted(set(site_ids) & in_region) if site_ids else sorted(in_region)
            if not site_ids:
                return pd.DataFrame(), pd.Series(dtype=float)

        fc = self.predict_batch(site_ids, horizon_weeks)
        plan = inv.to_daily_plan(
            self._daily, fc.rename(columns={"predicted_tonnes": "forecast_tonnes"}))
        sim = inv.simulate(plan, inv.safety_stock(self._sigma))

        alerts = inv.reorder_alerts(sim)
        alerts = alerts.assign(region=alerts.site_id.map(region_of))
        return alerts, inv.summarise(sim)


forecaster = Forecaster()
