"""Inference. Loads the persisted model and serves forecasts over the same
feature builder used in training.
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from mig_cement.config import settings
from mig_cement.data import load, preprocess
from mig_cement.features.build import FEATURES, TARGET, build_weekly_panel
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

        # `y` is the observed weekly consumption. 
        actual = (frame[TARGET].to_numpy() if TARGET in frame.columns
                  else np.full(len(frame), np.nan))

        return pd.DataFrame({
            "site_id": frame.site_id.to_numpy(),
            "date": pd.to_datetime(frame.date).to_numpy(),
            "predicted_tonnes": pred.round(2),
            "lower": np.clip(pred - margin, 0, None).round(2),
            "upper": (pred + margin).round(2),
            "horizon_week": frame.groupby("site_id").cumcount().to_numpy() + 1,
            "actual_tonnes": np.round(actual.astype(float), 2),
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

    # --- what-if ------------------------------------------------------------ #
    def predict_one(self, site_id: str, week: pd.Timestamp | None = None,
                    planned_pour_tonnes: float | None = None) -> dict:
        """Score a single site-week, optionally with the pour schedule changed.
        """
        self._require()
        window = self.servable()
        rows = window[window.site_id == site_id]
        if rows.empty:
            raise KeyError(site_id)

        if week is not None:
            rows = rows[rows.date == pd.Timestamp(week)]
            if rows.empty:
                available = sorted(window[window.site_id == site_id].date.dt.date)
                raise ValueError(
                    f"no servable week {pd.Timestamp(week).date()} for {site_id}. "
                    f"Available: {available[0]} to {available[-1]}.")

        row = rows.sort_values("date").iloc[[0]].copy()
        baseline = float(self._predict(row)[0])

        site_hist = self._panel[self._panel.site_id == site_id].planned_pour_tonnes
        site_lo, site_hi = float(site_hist.min()), float(site_hist.max())

        warning, in_range = None, True
        if planned_pour_tonnes is not None:
            if planned_pour_tonnes > site_hi:
                in_range = False
                warning = (
                    f"{site_id} has never poured more than {site_hi:.0f} t; you "
                    f"asked for {planned_pour_tonnes:.0f} t. The model has no "
                    "example of this site at that scale, so it borrows from other "
                    "sites and stops responding once the value passes the estate "
                    f"maximum of {self._panel.planned_pour_tonnes.max():.0f} t.")
            elif planned_pour_tonnes < site_lo:
                in_range = False
                warning = (
                    f"{site_id} has never poured less than {site_lo:.0f} t; you "
                    f"asked for {planned_pour_tonnes:.0f} t.")
            row["planned_pour_tonnes"] = planned_pour_tonnes

        pred = float(self._predict(row)[0])
        r = row.iloc[0]

        
        if planned_pour_tonnes is None:
            margin = INTERVAL_Z * float(self._sigma.get(site_id, FALLBACK_SIGMA))
            lower, upper = round(max(pred - margin, 0.0), 2), round(pred + margin, 2)
        else:
            lower = upper = None

        return {
            "site_id": site_id,
            "week": r.date.date(),
            "predicted_tonnes": round(pred, 2),
            "lower": lower,
            "upper": upper,
            "planned_pour_tonnes": round(float(r.planned_pour_tonnes), 2),
            "baseline_tonnes": (round(baseline, 2)
                                if planned_pour_tonnes is not None else None),
            "actual_tonnes": (round(float(r[TARGET]), 2)
                              if TARGET in row.columns and pd.notna(r[TARGET])
                              else None),
            "site_pour_range": [round(site_lo, 2), round(site_hi, 2)],
            "in_training_range": in_range,
            "warning": warning,
        }

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

        
        window = self.servable()
        if site_ids:
            window = window[window.site_id.isin(site_ids)]
        rows = (window.sort_values(["site_id", "date"])
                      .groupby("site_id", group_keys=False).head(horizon_weeks))
        fc = rows[["site_id", "date"]].assign(forecast_tonnes=self._predict(rows))

        plan = inv.to_daily_plan(self._daily, fc)
        sim = inv.simulate(plan, inv.safety_stock(self._sigma))

        alerts = inv.reorder_alerts(sim)
        alerts = alerts.assign(region=alerts.site_id.map(region_of))
        return alerts, inv.summarise(sim)


forecaster = Forecaster()
