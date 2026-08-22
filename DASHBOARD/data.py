"""Data access for the dashboard.

Everything the UI needs comes through here, so the views never touch the model,
the database or the simulator directly. That single seam is what let the
operational loaders move onto the FastAPI service without any view changing.

Set API_BASE_URL and the forecast, alerts, summary and site list come over HTTP;
leave it unset and they are computed locally from the saved model. The analytical
loaders (the baseline page, the daily simulation, the 33k-row panels) always read
locally - they are the wrong shape to push through JSON.

All loaders are cached. The simulation takes a few seconds and must not re-run on
every widget interaction.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import streamlit as st

import api_client
from mig_cement.config import settings
from mig_cement.data import load, preprocess
from mig_cement.features.build import build_weekly_panel
from mig_cement.inventory import simulate as inv

CACHE_TTL = 3600

# Four loaders can be served over HTTP; the rest read locally. The split is not
# arbitrary - the operational numbers (what to order, for which site, this week)
# are what the API exists to serve, while the analytical pages need 33k-row
# panels that would be far slower as JSON than as a local parquet read.
#
# `api_client.in_api_mode()` is driven by API_BASE_URL. Unset means local, which
# is how the notebooks and a bare `streamlit run` still work.


# --------------------------------------------------------------------------- #
# artefacts
# --------------------------------------------------------------------------- #
@st.cache_resource
def get_model():
    """The fitted pipeline. `cache_resource` because it is not serialisable data."""
    import joblib
    path = settings.models_dir / "rf_demand_forecaster.joblib"
    if not path.exists():
        return None
    return joblib.load(path)


@st.cache_data(ttl=CACHE_TTL)
def get_model_metadata() -> dict:
    path = settings.models_dir / "rf_demand_forecaster_meta.json"
    return json.loads(path.read_text()) if path.exists() else {}


# --------------------------------------------------------------------------- #
# panels
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=CACHE_TTL)
def get_clean_daily() -> pd.DataFrame:
    """Cleaned daily panel, straight from the database."""
    clean, _ = preprocess.build_clean_panel(load.load_panel(), mode="cap_deliveries")
    clean["date"] = pd.to_datetime(clean["date"])
    return clean


@st.cache_data(ttl=CACHE_TTL)
def get_weekly_panel() -> pd.DataFrame:
    weekly, _ = build_weekly_panel(get_clean_daily())
    return weekly


@st.cache_data(ttl=CACHE_TTL)
def get_forecasts() -> pd.DataFrame:
    """Weekly forecasts, with the actuals needed to score them.

    From the API when one is configured, otherwise the parquet the pipeline
    wrote. Both carry `forecast_tonnes` and `actual_tonnes`, so the performance
    page works either way.
    """
    if api_client.in_api_mode():
        fc = api_client.fetch_forecasts()
    else:
        path = settings.processed_dir / "test_forecasts.parquet"
        if not path.exists():
            return pd.DataFrame()
        fc = pd.read_parquet(path)

    if fc.empty:
        return fc
    fc["date"] = pd.to_datetime(fc["date"])
    fc["abs_error"] = (fc.actual_tonnes - fc.forecast_tonnes).abs()
    fc["pct_error"] = np.where(fc.actual_tonnes != 0,
                               fc.abs_error / fc.actual_tonnes, np.nan)
    return fc


@st.cache_data(ttl=CACHE_TTL)
def get_site_sigma() -> pd.Series:
    """Per-site forecast error, for safety stock. Falls back to a global figure."""
    path = settings.processed_dir / "per_site_sigma.parquet"
    if path.exists():
        return pd.read_parquet(path).sigma_weekly
    sites = get_weekly_panel().site_id.unique()
    return pd.Series(30.53, index=sites, name="sigma_weekly")


# --------------------------------------------------------------------------- #
# simulation
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=CACHE_TTL)
def get_simulation() -> pd.DataFrame:
    """Daily (s, S) simulation over the forecast window."""
    fc = get_forecasts()
    if fc.empty:
        return pd.DataFrame()

    # Disaggregation lives in `inventory.simulate` so the API and the dashboard
    # cannot drift apart on how a weekly forecast becomes a daily plan.
    daily = inv.to_daily_plan(get_clean_daily(),
                              fc[["site_id", "date", "forecast_tonnes"]])
    return inv.simulate(daily, inv.safety_stock(get_site_sigma()))


@st.cache_data(ttl=CACHE_TTL)
def _inventory() -> tuple[pd.DataFrame, pd.Series]:
    """Alerts and summary together - they come from one simulation run.

    Fetching them separately would run the simulation twice and let the two
    halves disagree.
    """
    if api_client.in_api_mode():
        return api_client.fetch_inventory()
    sim = get_simulation()
    if sim.empty:
        return pd.DataFrame(), pd.Series(dtype=float)
    return inv.reorder_alerts(sim), inv.summarise(sim)


def get_alerts() -> pd.DataFrame:
    return _inventory()[0]


def get_policy_summary() -> pd.Series:
    return _inventory()[1]


@st.cache_data(ttl=CACHE_TTL)
def get_targets_table() -> pd.DataFrame:
    """Business targets against MIG's recorded practice."""
    sim = get_simulation()
    if sim.empty:
        return pd.DataFrame()
    return inv.backtest_policy(sim, get_clean_daily())


def get_site_projection(site_id: str) -> pd.DataFrame:
    sim = get_simulation()
    return inv.project_silo_levels(sim, site_id) if not sim.empty else pd.DataFrame()


# --------------------------------------------------------------------------- #
# baseline — what the data looked like before the forecast
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=CACHE_TTL)
def get_baseline_window() -> pd.DataFrame:
    """Recorded practice over the same window the policy is scored on."""
    sim = get_simulation()
    if sim.empty:
        return pd.DataFrame()
    start = sim.date.min() + pd.Timedelta(days=inv.WARMUP_DAYS)
    w = get_clean_daily()
    w = w[(w.date >= start) & (w.date <= sim.date.max())].copy()
    w["utilisation"] = w.closing_inventory_tonnes / w.silo_capacity
    w["cover_days"] = w.closing_inventory_tonnes / w.consumed_tonnes.replace(0, np.nan)
    return w


@st.cache_data(ttl=CACHE_TTL)
def get_baseline_stats() -> dict:
    w = get_baseline_window()
    if w.empty:
        return {}
    pour = w[w.planned_pour_tonnes > 0]
    return {
        "pour_readiness": (pour.served_tonnes >= pour.planned_pour_tonnes - 1e-6).mean(),
        "rejected_pct": w.rejected_delivery_tonnes.sum() / w.deliveries_tonnes.sum(),
        "rejected_t": w.rejected_delivery_tonnes.sum(),
        "jammed_or_starved": ((w.utilisation > 0.9) | (w.utilisation < 0.1)).mean(),
        "in_band": ((w.utilisation >= 0.2) & (w.utilisation <= 0.8)).mean(),
        "ordered_t": w.deliveries_tonnes.sum(),
        "unmet_t": (pour.planned_pour_tonnes - pour.served_tonnes).clip(lower=0).sum(),
    }


@st.cache_data(ttl=CACHE_TTL)
def get_behaviour_profile() -> pd.DataFrame:
    """Ordering behaviour by site type — the bimodal failure."""
    w = get_baseline_window()
    if w.empty:
        return pd.DataFrame()
    return pd.DataFrame({
        "sites": w.groupby("behavior").site_id.nunique(),
        "mean fill %": 100 * w.groupby("behavior").utilisation.mean(),
        "median days of cover": w.groupby("behavior").cover_days.median(),
        "% days empty": 100 * (w.closing_inventory_tonnes < 0.01)
                              .groupby(w.behavior).mean(),
    }).round(1)


@st.cache_data(ttl=CACHE_TTL)
def get_rain_response() -> pd.DataFrame:
    """Consumption by rainfall band — the 15 mm cliff."""
    d = get_clean_daily()
    bands = pd.cut(d.rain_mm, [-0.01, 5, 10, 14, 15, 16, 20, 50])
    out = d.groupby(bands, observed=True).agg(
        days=("consumed_tonnes", "size"),
        pct_zero=("consumed_tonnes", lambda s: 100 * (s == 0).mean()),
        mean_consumed=("consumed_tonnes", "mean"),
        mean_planned=("planned_pour_tonnes", "mean")).reset_index()
    out["band"] = out.rain_mm.astype(str)
    return out


@st.cache_data(ttl=CACHE_TTL)
def get_capacity_constrained() -> pd.DataFrame:
    """Sites whose silo holds less than one week of demand."""
    wk = get_weekly_panel()
    s = wk.groupby("site_id").agg(capacity=("silo_capacity", "first"),
                                  mean_weekly=("y", "mean"))
    s["capacity / weekly demand"] = (s.capacity / s.mean_weekly).round(2)
    return s[s["capacity / weekly demand"] < 1.0].sort_values(
        "capacity / weekly demand").round(1)


@st.cache_data(ttl=CACHE_TTL)
def get_schedule_bias() -> dict:
    """Planned pour vs actual consumption — the over-ordering built into current practice."""
    d = get_clean_daily()
    over = d.planned_pour_tonnes - d.consumed_tonnes
    return {"pct_planned_ge_actual": (over >= -1e-9).mean(),
            "mean_over_order_t": over.mean(),
            "over_order_pct_of_demand": over.mean() / d.consumed_tonnes.mean()}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def get_sites() -> list[str]:
    if api_client.in_api_mode():
        fc = get_forecasts()
        if not fc.empty:
            return sorted(fc.site_id.unique())
    return sorted(get_weekly_panel().site_id.unique())


def get_regions() -> list[str]:
    return sorted(get_weekly_panel().region.unique())


def artefacts_present() -> tuple[bool, str]:
    """Guard so the app explains itself rather than raising a stack trace."""
    if api_client.in_api_mode():
        # In API mode the model lives in the api container, not here. Health is
        # the API's business; this only checks that it answered.
        try:
            body = api_client.health()
        except api_client.ApiUnavailable as exc:
            return False, str(exc)
        if not body.get("model_loaded"):
            return False, ("The API is up but reports no model loaded "
                           f"(status: {body.get('status')}). Run "
                           "`python -m mig_cement.pipeline` where the API can see it.")
        return True, ""

    if get_model() is None:
        return False, ("Model artefact not found. Run `make pipeline` "
                       "(or `python -m mig_cement.pipeline`) to train and save it.")
    if get_forecasts().empty:
        return False, ("No forecasts found. Run `make pipeline` to generate "
                       "`DATA/processed/test_forecasts.parquet`.")
    return True, ""
