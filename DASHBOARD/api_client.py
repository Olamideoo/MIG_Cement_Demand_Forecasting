"""HTTP client for the forecasting API.

`data.py` is the seam the views read through; this module is what that seam calls
when the dashboard is pointed at a running API instead of loading the model
itself. Nothing here knows about Streamlit, and nothing here draws anything.
"""

from __future__ import annotations

import os
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

API_BASE_URL = os.environ.get("API_BASE_URL", "").rstrip("/")
TIMEOUT = httpx.Timeout(60.0, connect=5.0)

ALERT_FIELDS = {
    "current_inventory_tonnes": "closing",
    "silo_capacity_tonnes": "capacity",
    "reorder_point_tonnes": "reorder_point",
    "suggested_order_tonnes": "suggested_order_t",
}


class ApiUnavailable(RuntimeError):
    """Raised when the API cannot be reached or answers with an error.

    Carries a message meant to be shown to a person, not a stack trace.
    """


def in_api_mode() -> bool:
    return bool(API_BASE_URL)


def _post(path: str, payload: dict) -> dict:
    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            response = client.post(f"{API_BASE_URL}{path}", json=payload)
    except httpx.RequestError as exc:
        raise ApiUnavailable(
            f"Could not reach the API at {API_BASE_URL} ({exc.__class__.__name__}). "
            "Is the api service running?") from exc

    if response.status_code == 503:
        raise ApiUnavailable(
            "The API is running but has no model loaded. Run "
            "`python -m mig_cement.pipeline`, or check that MODELS/ reached the "
            "container.")
    if response.is_error:
        raise ApiUnavailable(
            f"{path} returned {response.status_code}: {response.text[:200]}")

    return response.json()


# --------------------------------------------------------------------------- #
# health
# --------------------------------------------------------------------------- #
def health() -> dict:
    """Startup probe. Cheap by design - it never runs the model."""
    try:
        with httpx.Client(timeout=httpx.Timeout(10.0, connect=5.0)) as client:
            response = client.get(f"{API_BASE_URL}/health")
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as exc:
        raise ApiUnavailable(
            f"Could not reach the API at {API_BASE_URL} ({exc.__class__.__name__}). "
            "Start it with `make api`, or set API_BASE_URL to point somewhere "
            "else.") from exc


# --------------------------------------------------------------------------- #
# forecasts
# --------------------------------------------------------------------------- #
def fetch_forecasts(horizon_weeks: int = 8) -> pd.DataFrame:
    
    body = _post("/forecast", {"horizon_weeks": horizon_weeks})

    rows = [
        {"site_id": site["site_id"], **point}
        for site in body["sites"] for point in site["forecast"]
    ]
    if not rows:
        return pd.DataFrame()

    fc = pd.DataFrame(rows).rename(columns={
        "predicted_tonnes": "forecast_tonnes",
        "actual_tonnes": "actual_tonnes",
    })
    fc["date"] = pd.to_datetime(fc["date"])
    return fc.sort_values(["site_id", "date"]).reset_index(drop=True)


def fetch_sites(horizon_weeks: int = 1) -> list[str]:
    """Site list, derived from a forecast rather than its own endpoint.

    A dedicated /sites endpoint would be a second round trip to fill a dropdown.
    Asking for a one-week horizon keeps the response small.
    """
    body = _post("/forecast", {"horizon_weeks": horizon_weeks})
    return sorted(site["site_id"] for site in body["sites"])


# --------------------------------------------------------------------------- #
# inventory
# --------------------------------------------------------------------------- #
def fetch_inventory(horizon_weeks: int = 8) -> tuple[pd.DataFrame, pd.Series]:
    """Alerts and the policy summary in one call.

    Both come from the same simulation, so splitting them into two requests
    would run it twice and risk the two halves disagreeing.
    """
    body = _post("/inventory", {"horizon_weeks": horizon_weeks})

    alerts = pd.DataFrame(body["alerts"])
    if not alerts.empty:
        alerts = alerts.rename(columns=ALERT_FIELDS)

        alerts["utilisation"] = np.where(
            alerts.capacity > 0, alerts.closing / alerts.capacity, 0.0)

        
        alerts["days_to_stockout"] = (
            pd.to_numeric(alerts.days_to_stockout, errors="coerce")
              .fillna(np.inf))

        alerts = alerts.sort_values("days_to_stockout").reset_index(drop=True)

    return alerts, pd.Series(body["summary"], dtype=float)
