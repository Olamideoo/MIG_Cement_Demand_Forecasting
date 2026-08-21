"""HTTP client for the forecasting API.

`data.py` is the seam the views read through; this module is what that seam calls
when the dashboard is pointed at a running API instead of loading the model
itself. Nothing here knows about Streamlit, and nothing here draws anything.

Mode is explicit, never guessed:

    API_BASE_URL set    -> the dashboard calls the API
    API_BASE_URL unset  -> the dashboard loads the model directly, as before

There is deliberately no silent fallback from the first to the second. A
fallback looks friendly and hides failure twice: locally a broken API would
render as a working dashboard, and in Docker the dashboard container has no
model at all, so falling back would swap a clear "API unreachable" for a
confusing FileNotFoundError about a file that was never meant to be there.

Every function returns exactly the frame shape `data.py` returns in local mode,
including column names, so the views cannot tell the two apart.
"""

from __future__ import annotations

import os
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Read .env at the repo root so the mode does not have to be set by hand in
# every terminal. The path is explicit rather than cwd-relative because
# `streamlit run` can be invoked from anywhere.
#
# load_dotenv does not overwrite variables that are already set, which is the
# behaviour we want: docker compose passes API_BASE_URL directly, and that must
# win over whatever a stray .env inside the image might say.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

API_BASE_URL = os.environ.get("API_BASE_URL", "").rstrip("/")

# Generous, because /forecast runs the model for 30 sites and /inventory runs a
# daily simulation on top of that. Short enough that a hung API surfaces as an
# error rather than a spinner nobody interprets.
TIMEOUT = httpx.Timeout(60.0, connect=5.0)

# The API returns operational language; the simulator and the views use the
# internal names. Translating here keeps that vocabulary difference in one place.
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
    """Flatten the nested response into the frame the views expect.

    The API nests points under each site; `data.get_forecasts` has always
    returned one flat row per site-week, so the flattening happens here rather
    than leaking a different shape into the views.
    """
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

        # `utilisation` is derived rather than transported: it is just fill
        # fraction, and sending it would let the two drift apart.
        alerts["utilisation"] = np.where(
            alerts.capacity > 0, alerts.closing / alerts.capacity, 0.0)

        # JSON has no infinity. The API sends null for "does not run out inside
        # the window"; the views expect inf, which they then render as a dash.
        alerts["days_to_stockout"] = (
            pd.to_numeric(alerts.days_to_stockout, errors="coerce")
              .fillna(np.inf))

        alerts = alerts.sort_values("days_to_stockout").reset_index(drop=True)

    return alerts, pd.Series(body["summary"], dtype=float)
