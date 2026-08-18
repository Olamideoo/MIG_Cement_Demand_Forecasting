"""FastAPI application. Run: uvicorn mig_cement.api.main:app --reload

Two business endpoints, matching the two things the project produces:

    POST /forecast    the RF demand forecaster - 8 weekly forecasts per site
    POST /inventory   the (s, S) simulation built on those forecasts

plus one infrastructure endpoint:

    GET  /health      liveness probe for Docker, ECS and the load balancer

The model and the feature panel load once at startup, never per request.
Startup failure is not fatal: the service comes up and returns 503 with a usable
message, so a container that boots before its model volume is mounted is visibly
degraded rather than crash-looping.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Annotated

import pandas as pd
from fastapi import Body, FastAPI, HTTPException

from mig_cement.api import schemas
from mig_cement.api.predict import forecaster
from mig_cement.config import settings

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        forecaster.load()
        log.info("model %s loaded, %d sites servable",
                 forecaster.model_version, len(forecaster.sites()))
    except Exception:                                    # noqa: BLE001
        log.exception("startup: model unavailable, serving degraded")
    yield


app = FastAPI(
    title="MIG Cement Demand Forecasting API",
    description="8-week weekly demand forecasts per site, and the inventory "
                "simulation derived from them.",
    version="1.0.0",
    lifespan=lifespan,
)


def _guard() -> None:
    if not forecaster.is_loaded:
        raise HTTPException(503, "model not loaded - run `python -m mig_cement.pipeline`")


# --------------------------------------------------------------------------- #
# 0. liveness
# --------------------------------------------------------------------------- #
@app.get("/health", response_model=schemas.HealthResponse, tags=["infrastructure"])
def health() -> schemas.HealthResponse:
    """Liveness probe. Reports state, never computes it.

    Container orchestrators poll this every few seconds. It must not touch the
    model or the simulation - a probe that ran a forecast for 30 sites would put
    the service under constant load and time out under exactly the conditions it
    exists to detect.

    Returns 200 either way: `model_loaded: false` means the process is alive but
    has no artefact, which is a deploy problem, not a crash. Restarting the
    container would not fix it.
    """
    return schemas.HealthResponse(
        status="ok" if forecaster.is_loaded else "degraded",
        model_loaded=forecaster.is_loaded,
        model_version=forecaster.model_version if forecaster.is_loaded else None,
        sites_servable=len(forecaster.sites()) if forecaster.is_loaded else 0,
    )


# --------------------------------------------------------------------------- #
# request examples
#
# Swagger prefills its "Try it out" box from the first example below. Without
# these it invents placeholders like {"site_ids": ["string"]}, which are not
# valid inputs - editing them by hand is how you end up with malformed JSON.
# Every example here is a request that actually works.
# --------------------------------------------------------------------------- #
FORECAST_EXAMPLES = {
    "all_sites": {
        "summary": "Whole estate, full horizon",
        "description": "Omit site_id for all 30 sites. This is what the dashboard uses.",
        "value": {"horizon_weeks": 8},
    },
    "one_site": {
        "summary": "A single site",
        "value": {"site_id": "SITE_001", "horizon_weeks": 8},
    },
    "next_two_weeks": {
        "summary": "Short horizon, one site",
        "description": "Only the weeks needed for the next ordering cycle.",
        "value": {"site_id": "SITE_013", "horizon_weeks": 2},
    },
}

INVENTORY_EXAMPLES = {
    "all_sites": {
        "summary": "Whole estate",
        "description": "Every site, with the policy outcome across all of them.",
        "value": {"horizon_weeks": 8},
    },
    "by_region": {
        "summary": "One region",
        "description": "Regions are North, South, East and West.",
        "value": {"region": "North", "horizon_weeks": 8},
    },
    "specific_sites": {
        "summary": "Named sites",
        "description": "Site IDs are uppercase with an underscore: SITE_001.",
        "value": {"site_ids": ["SITE_001", "SITE_013"], "horizon_weeks": 8},
    },
    "sites_within_region": {
        "summary": "Named sites, narrowed to a region",
        "description": "site_ids and region intersect - SITE_001 is returned only "
                       "because it is in North.",
        "value": {"site_ids": ["SITE_001"], "region": "North", "horizon_weeks": 8},
    },
}


# --------------------------------------------------------------------------- #
# 1. demand forecast
# --------------------------------------------------------------------------- #
@app.post("/forecast", response_model=schemas.ForecastResponse)
def forecast(
    req: Annotated[schemas.ForecastRequest,
                   Body(openapi_examples=FORECAST_EXAMPLES)],
) -> schemas.ForecastResponse:
    """Weekly demand forecast from the random forest.

    Serves only weeks after the training cutoff - forecasting a week the model was
    fitted on would report memorised values as predictions.
    """
    _guard()
    try:
        rows = forecaster.predict_batch(
            [req.site_id] if req.site_id else None, req.horizon_weeks)
    except KeyError as exc:
        raise HTTPException(404, f"unknown site: {exc}") from None

    if rows.empty:
        raise HTTPException(404, "no unseen weeks available to forecast")

    return schemas.ForecastResponse(
        model_version=forecaster.model_version,
        generated_at=datetime.now(timezone.utc),
        horizon_weeks=req.horizon_weeks,
        sites=[
            schemas.SiteForecast(
                site_id=site,
                forecast=[schemas.ForecastPoint(**p) for p in
                          g.drop(columns=["site_id"]).to_dict(orient="records")])
            for site, g in rows.groupby("site_id")
        ],
    )


# --------------------------------------------------------------------------- #
# 2. inventory simulation
# --------------------------------------------------------------------------- #
@app.post("/inventory", response_model=schemas.SimulationResponse)
def inventory(
    req: Annotated[schemas.SimulationRequest,
                   Body(openapi_examples=INVENTORY_EXAMPLES)],
) -> schemas.SimulationResponse:
    """Daily (s, S) simulation over the forecast window.

    Returns the reorder position for every site plus the policy outcome. Severity
    is a property of the site, not a filter: every requested site is returned so a
    red one cannot be hidden by a query parameter.
    """
    _guard()
    try:
        alerts, summary = forecaster.simulate(
            site_ids=req.site_ids, region=req.region, horizon_weeks=req.horizon_weeks)
    except KeyError as exc:
        raise HTTPException(404, f"unknown site: {exc}") from None

    if alerts.empty:
        raise HTTPException(
            404, f"no sites match region={req.region!r}" if req.region
            else "simulation produced no rows")

    return schemas.SimulationResponse(
        model_version=forecaster.model_version,
        generated_at=datetime.now(timezone.utc),
        horizon_weeks=req.horizon_weeks,
        summary=schemas.PolicySummary(**{k: round(float(v), 4)
                                         for k, v in summary.items()}),
        alerts=[
            schemas.ReorderAlert(
                site_id=r.site_id,
                region=r.region,
                current_inventory_tonnes=round(float(r.closing), 2),
                reorder_point_tonnes=round(float(r.reorder_point), 2),
                silo_capacity_tonnes=round(float(r.capacity), 2),
                days_to_stockout=(None if not pd.notna(r.days_to_stockout)
                                  or r.days_to_stockout == float("inf")
                                  else round(float(r.days_to_stockout), 2)),
                suggested_order_tonnes=round(float(r.suggested_order_t), 2),
                severity=r.severity,
            )
            for r in alerts.itertuples()
        ],
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.api_host, port=settings.api_port)
