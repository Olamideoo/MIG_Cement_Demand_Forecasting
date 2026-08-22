"""FastAPI application. Run: uvicorn mig_cement.api.main:app --reload

Three business endpoints:

    POST /forecast    the RF demand forecaster - 8 weekly forecasts per site
    POST /predict     score one site-week, optionally with the pour changed
    POST /inventory   the (s, S) simulation built on those forecasts

plus one infrastructure endpoint:

    GET  /health      liveness probe for Docker, ECS and the load balancer
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
    # Liveness probe. Reports state, never computes it.

    return schemas.HealthResponse(
        status="ok" if forecaster.is_loaded else "degraded",
        model_loaded=forecaster.is_loaded,
        model_version=forecaster.model_version if forecaster.is_loaded else None,
        sites_servable=len(forecaster.sites()) if forecaster.is_loaded else 0,
    )


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
                # `where(notna)` turns NaN into None. JSON has no NaN, and pydantic
                # would otherwise coerce it to a float the client cannot test for.
                forecast=[schemas.ForecastPoint(**p) for p in
                          g.drop(columns=["site_id"])
                           .astype(object).where(g.drop(columns=["site_id"]).notna(), None)
                           .to_dict(orient="records")])
            for site, g in rows.groupby("site_id")
        ],
    )


PREDICT_EXAMPLES = {
    "as_scheduled": {
        "summary": "Score a week as it stands",
        "description": "No override - what the model expects given the real pour.",
        "value": {"site_id": "SITE_001", "week": "2024-10-07"},
    },
    "bigger_pour": {
        "summary": "What if the pour were larger?",
        "description": "Returns baseline_tonnes alongside, so the effect is visible.",
        "value": {"site_id": "SITE_001", "week": "2024-10-07",
                  "planned_pour_tonnes": 350},
    },
    "beyond_this_site": {
        "summary": "More than this site has ever poured",
        "description": "SITE_009's largest recorded pour is 118 t. Asking for 300 "
                       "returns a number, but with in_training_range false - the "
                       "model has no example of this site at that scale.",
        "value": {"site_id": "SITE_009", "week": "2024-10-07",
                  "planned_pour_tonnes": 300},
    },
    "beyond_every_site": {
        "summary": "Beyond anything in the data at all",
        "description": "Past the estate maximum the prediction stops moving "
                       "entirely - a random forest cannot extrapolate.",
        "value": {"site_id": "SITE_001", "week": "2024-10-07",
                  "planned_pour_tonnes": 900},
    },
}


@app.post("/predict", response_model=schemas.PredictResponse)
def predict(
    req: Annotated[schemas.PredictRequest,
                   Body(openapi_examples=PREDICT_EXAMPLES)],
) -> schemas.PredictResponse:
    _guard()
    try:
        result = forecaster.predict_one(
            req.site_id,
            week=req.week,
            planned_pour_tonnes=req.planned_pour_tonnes)
    except KeyError:
        raise HTTPException(404, f"unknown site: {req.site_id}") from None
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from None

    return schemas.PredictResponse(
        **result,
        model_version=forecaster.model_version,
        generated_at=datetime.now(timezone.utc),
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
