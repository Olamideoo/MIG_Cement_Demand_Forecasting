"""FastAPI application. Run: uvicorn mig_cement.api.main:app --reload"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException

from mig_cement.api import schemas
from mig_cement.api.predict import forecaster
from mig_cement.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        forecaster.load()
    except NotImplementedError:
        pass  # scaffold: model not yet available
    yield


app = FastAPI(
    title="MIG Cement Demand Forecasting API",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=schemas.HealthResponse)
def health() -> schemas.HealthResponse:
    return schemas.HealthResponse(
        status="ok", model_loaded=forecaster.model is not None
    )


@app.get("/model/info", response_model=schemas.ModelInfo)
def model_info() -> schemas.ModelInfo:
    raise HTTPException(501, "not implemented")


@app.post("/forecast", response_model=schemas.ForecastResponse)
def forecast(req: schemas.ForecastRequest) -> schemas.ForecastResponse:
    raise HTTPException(501, "not implemented")


@app.post("/forecast/batch", response_model=list[schemas.ForecastResponse])
def forecast_batch(req: schemas.BatchForecastRequest):
    raise HTTPException(501, "not implemented")


@app.post("/forecast/refresh")
def forecast_refresh(req: schemas.ForecastRequest):
    """On-demand recompute for one series. See the hybrid serving pattern in
    WORKFLOW.md section 2.3."""
    raise HTTPException(501, "not implemented")


@app.get("/inventory/alerts", response_model=list[schemas.ReorderAlert])
def inventory_alerts(region: str | None = None):
    raise HTTPException(501, "not implemented")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.api_host, port=settings.api_port)
