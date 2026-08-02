"""Request/response payloads - pydantic BaseModel.

Distinct from config.py, which uses BaseSettings for environment configuration.
This file is Contract 2 from WORKFLOW.md: frozen early so the Dash app can be
built against a stub before the model exists.
"""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, Field


class ForecastRequest(BaseModel):
    site_id: str = Field(..., examples=["SITE_001"])
    cement_type: str = Field(..., examples=["CEM_I"])
    horizon_weeks: int = Field(8, ge=1, le=8)


class ForecastPoint(BaseModel):
    date: date
    predicted_tonnes: float
    lower: float
    upper: float


class ForecastResponse(BaseModel):
    site_id: str
    cement_type: str
    forecast: list[ForecastPoint]
    model_version: str
    generated_at: datetime


class BatchForecastRequest(BaseModel):
    """Omit site_ids for all 30 sites. The dashboard overview needs this -
    90 sequential single-site calls would make the page unusable."""
    site_ids: list[str] | None = None
    cement_types: list[str] | None = None
    horizon_weeks: int = Field(8, ge=1, le=8)


class ReorderAlert(BaseModel):
    site_id: str
    cement_type: str
    current_inventory_tonnes: float
    reorder_point_tonnes: float
    silo_capacity_tonnes: float
    days_to_stockout: float | None
    severity: str = Field(..., examples=["red", "amber", "green"])


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ModelInfo(BaseModel):
    name: str
    version: str
    stage: str
    trained_at: datetime | None
    feature_count: int
