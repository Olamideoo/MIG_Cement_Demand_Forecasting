"""Request/response payloads - pydantic BaseModel.

Distinct from config.py, which uses BaseSettings for environment configuration.

Two business endpoints, so two contracts: one for the demand forecast, one for
the inventory simulation built on top of it. `HealthResponse` is the third, and
is infrastructure rather than business - see main.py.
"""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, Field

ALL_TYPES = "ALL"


# --------------------------------------------------------------------------- #
# 1. demand forecast
# --------------------------------------------------------------------------- #
class ForecastRequest(BaseModel):
    """Omit `site_id` for all 30 sites.

    The dashboard overview needs the whole estate at once, so one request covers
    both the single-site and the all-sites case rather than splitting them.
    """

    site_id: str | None = Field(None, examples=["SITE_001"],
                                description="Omit for every site.")
    horizon_weeks: int = Field(8, ge=1, le=8)


class ForecastPoint(BaseModel):
    date: date
    predicted_tonnes: float
    lower: float
    upper: float
    horizon_week: int = Field(..., ge=1, description="weeks ahead of the training cutoff")


class SiteForecast(BaseModel):
    site_id: str
    cement_type: str = Field(
        ALL_TYPES,
        description="Every site handles all three grades and the model forecasts "
                    "total site demand, so this is always ALL.")
    forecast: list[ForecastPoint]


class ForecastResponse(BaseModel):
    model_version: str
    generated_at: datetime
    horizon_weeks: int
    sites: list[SiteForecast]


# --------------------------------------------------------------------------- #
# 2. inventory simulation
# --------------------------------------------------------------------------- #
class SimulationRequest(BaseModel):
    site_ids: list[str] | None = Field(None, description="Omit for every site.")
    region: str | None = None
    horizon_weeks: int = Field(8, ge=1, le=8)


class ReorderAlert(BaseModel):
    site_id: str
    region: str | None = None
    current_inventory_tonnes: float
    reorder_point_tonnes: float
    silo_capacity_tonnes: float
    days_to_stockout: float | None = Field(
        None, description="null when the site does not run out inside the window")
    suggested_order_tonnes: float = 0.0
    severity: str = Field(..., examples=["red", "amber", "green"])


class PolicySummary(BaseModel):
    """Scored with the first week excluded - the simulation starts with nothing in
    transit, so week one understates readiness for reasons of setup, not policy."""

    pour_readiness: float
    stockout_days: float
    unmet_tonnes: float
    mean_utilisation: float
    write_offs_t: float
    total_ordered_t: float
    deliveries: float
    sites_meeting_target: float


class SimulationResponse(BaseModel):
    model_version: str
    generated_at: datetime
    horizon_weeks: int
    summary: PolicySummary
    alerts: list[ReorderAlert]


# --------------------------------------------------------------------------- #
# liveness
# --------------------------------------------------------------------------- #
class HealthResponse(BaseModel):
    """Cheap enough to poll every few seconds - no model call, no simulation."""

    status: str = Field(..., examples=["ok", "degraded"])
    model_loaded: bool
    model_version: str | None = None
    sites_servable: int = 0
