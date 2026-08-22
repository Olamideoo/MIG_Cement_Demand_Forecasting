"""Request/response payloads - pydantic BaseModel.
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
    actual_tonnes: float | None = Field(
        None,
        description="Observed consumption, where the week has already happened. "
                    "null for genuinely future weeks. Present so a client can "
                    "score the forecast without a second data source.")


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
# 1b. what-if prediction
# --------------------------------------------------------------------------- #
class PredictRequest(BaseModel):
    """Score one site-week, optionally with the schedule changed.

    Only `planned_pour_tonnes` is a field an operator can meaningfully set. The
    remaining features are either fixed properties of the site (capacity, region,
    behaviour) or forward sums derived from the daily pour schedule, so they are
    read from the real panel rather than typed. That keeps this endpoint from
    becoming a way to hand the model an impossible combination.
    """

    site_id: str = Field(..., examples=["SITE_001"])
    week: date | None = Field(
        None, examples=["2024-10-07"],
        description="Monday of the week to score. Defaults to the first week "
                    "after the training cutoff.")
    planned_pour_tonnes: float | None = Field(
        None, ge=0, examples=[400.0],
        description="Override the scheduled pour. Omit to score the week as it "
                    "actually stands.")


class PredictResponse(BaseModel):
    site_id: str
    week: date
    predicted_tonnes: float

    lower: float | None = Field(
        None,
        description="95% interval, from this site's own hold-out error. Returned "
                    "only when nothing was overridden. An interval measures how "
                    "well the model does on the real schedule, so putting one "
                    "around a hypothetical would claim a precision that has not "
                    "been measured.")
    upper: float | None = None

    planned_pour_tonnes: float = Field(
        ..., description="The value used, after any override.")
    baseline_tonnes: float | None = Field(
        None, description="Prediction without the override, so the effect of the "
                          "change is visible. null when nothing was overridden.")
    actual_tonnes: float | None = Field(
        None, description="Observed consumption, if that week has happened.")

    site_pour_range: list[float] = Field(
        ...,
        description="The smallest and largest pour ever recorded at this site. "
                    "An override outside it is being asked of a site that has "
                    "never done it.")
    in_training_range: bool = Field(
        ...,
        description="False when an override falls outside what THIS site has "
                    "done. A random forest averages the training leaves it lands "
                    "in, so outside that range it borrows from other sites and "
                    "eventually stops responding altogether.")
    warning: str | None = None
    model_version: str
    generated_at: datetime


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
