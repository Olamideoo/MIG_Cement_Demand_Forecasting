"""Thin client for the forecasting API. Cached - the overview page needs all
90 series and must not hammer the service."""

from __future__ import annotations

import requests

from mig_cement.config import settings

TIMEOUT = 10


def get_health() -> dict:
    return requests.get(f"{settings.api_base_url}/health", timeout=TIMEOUT).json()


def post_forecast(site_id: str, cement_type: str, horizon_weeks: int = 8) -> dict:
    raise NotImplementedError


def post_forecast_batch(horizon_weeks: int = 8) -> list[dict]:
    raise NotImplementedError


def get_alerts(region: str | None = None) -> list[dict]:
    raise NotImplementedError
