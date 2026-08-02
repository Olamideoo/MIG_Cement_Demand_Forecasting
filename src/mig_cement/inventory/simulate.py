"""Silo projection, reorder points and policy backtesting.

This module is where the four business targets are actually evidenced:
  - MAPE <= 15% (weekly grain)      -> models/evaluate.py
  - >= 98% pour readiness            -> backtest_policy()
  - 20% silo utilisation improvement -> backtest_policy()
  - 30% write-off reduction          -> backtest_policy()

Lead time is not present in the source data. settings.lead_time_days is an
assumption and must be sensitivity-tested, not presented as fact.
"""

from __future__ import annotations

import pandas as pd

from mig_cement.config import settings


def project_silo_levels(forecast: pd.DataFrame, opening: float,
                        scheduled_deliveries: pd.DataFrame,
                        silo_capacity: float) -> pd.DataFrame:
    """Roll forward: opening + deliveries - forecast demand, capped at capacity.

    Returns a frame with projected opening/closing per day and a `stockout` flag
    where projected closing hits zero. This drives the dashboard's primary chart.
    """
    raise NotImplementedError


def safety_stock(forecast_std: float,
                 lead_time_days: int | None = None,
                 z: float | None = None) -> float:
    """z * sigma * sqrt(lead_time). z=2.05 targets ~98% service level."""
    lt = lead_time_days or settings.lead_time_days
    zz = z if z is not None else settings.service_level_z
    return float(zz * forecast_std * (lt ** 0.5))


def reorder_point(expected_lead_time_demand: float, safety: float,
                  silo_capacity: float) -> float:
    """Reorder point, capped at capacity - you cannot hold more than the silo."""
    return float(min(expected_lead_time_demand + safety, silo_capacity))


def backtest_policy(panel: pd.DataFrame, forecasts: pd.DataFrame) -> dict[str, float]:
    """Simulate the reorder policy over the held-out period.

    Returns pour readiness %, mean silo utilisation, implied write-offs, and the
    same three under a reactive-ordering counterfactual for comparison.
    """
    raise NotImplementedError
