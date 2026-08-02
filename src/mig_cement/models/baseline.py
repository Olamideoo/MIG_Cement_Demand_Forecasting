"""Baselines. The bar every model must clear.

The one that matters is `planned_pour_baseline` - MIG already has planned pour
figures, so a model that cannot beat them adds no value. Report this prominently.
"""

from __future__ import annotations

import pandas as pd


def naive(df: pd.DataFrame) -> pd.Series:
    """Last observed value carried forward."""
    raise NotImplementedError


def seasonal_naive(df: pd.DataFrame, period: int = 7) -> pd.Series:
    """Value from `period` steps back. Expect weak performance - no seasonality."""
    raise NotImplementedError


def moving_average(df: pd.DataFrame, window: int = 28) -> pd.Series:
    raise NotImplementedError


def planned_pour_baseline(df: pd.DataFrame) -> pd.Series:
    """Forecast = planned_pour_tonnes. THE benchmark to beat."""
    return df["planned_pour_tonnes"]
