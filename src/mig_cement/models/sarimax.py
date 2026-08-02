"""SARIMAX with exogenous regressors, fitted per (site_id, cement_type) series.

90 series. Expect non-convergence on some - catch, log, and fall back rather than
letting one bad fit kill the run. Seasonal terms are unlikely to help (see audit).
"""

from __future__ import annotations

import pandas as pd

EXOG = ["planned_pour_tonnes", "rain_mm", "avg_temp_c"]


def fit_series(y: pd.Series, exog: pd.DataFrame, order=(1, 0, 1)):
    """Fit one series. Returns a fitted statsmodels result or None on failure."""
    raise NotImplementedError


def fit_all(panel: pd.DataFrame) -> dict[tuple[str, str], object]:
    """Fit every series. Cache results - refitting 90 SARIMAX models is slow."""
    raise NotImplementedError


def forecast(fitted, exog_future: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Point forecast plus prediction interval."""
    raise NotImplementedError
