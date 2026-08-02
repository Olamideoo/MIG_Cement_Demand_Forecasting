"""Metrics and backtesting.

MAPE is undefined at zero and 12.2% of observed rows are zero, so WAPE and MASE
are the primary metrics. MAPE is reported only on weekly aggregates, where it is
well defined - see WORKFLOW.md 1.3(e).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted absolute percentage error. Primary metric."""
    denom = np.abs(y_true).sum()
    return float(np.abs(y_true - y_pred).sum() / denom) if denom else np.nan


def mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray,
         season: int = 1) -> float:
    """Mean absolute scaled error, scaled by in-sample naive error."""
    scale = np.abs(y_train[season:] - y_train[:-season]).mean()
    return float(np.abs(y_true - y_pred).mean() / scale) if scale else np.nan


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean error. Negative means systematic under-forecasting, which for an
    ordering system means stockouts."""
    return float(np.mean(y_pred - y_true))


def mape_weekly(df: pd.DataFrame) -> float:
    """MAPE on weekly aggregates only. Still drops ~9% zero weeks."""
    raise NotImplementedError


def evaluate(y_true, y_pred, y_train=None) -> dict[str, float]:
    """Full metric set for MLflow logging."""
    raise NotImplementedError


def rolling_origin_backtest(panel: pd.DataFrame, model, horizon_weeks: int = 8):
    """Expanding-window backtest with `horizon_weeks` ahead forecasts."""
    raise NotImplementedError
