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


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAPE over non-zero actuals only. Reports NaN if nothing is left."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    m = np.abs(y_true) > 1e-9
    if not m.any():
        return np.nan
    return float(np.mean(np.abs((y_true[m] - y_pred[m]) / y_true[m])))


def evaluate(y_true, y_pred, y_train=None, season: int = 1) -> dict[str, float]:
    """Full metric set. WAPE is primary; MAPE is only meaningful on aggregates."""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    out = {
        "WAPE": wape(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "bias": bias(y_true, y_pred),
        "MAPE_nonzero": mape(y_true, y_pred),
        "pct_zero_actual": float((np.abs(y_true) < 1e-9).mean()),
    }
    if y_train is not None:
        out["MASE"] = mase(y_true, y_pred, np.asarray(y_train, float), season)
    return out


def time_split(df: pd.DataFrame, train_end: str, val_end: str,
               date_col: str = "date") -> dict[str, pd.DataFrame]:
    """Chronological train / validation / test split. Never random."""
    d = pd.to_datetime(df[date_col])
    return {
        "train": df[d <= train_end],
        "val": df[(d > train_end) & (d <= val_end)],
        "test": df[d > val_end],
    }
