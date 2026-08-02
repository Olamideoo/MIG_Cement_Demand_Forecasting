"""Feature construction.

CRITICAL: this is the single implementation used by BOTH models/train.py and
api/predict.py. Never copy it into the API - see WORKFLOW.md section 2.3.

It must work on the full panel and on one (site_id, cement_type) series, so
every operation is grouped by the series key and every rolling statistic is
shifted by 1 to avoid leakage.

The justification for each feature lives in NOTEBOOKS/03_Feature_Engineering.ipynb.
"""

from __future__ import annotations

import pandas as pd

KEY = ["site_id", "cement_type"]
TARGET = "y"

LAGS = [1, 7, 14, 28]
WINDOWS = [7, 14, 28]
RAIN_HEAVY_MM = 10.0  # threshold set in notebook 03; do not hardcode elsewhere


def add_lags(df: pd.DataFrame, col: str = TARGET) -> pd.DataFrame:
    """Lagged target values. Safe by construction - lag >= 1."""
    out = df.copy()
    g = out.groupby(KEY, observed=True)[col]
    for lag in LAGS:
        out[f"{col}_lag_{lag}"] = g.shift(lag)
    return out


def add_rolling(df: pd.DataFrame, col: str = TARGET) -> pd.DataFrame:
    """Rolling mean/std of the target. shift(1) BEFORE rolling - the window must
    exclude the current row or the target leaks into its own feature."""
    out = df.copy()
    g = out.groupby(KEY, observed=True)[col]
    for w in WINDOWS:
        shifted = g.shift(1)
        out[f"{col}_roll_mean_{w}"] = shifted.groupby(
            [out[k] for k in KEY], observed=True
        ).transform(lambda s, w=w: s.rolling(w, min_periods=1).mean())
        out[f"{col}_roll_std_{w}"] = shifted.groupby(
            [out[k] for k in KEY], observed=True
        ).transform(lambda s, w=w: s.rolling(w, min_periods=2).std())
    return out


def add_weather(df: pd.DataFrame) -> pd.DataFrame:
    """Weather features. Rain is threshold-like rather than linear (notebook 03);
    raw correlation with consumption is only -0.18, temperature is negligible."""
    out = df.copy()
    out["rain_heavy"] = (out.rain_mm > RAIN_HEAVY_MM).astype(int)
    out["temp_below_5"] = (out.avg_temp_c < 5).astype(int)
    return out


def add_pour_schedule(df: pd.DataFrame) -> pd.DataFrame:
    """Planned pour features. planned_pour_tonnes is the strongest single
    predictor (r=0.78) and IS known ahead of time, so forward sums are legal."""
    raise NotImplementedError


def add_inventory_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cover days, silo utilisation, turnover. Uses opening inventory only -
    closing inventory of day t is not known when forecasting day t."""
    raise NotImplementedError


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar features. Note the audit found no meaningful weekday or monthly
    seasonality (23.2-23.9 t by weekday), so expect these to contribute little."""
    out = df.copy()
    out["dayofweek"] = out.date.dt.dayofweek
    out["month"] = out.date.dt.month
    out["is_weekend"] = (out.dayofweek >= 5).astype(int)
    return out


def build_features(df: pd.DataFrame, *, for_inference: bool = False) -> pd.DataFrame:
    """Full feature pipeline. Same call signature at train and serve time.

    Args:
        df: clean panel, one row per (date, site_id, cement_type), target in `y`.
        for_inference: when True, skip target-derived columns that are unavailable
            for future dates and expect them supplied by the recursive forecaster.
    """
    raise NotImplementedError


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Ordered feature list. Persist this with the model - column order must match
    between training and inference or predictions silently degrade."""
    drop = {TARGET, "date", "consumed_tonnes", "closing_inventory_tonnes"}
    return [c for c in df.columns if c not in drop]
