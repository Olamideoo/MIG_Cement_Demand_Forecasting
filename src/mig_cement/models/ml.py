"""Global tree-based models over the pooled panel.

Given weak seasonality and a strong exogenous signal, this is expected to win.
site_id / cement_type / region / behavior enter as categoricals.
"""

from __future__ import annotations

import pandas as pd

CATEGORICALS = ["site_id", "cement_type", "region", "behavior"]


def build_random_forest(**kwargs):
    raise NotImplementedError


def build_gradient_boosting(**kwargs):
    """LightGBM or XGBoost. Supports quantile objectives for the prediction
    intervals that Phase 4 safety stock needs."""
    raise NotImplementedError


def fit(model, X: pd.DataFrame, y: pd.Series):
    raise NotImplementedError
