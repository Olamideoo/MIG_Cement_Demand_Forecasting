"""Feature construction — the weekly modelling panel.

CRITICAL: this is the single implementation used by BOTH training and inference.
`pipeline.py` calls it to build the training panel; `api/predict.py` calls it to build
the panel for a forecast request. Never copy this logic into either.

The frozen model uses 8 features, all derived from the pour schedule and site
attributes. Weather, inventory state and target lags are deliberately excluded —
none are knowable at an 8-week ordering horizon. See notebook 04 for the ablation.
"""

from __future__ import annotations

import pandas as pd

TARGET = "y"
SCHEDULE_COLS = ["planned_pour_tonnes", "planned_pour_next_7",
                 "planned_pour_next_14", "days_since_planned_pour"]
CATEGORICAL_COLS = ["site_id", "region", "behavior"]
FEATURES = SCHEDULE_COLS + ["silo_capacity"] + CATEGORICAL_COLS

WEEKLY_AGG = {
    TARGET: "sum",
    "planned_pour_tonnes": "sum",
    "planned_pour_next_7": "last",
    "planned_pour_next_14": "last",
    "days_since_planned_pour": "first",
    "silo_capacity": "first",
    "region": "first",
    "behavior": "first",
}


def add_schedule_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Forward-looking pour features.

    The schedule is known in advance, so summing ahead uses no information the
    business would not already hold when placing an order.
    """
    out = daily.sort_values(["site_id", "date"]).copy()

    def forward_sum(s: pd.Series, window: int) -> pd.Series:
        return s.iloc[::-1].rolling(window, min_periods=1).sum().iloc[::-1]

    for window in (7, 14):
        out[f"planned_pour_next_{window}"] = (
            out.groupby("site_id")["planned_pour_tonnes"]
               .transform(lambda s, w=window: forward_sum(s, w))
        )

    pour_day = out["date"].where(out["planned_pour_tonnes"] > 0)
    last_pour = pour_day.groupby(out["site_id"]).ffill()
    out["days_since_planned_pour"] = (out["date"] - last_pour).dt.days
    return out


def to_weekly(daily: pd.DataFrame, drop_partial: bool = True
              ) -> tuple[pd.DataFrame, int]:
    """Aggregate to one row per site-week, labelled by the Monday start.

    Partial weeks are dropped by default. `resample` opens and closes each series
    with buckets holding fewer than 7 days, and those average ~50 t against ~166 t
    for a full week — they read as a demand collapse rather than a short bucket.
    """
    df = daily.copy()
    df["week"] = df["date"].dt.to_period("W-SUN").dt.start_time

    agg = {k: v for k, v in WEEKLY_AGG.items() if k in df.columns}
    weekly = df.groupby(["site_id", "week"], as_index=False).agg(agg)
    weekly["n_days"] = df.groupby(["site_id", "week"]).size().values

    partial = weekly.n_days < 7
    if drop_partial:
        weekly = weekly[~partial]

    weekly = (weekly.drop(columns="n_days")
              .rename(columns={"week": "date"})
              .sort_values(["site_id", "date"]).reset_index(drop=True))

    for col in CATEGORICAL_COLS:
        if col in weekly.columns:
            weekly[col] = weekly[col].astype(str)

    return weekly, int(partial.sum())


def build_weekly_panel(clean_daily: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Cleaned daily panel -> weekly per-site modelling panel."""
    return to_weekly(add_schedule_features(clean_daily))
