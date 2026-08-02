"""Cleaning and repair.

Two decisions documented in REPORTS/decisions.md drive this module:
  1. Missing dates: 90 series x 1,096 days = 98,640 possible rows, only 32,880
     present (33.3% coverage). Absent dates are treated as zero-activity days.
  2. Inventory ledger: 34.8% of raw rows exceed silo capacity because the source
     system never caps deliveries. The ledger is recomputed with a capacity ceiling
     and the excess recorded as rejected delivery volume.
"""

from __future__ import annotations

import pandas as pd

FILL_ZERO = ["planned_pour_tonnes", "consumed_tonnes", "deliveries_tonnes"]


def reindex_full_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Expand to the full (date x site_id x cement_type) grid.

    Zero-fills activity columns; forward-fills static site attributes and weather.
    TODO(phase-1): confirm the zero-activity interpretation before relying on this.
    """
    raise NotImplementedError


def repair_ledger(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute the inventory ledger per (site_id, cement_type), chronologically.

        opening_t  = closing_{t-1}
        received_t = min(deliveries_t, silo_capacity - opening_t + consumed_t)
        closing_t  = clip(opening_t + received_t - consumed_t, 0, silo_capacity)

    Adds `rejected_delivery_tonnes` so the discarded volume stays auditable.
    TODO(phase-1): implement and quantify total rejected volume by behavior.
    """
    raise NotImplementedError


def flag_censored_demand(df: pd.DataFrame) -> pd.DataFrame:
    """Add `was_constrained` where consumed < planned (39.7% of raw rows).

    consumed_tonnes is censored demand, not true demand - see WORKFLOW.md 1.3(c).
    """
    out = df.copy()
    out["was_constrained"] = (
        out.consumed_tonnes < out.planned_pour_tonnes - 1e-6
    ).astype(int)
    return out


def build_clean_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Full pipeline: reindex -> repair ledger -> flag censoring."""
    raise NotImplementedError
