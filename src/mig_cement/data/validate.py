"""Data quality gates.

Runtime validation - these run every time the pipeline runs, not just in tests.
Each check returns the offending rows so the caller can decide to raise or repair.
Thresholds and expectations come from the audit in WORKFLOW.md section 1.3.
"""

from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = [
    "date", "site_id", "cement_type", "planned_pour_tonnes", "consumed_tonnes",
    "opening_inventory_tonnes", "deliveries_tonnes", "closing_inventory_tonnes",
    "rain_mm", "avg_temp_c", "silo_capacity",
]
BALANCE_TOL = 0.01


def check_schema(df: pd.DataFrame) -> list[str]:
    """Return the names of any required columns that are missing."""
    return [c for c in REQUIRED_COLUMNS if c not in df.columns]


def check_primary_key(df: pd.DataFrame) -> pd.DataFrame:
    """Rows duplicating (date, site_id, cement_type). Expected: empty."""
    return df[df.duplicated(["date", "site_id", "cement_type"], keep=False)]


def check_non_negative(df: pd.DataFrame) -> pd.DataFrame:
    """Rows with negative consumption, deliveries or inventory. Expected: empty."""
    cols = ["consumed_tonnes", "deliveries_tonnes",
            "opening_inventory_tonnes", "closing_inventory_tonnes"]
    return df[(df[cols] < 0).any(axis=1)]


def check_balance(df: pd.DataFrame, tol: float = BALANCE_TOL) -> pd.DataFrame:
    """opening + deliveries - consumed == closing. Audit found 2 breaches."""
    resid = (df.opening_inventory_tonnes + df.deliveries_tonnes
             - df.consumed_tonnes - df.closing_inventory_tonnes)
    return df[resid.abs() > tol]


def check_capacity(df: pd.DataFrame) -> pd.DataFrame:
    """closing <= silo_capacity. Audit found 34.8% of raw rows breach this,
    almost entirely at 'conservative' sites. Must be near-zero after repair."""
    return df[df.closing_inventory_tonnes > df.silo_capacity]


def validate_raw(df: pd.DataFrame) -> dict[str, int]:
    """Summary counts for the raw panel. Nothing raises - raw data is known bad."""
    return {
        "missing_columns": len(check_schema(df)),
        "duplicate_keys": len(check_primary_key(df)),
        "negative_values": len(check_non_negative(df)),
        "balance_breaches": len(check_balance(df)),
        "capacity_breaches": len(check_capacity(df)),
    }


def validate_clean(df: pd.DataFrame) -> None:
    """Post-repair gate. Raises - a clean panel must satisfy all of these."""
    if missing := check_schema(df):
        raise ValueError(f"missing columns: {missing}")
    for name, offenders in [
        ("duplicate keys", check_primary_key(df)),
        ("negative values", check_non_negative(df)),
        ("balance breaches", check_balance(df)),
        ("capacity breaches", check_capacity(df)),
    ]:
        if len(offenders):
            raise ValueError(f"{len(offenders)} {name} remain after cleaning")
