"""Data quality gates.

Runtime validation, not tests. Each check returns the offending rows so the caller
can decide whether to raise or repair.
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
    """Required columns that are missing."""
    return [c for c in REQUIRED_COLUMNS if c not in df.columns]


def check_primary_key(df: pd.DataFrame) -> pd.DataFrame:
    """Rows duplicating (date, site_id). The brief names cement_type as part of
    the key, but the data is one row per site-day."""
    return df[df.duplicated(["date", "site_id"], keep=False)]


def check_ledger_continuity(df: pd.DataFrame, tol: float = BALANCE_TOL) -> pd.DataFrame:
    """opening_t == closing_{t-1} within each site, in date order."""
    d = df.sort_values(["site_id", "date"]).copy()
    prev_close = d.groupby("site_id").closing_inventory_tonnes.shift(1)
    broken = (d.opening_inventory_tonnes - prev_close).abs() > tol
    return d[broken.fillna(False)]


def check_non_negative(df: pd.DataFrame) -> pd.DataFrame:
    """Rows with negative consumption, deliveries or inventory."""
    cols = ["consumed_tonnes", "deliveries_tonnes",
            "opening_inventory_tonnes", "closing_inventory_tonnes"]
    return df[(df[cols] < 0).any(axis=1)]


def check_balance(df: pd.DataFrame, tol: float = BALANCE_TOL) -> pd.DataFrame:
    """opening + deliveries - consumed == closing. 2 breaches in the raw data."""
    resid = (df.opening_inventory_tonnes + df.deliveries_tonnes
             - df.consumed_tonnes - df.closing_inventory_tonnes)
    return df[resid.abs() > tol]


def check_capacity(df: pd.DataFrame) -> pd.DataFrame:
    """closing <= silo_capacity. 34.8% of raw rows breach this; must be zero
    after repair."""
    return df[df.closing_inventory_tonnes > df.silo_capacity]


def validate_raw(df: pd.DataFrame) -> dict[str, int]:
    """Summary counts for the raw panel. Nothing raises."""
    return {
        "missing_columns": len(check_schema(df)),
        "duplicate_keys": len(check_primary_key(df)),
        "negative_values": len(check_non_negative(df)),
        "balance_breaches": len(check_balance(df)),
        "capacity_breaches": len(check_capacity(df)),
        "ledger_discontinuities": len(check_ledger_continuity(df)),
    }


def validate_clean(df: pd.DataFrame) -> None:
    """Post-repair gate. Raises on any surviving defect.

    Balance is checked against `served_tonnes` when present, since that is what
    the repaired ledger actually issued.
    """
    if missing := check_schema(df):
        raise ValueError(f"missing columns: {missing}")

    checks: list[tuple[str, pd.DataFrame]] = [
        ("duplicate keys", check_primary_key(df)),
        ("negative values", check_non_negative(df)),
        ("capacity breaches", check_capacity(df)),
        ("ledger discontinuities", check_ledger_continuity(df)),
    ]
    if "served_tonnes" in df.columns:
        resid = (
            df.opening_inventory_tonnes
            + df.received_tonnes
            - df.served_tonnes
            - df.closing_inventory_tonnes
        )
        checks.append(("balance breaches", df[resid.abs() > BALANCE_TOL]))
    else:
        checks.append(("balance breaches", check_balance(df)))

    for name, offenders in checks:
        if len(offenders):
            raise ValueError(f"{len(offenders)} {name} remain after cleaning")
