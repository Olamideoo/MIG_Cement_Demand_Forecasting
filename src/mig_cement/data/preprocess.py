"""Cleaning and repair — Brief Step 1.

Grain is site-day: 30 sites x 1,096 consecutive days = 32,880 rows, no gaps, one
cement_type per site-day. The silo ledger runs per site (opening_t == closing_{t-1}
holds on 100% of rows at site grain, 42.7% at site-type grain).

Main defect: the source never caps deliveries at silo capacity, so 34.8% of rows
close above it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from mig_cement.config import settings

SITE_KEY = "site_id"
ACTIVITY_COLS = ["planned_pour_tonnes", "consumed_tonnes", "deliveries_tonnes"]
INVENTORY_COLS = ["opening_inventory_tonnes", "closing_inventory_tonnes"]

EXPECTED_SITES = 30
EXPECTED_DAYS = 1096


@dataclass
class CleaningReport:
    """Quantifies everything the cleaning pass changed."""

    rows_in: int = 0
    rows_out: int = 0
    sites: int = 0
    calendar_complete: bool = False
    negatives_clipped: dict[str, int] = field(default_factory=dict)
    capacity_breaches_before: int = 0
    capacity_breaches_after: int = 0
    rejected_delivery_tonnes: float = 0.0
    delivery_tonnes_before: float = 0.0
    consumption_tonnes_before: float = 0.0
    consumption_tonnes_after: float = 0.0
    induced_shortfall_rows: int = 0
    censored_rows: int = 0

    @property
    def rejected_delivery_pct(self) -> float:
        if not self.delivery_tonnes_before:
            return 0.0
        return 100 * self.rejected_delivery_tonnes / self.delivery_tonnes_before

    @property
    def consumption_change_pct(self) -> float:
        if not self.consumption_tonnes_before:
            return 0.0
        delta = self.consumption_tonnes_after - self.consumption_tonnes_before
        return 100 * delta / self.consumption_tonnes_before

    def summary(self) -> pd.Series:
        return pd.Series(
            {
                "rows in": self.rows_in,
                "rows out": self.rows_out,
                "sites": self.sites,
                "calendar complete": self.calendar_complete,
                "negatives clipped": sum(self.negatives_clipped.values()),
                "capacity breaches before": self.capacity_breaches_before,
                "capacity breaches after": self.capacity_breaches_after,
                "rejected deliveries (t)": round(self.rejected_delivery_tonnes, 1),
                "rejected deliveries (%)": round(self.rejected_delivery_pct, 2),
                "consumption change (%)": round(self.consumption_change_pct, 2),
                "rows with induced shortfall": self.induced_shortfall_rows,
                "censored rows (consumed < planned)": self.censored_rows,
            }
        )


def check_calendar(df: pd.DataFrame) -> dict[str, object]:
    """Assert completeness at site-day grain rather than filling gaps."""
    per_site = df.groupby(SITE_KEY).date.agg(["min", "max", "nunique", "size"])
    span = (per_site["max"] - per_site["min"]).dt.days + 1
    return {
        "n_sites": int(df[SITE_KEY].nunique()),
        "n_dates": int(df.date.nunique()),
        "complete": bool(
            (per_site["nunique"] == span).all() and (per_site["size"] == span).all()
        ),
        "duplicated_site_days": int(df.duplicated([SITE_KEY, "date"]).sum()),
        "multi_type_site_days": int((df.groupby([SITE_KEY, "date"]).size() > 1).sum()),
    }


def fix_negatives(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Clip negative activity and inventory to zero. None present in this extract."""
    out = df.copy()
    counts: dict[str, int] = {}
    for col in ACTIVITY_COLS + INVENTORY_COLS:
        n = int((out[col] < 0).sum())
        if n:
            out[col] = out[col].clip(lower=0)
        counts[col] = n
    return out, counts


def repair_ledger(df: pd.DataFrame, mode: str = "cap_deliveries") -> pd.DataFrame:
    """Recompute the inventory ledger per site, chronologically.

    mode="cap_deliveries":
        received_t  = min(deliveries_t, capacity - opening_t)
        available_t = opening_t + received_t
        served_t    = min(consumed_t, available_t)
        closing_t   = available_t - served_t
        opening_t+1 = closing_t

    mode="flag_only" leaves the recorded ledger untouched and only adds the
    diagnostic columns.

    `consumed_tonnes` is preserved as recorded; the repaired figure goes to
    `served_tonnes`, and rows where capping forced a shortfall are flagged.
    """
    if mode not in {"cap_deliveries", "flag_only"}:
        raise ValueError(f"unknown mode: {mode!r}")

    out = df.sort_values([SITE_KEY, "date"]).reset_index(drop=True).copy()

    if mode == "flag_only":
        out["received_tonnes"] = out["deliveries_tonnes"]
        out["rejected_delivery_tonnes"] = 0.0
        out["served_tonnes"] = out["consumed_tonnes"]
        out["induced_shortfall"] = 0
        return out

    n = len(out)
    opening = np.zeros(n)
    closing = np.zeros(n)
    received = np.zeros(n)
    rejected = np.zeros(n)
    served = np.zeros(n)

    deliveries = out["deliveries_tonnes"].to_numpy(dtype=float)
    consumed = out["consumed_tonnes"].to_numpy(dtype=float)
    capacity = out["silo_capacity"].to_numpy(dtype=float)
    recorded_open = out["opening_inventory_tonnes"].to_numpy(dtype=float)
    site = out[SITE_KEY].to_numpy()

    carry = 0.0
    for i in range(n):
        first_row_of_site = i == 0 or site[i] != site[i - 1]
        open_i = min(recorded_open[i], capacity[i]) if first_row_of_site else carry

        recv = min(deliveries[i], max(capacity[i] - open_i, 0.0))
        available = open_i + recv
        srv = min(consumed[i], available)

        opening[i] = open_i
        received[i] = recv
        rejected[i] = deliveries[i] - recv
        served[i] = srv
        closing[i] = available - srv
        carry = closing[i]

    out["opening_inventory_tonnes"] = opening
    out["received_tonnes"] = received
    out["rejected_delivery_tonnes"] = rejected
    out["served_tonnes"] = served
    out["closing_inventory_tonnes"] = closing
    out["induced_shortfall"] = (served < consumed - 1e-6).astype(int)
    return out


def flag_censored_demand(df: pd.DataFrame) -> pd.DataFrame:
    """Flag rows where consumption fell short of the planned pour (39.7%)."""
    out = df.copy()
    out["was_constrained"] = (
        out.consumed_tonnes < out.planned_pour_tonnes - 1e-6
    ).astype(int)
    out["unmet_tonnes"] = (out.planned_pour_tonnes - out.consumed_tonnes).clip(lower=0)
    return out


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Utilisation, cover days, and the target column `y`."""
    out = df.copy()
    out["silo_utilisation"] = out.closing_inventory_tonnes / out.silo_capacity
    out["cover_days"] = np.where(
        out.consumed_tonnes > 0,
        out.opening_inventory_tonnes / out.consumed_tonnes,
        np.nan,
    )
    out["y"] = out["consumed_tonnes"]
    return out


def build_clean_panel(
    df: pd.DataFrame, mode: str = "cap_deliveries"
) -> tuple[pd.DataFrame, CleaningReport]:
    """Full Step 1 pipeline. Returns the clean panel and a quantified report."""
    rep = CleaningReport(rows_in=len(df), sites=int(df[SITE_KEY].nunique()))
    rep.calendar_complete = bool(check_calendar(df)["complete"])
    rep.capacity_breaches_before = int(
        (df.closing_inventory_tonnes > df.silo_capacity).sum()
    )
    rep.delivery_tonnes_before = float(df.deliveries_tonnes.sum())
    rep.consumption_tonnes_before = float(df.consumed_tonnes.sum())

    out, rep.negatives_clipped = fix_negatives(df)
    out = repair_ledger(out, mode=mode)
    out = flag_censored_demand(out)
    out = add_derived(out)

    rep.capacity_breaches_after = int(
        (out.closing_inventory_tonnes > out.silo_capacity + 1e-6).sum()
    )
    rep.rejected_delivery_tonnes = float(out.rejected_delivery_tonnes.sum())
    rep.consumption_tonnes_after = float(out.served_tonnes.sum())
    rep.induced_shortfall_rows = int(out.induced_shortfall.sum())
    rep.censored_rows = int(out.was_constrained.sum())
    rep.rows_out = len(out)
    return out, rep


def run(mode: str = "cap_deliveries", out_path: Path | None = None) -> Path:
    """CLI entrypoint: raw SQLite -> validated clean panel parquet."""
    from mig_cement.data import load, validate

    panel = load.load_panel()
    print("raw diagnostics:", validate.validate_raw(panel))

    clean, report = build_clean_panel(panel, mode=mode)
    print()
    print(report.summary().to_string())

    validate.validate_clean(clean)

    dest = out_path or (settings.interim_dir / "operations_clean.parquet")
    dest.parent.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(dest, index=False)
    print(f"\nwrote {len(clean):,} rows -> {dest}")
    return dest


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Step 1 — data ingestion and cleaning")
    p.add_argument("--mode", default="cap_deliveries",
                   choices=["cap_deliveries", "flag_only"])
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    run(args.mode, args.out)
