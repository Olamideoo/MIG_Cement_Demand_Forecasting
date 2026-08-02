"""SQLite -> pandas. The only module that talks to the raw database."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from mig_cement.config import settings


def connect(db_path: Path | None = None) -> sqlite3.Connection:
    return sqlite3.connect(db_path or settings.db_path)


def load_operations(db_path: Path | None = None) -> pd.DataFrame:
    """Operations fact table, 32,880 rows across 2022-01-01..2024-12-31."""
    with connect(db_path) as con:
        return pd.read_sql("SELECT * FROM Operations", con, parse_dates=["date"])


def load_sites(db_path: Path | None = None) -> pd.DataFrame:
    """30 sites: site_id, region, silo_capacity, behavior."""
    with connect(db_path) as con:
        return pd.read_sql("SELECT * FROM Sites", con)


def load_cement_types(db_path: Path | None = None) -> pd.DataFrame:
    with connect(db_path) as con:
        return pd.read_sql("SELECT * FROM CementTypes", con)


def load_panel(db_path: Path | None = None) -> pd.DataFrame:
    """Operations joined to site attributes. The standard entry point."""
    ops = load_operations(db_path)
    sites = load_sites(db_path)
    return ops.merge(sites[["site_id", "region", "behavior"]], on="site_id", how="left")
