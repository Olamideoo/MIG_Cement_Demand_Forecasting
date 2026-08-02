"""Shared fixtures. Tests use synthetic frames, never the real database -
a test that needs a 3 MB SQLite file is a slow integration test, not a unit test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def clean_panel() -> pd.DataFrame:
    """Small well-formed panel: 2 sites x 1 cement type x 10 days."""
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    rows = []
    for site, cap in [("SITE_001", 100.0), ("SITE_002", 200.0)]:
        opening = 50.0
        for d in dates:
            consumed, delivered = 10.0, 12.0
            closing = opening + delivered - consumed
            rows.append({
                "date": d, "site_id": site, "cement_type": "CEM_I",
                "planned_pour_tonnes": 10.0, "consumed_tonnes": consumed,
                "opening_inventory_tonnes": opening, "deliveries_tonnes": delivered,
                "closing_inventory_tonnes": closing, "rain_mm": 1.0,
                "avg_temp_c": 12.0, "silo_capacity": cap,
            })
            opening = closing
    return pd.DataFrame(rows)


@pytest.fixture
def leaky_series() -> pd.DataFrame:
    """One series with a large spike at the end - used to prove that rolling
    features at time t do not move when a FUTURE value changes."""
    dates = pd.date_range("2024-01-01", periods=20, freq="D")
    y = np.ones(20) * 10.0
    y[-1] = 1000.0
    return pd.DataFrame({
        "date": dates, "site_id": "SITE_001", "cement_type": "CEM_I", "y": y,
    })
