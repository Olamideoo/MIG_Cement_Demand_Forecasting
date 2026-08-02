"""The validators must FAIL on bad data.

A validator that always passes is the classic silent bug - you write it, it never
fires, and you assume the data is clean. These tests inject known-bad rows.
"""

from __future__ import annotations

import pandas as pd
import pytest

from mig_cement.data import validate


def test_clean_panel_passes(clean_panel):
    assert validate.check_schema(clean_panel) == []
    assert len(validate.check_primary_key(clean_panel)) == 0
    assert len(validate.check_balance(clean_panel)) == 0
    assert len(validate.check_capacity(clean_panel)) == 0


def test_missing_column_detected(clean_panel):
    assert "rain_mm" in validate.check_schema(clean_panel.drop(columns=["rain_mm"]))


def test_duplicate_key_detected(clean_panel):
    broken = pd.concat([clean_panel, clean_panel.iloc[[0]]], ignore_index=True)
    assert len(validate.check_primary_key(broken)) == 2


def test_negative_consumption_detected(clean_panel):
    broken = clean_panel.copy()
    broken.loc[0, "consumed_tonnes"] = -5.0
    assert len(validate.check_non_negative(broken)) == 1


def test_balance_breach_detected(clean_panel):
    broken = clean_panel.copy()
    broken.loc[0, "closing_inventory_tonnes"] += 5.0
    assert len(validate.check_balance(broken)) == 1


def test_capacity_breach_detected(clean_panel):
    broken = clean_panel.copy()
    broken.loc[0, "closing_inventory_tonnes"] = 999.0
    assert len(validate.check_capacity(broken)) == 1


def test_validate_clean_raises_on_bad_data(clean_panel):
    broken = clean_panel.copy()
    broken.loc[0, "closing_inventory_tonnes"] = 999.0
    with pytest.raises(ValueError):
        validate.validate_clean(broken)
