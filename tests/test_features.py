"""Leakage tests.

Eyeballing a 40-column feature matrix for lookahead is unreliable. These assert
it mechanically: a feature at time t must not change when a future value changes.
"""

from __future__ import annotations

import pytest

from mig_cement.features import build


def test_lags_are_shifted(leaky_series):
    out = build.add_lags(leaky_series)
    # the lag-1 feature on the final row equals the PREVIOUS value, not the spike
    assert out["y_lag_1"].iloc[-1] == pytest.approx(10.0)


def test_rolling_excludes_current_row(leaky_series):
    out = build.add_rolling(leaky_series)
    # the spike is the last value; the rolling mean on that row must not include it
    assert out["y_roll_mean_7"].iloc[-1] == pytest.approx(10.0)


def test_future_value_does_not_change_past_features(leaky_series):
    base = build.add_rolling(leaky_series)["y_roll_mean_7"].iloc[:-1]
    mutated = leaky_series.copy()
    mutated.loc[mutated.index[-1], "y"] = 99999.0
    after = build.add_rolling(mutated)["y_roll_mean_7"].iloc[:-1]
    assert (base.fillna(-1) == after.fillna(-1)).all()


def test_rain_threshold_feature(clean_panel):
    panel = clean_panel.copy()
    panel.loc[0, "rain_mm"] = 50.0
    out = build.add_weather(panel)
    assert out.loc[0, "rain_heavy"] == 1
    assert out.loc[1, "rain_heavy"] == 0
