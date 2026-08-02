"""Inventory simulation is a pure deterministic function and easy to get subtly
wrong - order of operations on the capacity cap, off-by-one on lead time.

Hand-compute a short scenario and assert the code reproduces it.
"""

from __future__ import annotations

import pytest

from mig_cement.inventory import simulate


def test_safety_stock_scales_with_lead_time():
    a = simulate.safety_stock(10.0, lead_time_days=1, z=2.0)
    b = simulate.safety_stock(10.0, lead_time_days=4, z=2.0)
    assert b == pytest.approx(2 * a)  # sqrt(4) / sqrt(1)


def test_safety_stock_zero_variance():
    assert simulate.safety_stock(0.0, lead_time_days=3, z=2.05) == 0.0


def test_reorder_point_capped_at_capacity():
    assert simulate.reorder_point(500.0, 100.0, silo_capacity=300.0) == 300.0


def test_reorder_point_normal_case():
    assert simulate.reorder_point(50.0, 20.0, silo_capacity=300.0) == 70.0


@pytest.mark.skip(reason="implement in phase 4")
def test_silo_projection_hand_computed():
    """opening 100, capacity 200, demand 30/day, delivery 20 on day 2.
    Expected closing: d1 70, d2 60, d3 30, d4 0 (stockout flagged)."""
    ...
