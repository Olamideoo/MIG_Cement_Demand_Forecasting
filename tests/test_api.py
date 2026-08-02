"""API contract tests. These are what stop the Dash app and the service drifting
apart while two people work in parallel."""

from __future__ import annotations

from fastapi.testclient import TestClient

from mig_cement.api.main import app

client = TestClient(app)


def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_forecast_rejects_bad_horizon():
    r = client.post("/forecast", json={
        "site_id": "SITE_001", "cement_type": "CEM_I", "horizon_weeks": 99,
    })
    assert r.status_code == 422


def test_forecast_rejects_missing_field():
    r = client.post("/forecast", json={"site_id": "SITE_001"})
    assert r.status_code == 422


def test_openapi_exposes_contract():
    spec = client.get("/openapi.json").json()
    assert "/forecast" in spec["paths"]
    assert "/inventory/alerts" in spec["paths"]
