"""Contract tests for the forecasting API.

Scope is deliberately the HTTP surface, because that surface is what the
dashboard and any future client depend on. The tests run through FastAPI's
`TestClient`, so no server, no port and no network - they exercise the real
application object in-process.

Two things shape the design:

1. **Derived data may be absent.** The model itself is committed, but
   `DATA/processed/` is gitignored, so a fresh clone has no
   `test_forecasts.parquet` and no `per_site_sigma.parquet` until the pipeline
   has been run. Tests that need them skip cleanly via `skipif` on the file's
   existence, rather than failing and teaching everyone to ignore red builds.
   The check is automatic rather than a marker, so the suite adapts to whatever
   is present instead of depending on the right flag being passed.

2. **Startup is expensive.** Loading the model and rebuilding the weekly panel
   from SQLite takes several seconds, so the client is module-scoped and that
   cost is paid once for the whole file.

Run:
    pytest tests/api_test.py -v
    pytest tests/api_test.py -q -rs                 # -rs lists what skipped and why
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from mig_cement.api.main import app
from mig_cement.api.predict import forecaster
from mig_cement.config import settings

MODEL_PATH = settings.models_dir / "rf_demand_forecaster.joblib"
FORECASTS_PATH = settings.processed_dir / "test_forecasts.parquet"

needs_model = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason=f"{MODEL_PATH.name} not found - run `python -m mig_cement.pipeline`")


@pytest.fixture(scope="module")
def client():
    """One client for the file. The `with` block runs FastAPI's lifespan, which
    is what loads the model and warms the panel."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def unloaded(client):
    """Temporarily strip the model to exercise the degraded path.

    Restores in teardown even if the test fails - a leaked None here would make
    every later test in the module fail for the wrong reason.
    """
    model, panel = forecaster.model, forecaster._panel
    forecaster.model = None
    try:
        yield client
    finally:
        forecaster.model, forecaster._panel = model, panel


# --------------------------------------------------------------------------- #
# liveness
# --------------------------------------------------------------------------- #
def test_health_is_reachable(client):
    """Always 200 - the probe reports state, it does not judge it."""
    assert client.get("/health").status_code == 200


def test_health_does_not_run_the_model(client):
    """A probe polled every few seconds must not do real work.

    Ten calls should be near-instant. If someone later makes /health call
    predict() this fails long before it reaches a load balancer.
    """
    import time
    start = time.perf_counter()
    for _ in range(10):
        client.get("/health")
    assert (time.perf_counter() - start) / 10 < 0.1


@needs_model
def test_health_reports_a_loaded_model(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["sites_servable"] == 30
    assert body["model_version"] not in (None, "", "unloaded")


def test_health_stays_200_when_degraded(unloaded):
    """A missing model volume is a deploy problem, not a crash.

    Returning 503 here would make an orchestrator kill and reschedule the task
    in a loop instead of leaving it up to be diagnosed.
    """
    response = unloaded.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert response.json()["model_loaded"] is False


def test_business_endpoints_refuse_to_serve_when_degraded(unloaded):
    """/health is lenient; the endpoints that need the model are not."""
    assert unloaded.post("/forecast", json={"horizon_weeks": 8}).status_code == 503
    assert unloaded.post("/inventory", json={"horizon_weeks": 8}).status_code == 503


# --------------------------------------------------------------------------- #
# request validation - no model required, pydantic rejects before the handler
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("horizon", [0, -1, 9, 99])
def test_horizon_outside_the_trained_range_is_rejected(client, horizon):
    """The model was trained and validated on 8 weeks. Asking for 20 would be
    extrapolating past anything measured, so it is refused at the door."""
    r = client.post("/forecast", json={"horizon_weeks": horizon})
    assert r.status_code == 422
    assert r.json()["detail"][0]["loc"] == ["body", "horizon_weeks"]


def test_horizon_must_be_an_integer(client):
    assert client.post("/forecast", json={"horizon_weeks": "eight"}).status_code == 422


def test_empty_body_uses_defaults(client):
    """`{}` means the whole estate over the full horizon - the dashboard relies
    on this, so it must not become a 422."""
    assert client.post("/forecast", json={}).status_code in (200, 503)


# --------------------------------------------------------------------------- #
# forecast
# --------------------------------------------------------------------------- #
@needs_model
def test_forecast_returns_every_site_by_default(client):
    body = client.post("/forecast", json={"horizon_weeks": 8}).json()
    assert len(body["sites"]) == 30
    assert all(len(s["forecast"]) == 8 for s in body["sites"])
    assert body["horizon_weeks"] == 8
    assert body["model_version"]


@needs_model
def test_forecast_for_one_site(client):
    body = client.post("/forecast",
                       json={"site_id": "SITE_001", "horizon_weeks": 3}).json()
    assert len(body["sites"]) == 1
    assert body["sites"][0]["site_id"] == "SITE_001"
    assert len(body["sites"][0]["forecast"]) == 3


@needs_model
def test_forecast_shape_and_ordering(client):
    """Every point carries the fields the dashboard reads, horizons run 1..n in
    order, and the interval brackets the prediction."""
    body = client.post("/forecast",
                       json={"site_id": "SITE_001", "horizon_weeks": 8}).json()
    points = body["sites"][0]["forecast"]

    assert [p["horizon_week"] for p in points] == list(range(1, 9))
    for p in points:
        assert p["lower"] <= p["predicted_tonnes"] <= p["upper"]
        assert p["lower"] >= 0            # demand cannot be negative
        assert p["predicted_tonnes"] >= 0


@needs_model
def test_forecast_carries_actuals_where_the_week_has_happened(client):
    """The servable window sits after the training cutoff but inside the recorded
    data, so these weeks have already occurred and the truth is known.

    Returning it lets a client score the forecast without a second data source -
    which is what the dashboard's performance page relies on.
    """
    body = client.post("/forecast", json={"horizon_weeks": 8}).json()
    points = [p for s in body["sites"] for p in s["forecast"]]

    observed = [p for p in points if p["actual_tonnes"] is not None]
    assert len(observed) == len(points) == 240
    assert all(p["actual_tonnes"] >= 0 for p in observed)


@needs_model
def test_actuals_in_the_response_reproduce_the_holdout_mape(client):
    """Guards the field's meaning, not just its presence.

    A plausible-looking column that is subtly the wrong series would still pass a
    not-null check. Scoring it against the model card catches that.
    """
    body = client.post("/forecast", json={"horizon_weeks": 8}).json()
    points = [p for s in body["sites"] for p in s["forecast"]]
    df = pd.DataFrame([p for p in points if p["actual_tonnes"]])

    mape = float(np.mean(np.abs(
        (df.actual_tonnes - df.predicted_tonnes) / df.actual_tonnes)))
    expected = forecaster.metadata["holdout_metrics"]["MAPE"]
    assert abs(mape - expected) < 0.001, f"{mape:.4f} vs model card {expected}"


@needs_model
def test_forecast_dates_are_weekly_and_ascending(client):
    body = client.post("/forecast",
                       json={"site_id": "SITE_001", "horizon_weeks": 8}).json()
    dates = pd.to_datetime([p["date"] for p in body["sites"][0]["forecast"]])
    assert dates.is_monotonic_increasing
    assert set(dates.to_series().diff().dropna().dt.days) == {7}


@needs_model
def test_forecast_serves_only_unseen_weeks(client):
    """Forecasting a week the model was fitted on would report memorised values
    as predictions. The served window must start after the training cutoff."""
    cutoff = pd.Timestamp(forecaster.metadata["trained_on"]["to"])
    body = client.post("/forecast", json={"horizon_weeks": 8}).json()
    earliest = min(pd.Timestamp(p["date"])
                   for s in body["sites"] for p in s["forecast"])
    assert earliest > cutoff


def test_unknown_site_is_a_404_not_a_500(client):
    r = client.post("/forecast", json={"site_id": "SITE_999"})
    assert r.status_code in (404, 503)
    if r.status_code == 404:
        assert "SITE_999" in r.json()["detail"]


# --------------------------------------------------------------------------- #
# predict - single site-week, with an optional what-if on the pour
# --------------------------------------------------------------------------- #
@needs_model
def test_predict_scores_a_week_as_it_stands(client):
    body = client.post("/predict",
                       json={"site_id": "SITE_001", "week": "2024-10-07"}).json()
    assert body["site_id"] == "SITE_001"
    assert body["week"] == "2024-10-07"
    assert body["lower"] <= body["predicted_tonnes"] <= body["upper"]
    assert body["baseline_tonnes"] is None       # nothing was overridden
    assert body["in_training_range"] is True
    assert body["warning"] is None


@needs_model
def test_predict_with_no_week_uses_the_first_servable_one(client):
    """The dashboard and curl users should not have to know the cutoff date."""
    body = client.post("/predict", json={"site_id": "SITE_001"}).json()
    cutoff = pd.Timestamp(forecaster.metadata["trained_on"]["to"])
    assert pd.Timestamp(body["week"]) > cutoff


@needs_model
def test_predict_matches_forecast_for_the_same_week(client):
    """Two endpoints, one model - they must not disagree about the same row."""
    p = client.post("/predict",
                    json={"site_id": "SITE_001", "week": "2024-10-07"}).json()
    f = client.post("/forecast",
                    json={"site_id": "SITE_001", "horizon_weeks": 1}).json()
    point = f["sites"][0]["forecast"][0]

    assert point["date"] == p["week"]
    assert abs(point["predicted_tonnes"] - p["predicted_tonnes"]) < 0.01


@needs_model
def test_an_override_moves_the_prediction_and_reports_the_baseline(client):
    """Inside the training range the model responds to the pour, and the caller
    can see the size of the effect without making a second request."""
    body = client.post("/predict", json={"site_id": "SITE_001",
                                         "week": "2024-10-07",
                                         "planned_pour_tonnes": 350}).json()
    assert body["planned_pour_tonnes"] == 350
    assert body["baseline_tonnes"] is not None
    assert body["predicted_tonnes"] != body["baseline_tonnes"]
    assert body["in_training_range"] is True


@needs_model
def test_an_override_beyond_the_training_range_is_flagged(client):
    """The important one.

    A random forest averages training leaves, so past the largest pour ever
    recorded the prediction stops moving. Left unflagged it would return a
    confident number for an input it cannot answer, so the response must say so.
    """
    body = client.post("/predict", json={"site_id": "SITE_001",
                                         "week": "2024-10-07",
                                         "planned_pour_tonnes": 900}).json()
    assert body["in_training_range"] is False
    assert body["warning"] and "never poured more than" in body["warning"]


@needs_model
def test_the_range_check_is_per_site_not_per_estate(client):
    """The bug this replaced.

    SITE_009 has never poured above 118 t while the estate maximum is 403 t. An
    estate-wide check waves 300 t through as in-range, which is exactly the
    plausible-looking answer the flag exists to challenge.
    """
    body = client.post("/predict", json={"site_id": "SITE_009",
                                         "week": "2024-10-07",
                                         "planned_pour_tonnes": 300}).json()
    site_max = body["site_pour_range"][1]
    estate_max = float(forecaster._panel.planned_pour_tonnes.max())

    assert site_max < 300 < estate_max          # the case that used to slip past
    assert body["in_training_range"] is False
    assert "SITE_009" in body["warning"]


@needs_model
def test_the_site_range_is_reported_and_brackets_the_real_pour(client):
    body = client.post("/predict",
                       json={"site_id": "SITE_009", "week": "2024-10-07"}).json()
    lo, hi = body["site_pour_range"]
    assert lo <= body["planned_pour_tonnes"] <= hi


@needs_model
def test_an_interval_is_returned_only_without_an_override(client):
    """Sigma measures error against the real schedule. Wrapping a hypothetical in
    that band would claim a precision nobody has measured, so it is withheld."""
    real = client.post("/predict",
                       json={"site_id": "SITE_001", "week": "2024-10-07"}).json()
    assert real["lower"] is not None and real["upper"] is not None
    assert real["lower"] <= real["predicted_tonnes"] <= real["upper"]

    whatif = client.post("/predict", json={"site_id": "SITE_001",
                                           "week": "2024-10-07",
                                           "planned_pour_tonnes": 350}).json()
    assert whatif["lower"] is None and whatif["upper"] is None


@needs_model
def test_predictions_saturate_beyond_the_training_range(client):
    """Demonstrates why the flag exists: ten times the pour, same answer.

    If a future model could extrapolate this would fail, which is the right
    outcome - the warning would then be wrong and should be revisited.
    """
    trained_max = float(forecaster._panel.planned_pour_tonnes.max())
    big = client.post("/predict", json={"site_id": "SITE_001",
                                        "week": "2024-10-07",
                                        "planned_pour_tonnes": trained_max * 2}).json()
    huge = client.post("/predict", json={"site_id": "SITE_001",
                                         "week": "2024-10-07",
                                         "planned_pour_tonnes": trained_max * 10}).json()
    assert big["predicted_tonnes"] == huge["predicted_tonnes"]


@needs_model
def test_predict_rejects_a_week_outside_the_servable_window(client):
    """The message names the range rather than just refusing."""
    r = client.post("/predict", json={"site_id": "SITE_001", "week": "2019-01-07"})
    assert r.status_code == 404
    assert "Available" in r.json()["detail"]


def test_predict_rejects_a_negative_pour(client):
    assert client.post("/predict", json={"site_id": "SITE_001",
                                         "planned_pour_tonnes": -5}
                       ).status_code == 422


def test_predict_unknown_site_is_a_404(client):
    r = client.post("/predict", json={"site_id": "SITE_999"})
    assert r.status_code in (404, 503)


# --------------------------------------------------------------------------- #
# inventory
# --------------------------------------------------------------------------- #
@needs_model
def test_inventory_returns_all_sites_with_a_summary(client):
    body = client.post("/inventory", json={"horizon_weeks": 8}).json()
    assert len(body["alerts"]) == 30
    assert set(body["summary"]) >= {"pour_readiness", "stockout_days",
                                    "write_offs_t", "total_ordered_t"}
    assert 0.0 <= body["summary"]["pour_readiness"] <= 1.0


@needs_model
def test_every_alert_is_internally_consistent(client):
    """Physical constraints the simulator must never violate: stock fits in the
    silo, is not negative, and severity is one of the three known values."""
    alerts = client.post("/inventory", json={"horizon_weeks": 8}).json()["alerts"]
    for a in alerts:
        assert 0 <= a["current_inventory_tonnes"] <= a["silo_capacity_tonnes"]
        assert a["severity"] in {"red", "amber", "green"}
        assert a["suggested_order_tonnes"] >= 0


@needs_model
def test_severity_is_not_uniform(client):
    """A run where every site is green usually means the severity rule broke,
    not that the estate is perfect."""
    alerts = client.post("/inventory", json={"horizon_weeks": 8}).json()["alerts"]
    assert len({a["severity"] for a in alerts}) > 1


@needs_model
def test_inventory_filters_by_region(client):
    body = client.post("/inventory", json={"region": "North"}).json()
    assert {a["region"] for a in body["alerts"]} == {"North"}
    assert 0 < len(body["alerts"]) < 30


@needs_model
def test_inventory_filters_by_site(client):
    body = client.post("/inventory",
                       json={"site_ids": ["SITE_001", "SITE_013"]}).json()
    assert {a["site_id"] for a in body["alerts"]} == {"SITE_001", "SITE_013"}


@needs_model
def test_site_and_region_intersect(client):
    """Documented behaviour: the two filters narrow together rather than adding.
    SITE_001 is in North, so it survives; against South it does not."""
    keep = client.post("/inventory",
                       json={"site_ids": ["SITE_001"], "region": "North"})
    drop = client.post("/inventory",
                       json={"site_ids": ["SITE_001"], "region": "South"})
    assert keep.status_code == 200 and len(keep.json()["alerts"]) == 1
    assert drop.status_code == 404


def test_unknown_region_is_a_404(client):
    r = client.post("/inventory", json={"region": "Atlantis"})
    assert r.status_code in (404, 503)


# --------------------------------------------------------------------------- #
# regression - the API and the training pipeline must not drift apart
# --------------------------------------------------------------------------- #
@needs_model
@pytest.mark.skipif(not FORECASTS_PATH.exists(),
                    reason="test_forecasts.parquet not found - run the pipeline")
def test_api_reproduces_the_pipeline_holdout(client):
    """The strongest test in the file.

    `pipeline.py` scores the hold-out and writes test_forecasts.parquet; the API
    predicts the same weeks live from the database. If feature construction,
    column order or the servable window ever drift between training and serving,
    these two stop agreeing - and this is the only place that would notice.

    Tolerance is 0.01 because the API rounds to 2 dp for transport.
    """
    body = client.post("/forecast", json={"horizon_weeks": 8}).json()
    api = pd.DataFrame([{"site_id": s["site_id"], "date": p["date"],
                         "api": p["predicted_tonnes"]}
                        for s in body["sites"] for p in s["forecast"]])

    ref = pd.read_parquet(FORECASTS_PATH)
    ref["date"] = pd.to_datetime(ref["date"]).dt.date.astype(str)

    merged = api.merge(ref[["site_id", "date", "forecast_tonnes"]],
                       on=["site_id", "date"], how="inner")

    assert len(merged) == len(ref), "API did not cover every hold-out site-week"
    assert np.abs(merged.api - merged.forecast_tonnes).max() < 0.01


@needs_model
def test_served_features_match_the_trained_features(client):
    """Guards the train/serve contract directly.

    `Forecaster.load` refuses to start if the artefact's feature order differs
    from `features.build.FEATURES`. This asserts the guard is actually holding,
    so a silently retrained model with reordered columns cannot slip through.
    """
    from mig_cement.features.build import FEATURES
    assert forecaster.feature_columns == FEATURES
    assert forecaster.metadata["features"] == FEATURES


# --------------------------------------------------------------------------- #
# documentation
# --------------------------------------------------------------------------- #
def test_openapi_exposes_exactly_the_expected_endpoints(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert set(paths) == {"/health", "/forecast", "/predict", "/inventory"}


@needs_model
@pytest.mark.parametrize("endpoint", ["/forecast", "/predict", "/inventory"])
def test_every_documented_example_actually_works(client, endpoint):
    """Swagger prefills its request box from these. An example that 422s is
    worse than none, because it teaches the reader the wrong shape."""
    spec = client.get("/openapi.json").json()
    examples = (spec["paths"][endpoint]["post"]["requestBody"]
                ["content"]["application/json"]["examples"])
    assert examples, f"{endpoint} has no request examples"

    for name, example in examples.items():
        r = client.post(endpoint, json=example["value"])
        assert r.status_code == 200, f"{endpoint} example {name!r} -> {r.status_code}"
