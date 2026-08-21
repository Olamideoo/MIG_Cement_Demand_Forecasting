# Training job. Runs once and exits - it is not a service.
#
#   docker compose run --rm model
#
# Reads the committed SQLite database, trains the random forest, scores the
# hold-out and writes three artefacts:
#
#   MODELS/rf_demand_forecaster.joblib        the model the API serves
#   MODELS/rf_demand_forecaster_meta.json     features, cutoff, metrics, versions
#   DATA/processed/test_forecasts.parquet     hold-out forecasts + actuals
#   mlruns/                                   the MLflow record of the run
#
# Those must land on a mounted volume, not inside the container, or the artefact
# dies with the process. compose mounts ../MODELS and ../DATA/processed for
# exactly this reason.
#
# The API image bakes a copy of MODELS/ so it can deploy to ECS as one unit; the
# local mount shadows that copy, so retraining here is picked up on the next API
# restart without rebuilding anything.

FROM python:3.11-slim AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.11-slim
RUN useradd -m -u 1000 appuser
COPY --from=builder /install /usr/local
WORKDIR /app

COPY src/ ./src/
COPY pyproject.toml ./
RUN pip install --no-cache-dir --no-deps -e .

# The database is committed (3 MB) so the image is self-contained: it can train
# with no volumes attached, which is what makes the CI smoke test possible.
COPY DATA/raw/ ./DATA/raw/

# Written to at runtime, so they must exist and be writable by appuser before
# the switch below - a mount will cover them, but an unmounted run still works.
RUN mkdir -p MODELS DATA/processed DATA/interim mlruns && chown -R appuser:appuser /app

USER appuser

# Same entrypoint as `make pipeline`. One command, one definition of training.
CMD ["python", "-m", "mig_cement.pipeline"]
