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

FROM python:3.11-slim AS builder
WORKDIR /build

COPY requirements.txt requirements-train.txt ./
RUN pip install --no-cache-dir --prefix=/install -r requirements-train.txt

FROM python:3.11-slim
RUN useradd -m -u 1000 appuser
COPY --from=builder /install /usr/local
WORKDIR /app

COPY src/ ./src/
COPY pyproject.toml ./
RUN pip install --no-cache-dir --no-deps -e .

COPY DATA/raw/ ./DATA/raw/

RUN mkdir -p MODELS DATA/processed DATA/interim mlruns && chown -R appuser:appuser /app

USER appuser

CMD ["python", "-m", "mig_cement.pipeline"]
