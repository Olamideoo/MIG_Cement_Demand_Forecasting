# Forecasting API. Serves /health, /forecast, /predict and /inventory.
#
# The model artefact is baked in rather than mounted. MODELS/*.joblib is
# gitignored, so an image that expected a volume would deploy to ECS with
# nothing to serve; baking it means the image tag *is* the model version and the
# task definition needs no shared storage. Locally, compose mounts ../MODELS on
# top of this copy, so `docker compose run --rm model` is picked up on the next
# API restart without a rebuild.

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

# The database is needed at runtime, not just for training: the service rebuilds
# the weekly feature panel from it at startup, using the same builder the
# pipeline used, which is what keeps training and serving from drifting.
COPY DATA/raw/ ./DATA/raw/

# per_site_sigma.parquet drives per-site safety stock. Without it the service
# silently falls back to one estate-wide sigma of 30.53 t in place of values
# ranging 2.6-60 t, quietly changing every reorder point.
COPY DATA/processed/ ./DATA/processed/

# Uppercase both sides. The build context is case-sensitive on Linux even when
# the developer's filesystem is not, and config.py resolves models_dir as
# REPO_ROOT/"MODELS" - which is /app/MODELS here.
COPY MODELS/ ./MODELS/

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Hits the liveness probe, which reports state without touching the model. A
# check that ran a forecast would load the service every 30s and time out under
# exactly the conditions it exists to detect.
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "mig_cement.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
