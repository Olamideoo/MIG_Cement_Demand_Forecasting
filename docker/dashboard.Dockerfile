# Streamlit operations dashboard.
#
# Previously this ran `gunicorn --chdir DASHBOARD app:server`, which was correct
# when the dashboard was Dash. Streamlit is not a WSGI application and exposes no
# `server` attribute, so gunicorn cannot serve it at all - it needs its own
# runner and its own port.
#
# No MODELS/ here. With API_BASE_URL set, forecasts, alerts and the policy
# summary come over HTTP, so this image never loads the model. DATA/raw is still
# needed: the baseline and performance pages read 33k-row panels locally, which
# would be far slower pushed through JSON than read from SQLite.

FROM python:3.11-slim AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.11-slim
RUN useradd -m -u 1000 appuser
COPY --from=builder /install /usr/local
WORKDIR /app

COPY src/ ./src/
COPY DASHBOARD/ ./DASHBOARD/
COPY pyproject.toml ./
RUN pip install --no-cache-dir --no-deps -e .

COPY DATA/raw/ ./DATA/raw/

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

# Streamlit's own health path. Cheap, and it does not touch the API.
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health').status==200 else 1)"

# --server.address is required: Streamlit binds to localhost by default, which
# inside a container means the port maps to nothing from outside.
# Usage stats are disabled because a container should not phone home silently.
CMD ["streamlit", "run", "DASHBOARD/app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
