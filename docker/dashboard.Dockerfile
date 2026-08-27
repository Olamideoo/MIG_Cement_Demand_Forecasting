# Streamlit operations dashboard.

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

# Streamlit's own health check.
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health').status==200 else 1)"


CMD ["streamlit", "run", "DASHBOARD/app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
