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
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "mig_cement.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
