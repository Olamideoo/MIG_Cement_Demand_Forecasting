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
USER appuser
EXPOSE 8050
CMD ["gunicorn", "--chdir", "DASHBOARD", "app:server", "--bind", "0.0.0.0:8050"]
