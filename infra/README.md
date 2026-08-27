# Infrastructure

Single EC2 instance running the compose stack behind nginx.

**Live:** <http://13.59.81.166/> · [API docs](http://13.59.81.166/api/docs) · [MLflow](http://13.59.81.166/mlflow/)

| | |
|---|---|
| Instance | `t3.small` (2 GB), Ubuntu 26.04, `us-east-2b` |
| Address | Elastic IP `13.59.81.166` |
| Open ports | 80 (nginx), 22 (own IP only) |
| Routing | `/` → dashboard:8501 · `/api/` → api:8000 · `/mlflow/` → mlflow:5000 |

2 GB is the practical minimum — a `t3.micro` was OOM-killed during image builds.

## Setup

```bash
sudo apt update && sudo apt install -y docker.io docker-compose-v2 git nginx
sudo systemctl enable --now docker
sudo usermod -aG docker $USER && exit          # log back in
```

```bash
git clone https://github.com/Olamideoo/MIG_Cement_Demand_Forecasting.git
cd MIG_Cement_Demand_Forecasting
printf 'PUBLIC_HOST=13.59.81.166\n' > docker/.env
```

`docker/.env`, not `./.env` — Compose reads it from the compose file's directory, and a
root `.env` is silently ignored.

Then create `docker/docker-compose.override.yml` (server-only, not in git):

```yaml
services:
  api:
    command: ["uvicorn", "mig_cement.api.main:app",
              "--host", "0.0.0.0", "--port", "8000", "--root-path", "/api"]
  mlflow:
    command:
      - mlflow
      - server
      - --backend-store-uri
      - sqlite:////app/mlruns/mlflow.db
      - --artifacts-destination
      - /app/mlruns/artifacts
      - --serve-artifacts
      - --host
      - "0.0.0.0"
      - --port
      - "5000"
      - --allowed-hosts
      - "mlflow:5000,localhost:5000,127.0.0.1:5000,13.59.81.166"
      - --cors-allowed-origins
      - "http://13.59.81.166"
      - --workers
      - "1"
```

`--root-path` makes Swagger work under `/api`. The MLflow flags are its Host and Origin
checks, which reject the proxy and the model container by default.

## nginx

`/etc/nginx/sites-available/default`, replacing the contents:

```nginx
server {
    listen 80;
    server_name 13.59.81.166;

    client_max_body_size 100M;
    proxy_read_timeout 300s;
    proxy_set_header Host              $host;
    proxy_set_header X-Real-IP         $remote_addr;
    proxy_set_header X-Forwarded-For   $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;

    location /api/ {
        proxy_pass http://127.0.0.1:8000/;
    }

    location /mlflow/ {
        proxy_pass http://127.0.0.1:5000/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade    $http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade    $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 86400;
    }
}
```

```bash
sudo nginx -t && sudo systemctl restart nginx && sudo systemctl enable nginx
```

The `Upgrade`/`Connection` headers on `location /` are required — Streamlit holds a
WebSocket open, and without them the page hangs on "Please wait…".

## Deploying

```bash
cd ~/MIG_Cement_Demand_Forecasting && git pull
C=docker/docker-compose.yml; O=docker/docker-compose.override.yml

docker compose -f $C -f $O up -d --build mlflow
docker compose -f $C -f $O run --rm --build model     # train, ~90s
docker compose -f $C -f $O up -d --build
docker compose -f $C -f $O ps
```

Both `-f` files, every time — without the override the services revert to their
committed commands.

Train before starting the API. `DATA/processed/` is gitignored, and without
`per_site_sigma.parquet` the API falls back to one estate-wide error estimate, silently
changing every reorder point.

## Checks

```bash
curl -s -o /dev/null -w '%{http_code}\n' http://localhost/          # 200
curl -s -o /dev/null -w '%{http_code}\n' http://localhost/api/health
curl -s localhost:8000/health | python3 -m json.tool | head
```

The dashboard sidebar should read `Source API`.

| Symptom | Cause |
|---|---|
| Dashboard hangs on "Please wait…" | WebSocket headers missing |
| `/api/docs` "Try it out" fails | override file not passed |
| MLflow `INTERNAL_ERROR` or 403 | address missing from `--allowed-hosts` / `--cors-allowed-origins` |
| Reorder points identical across sites | trained after the API started, or not at all |
| Build killed partway | out of RAM — `free -h` |

## Not built

No TLS (needs a domain), no authentication, no backups of `mlruns/` or `MODELS/`, no
monitoring, no rollback. The managed alternative would be ECR + ECS Fargate behind an
ALB, S3 for artefacts, RDS for the tracking store, CloudWatch for logs and alarms.
