.PHONY: setup lint format clean-data pipeline features test api dashboard mlflow up down retrain deploy

setup:            ## create venv, install deps, install package editable
	python -m venv .venv
	.venv/bin/pip install -U pip
	.venv/bin/pip install -r requirements-dev.txt
	.venv/bin/pip install -e .
	.venv/bin/nbstripout --install

lint:             ## ruff over source, dashboard and tests
	ruff check src DASHBOARD tests

format:           ## autofix what ruff can
	ruff check --fix src DASHBOARD tests

clean-data:       ## raw SQLite -> validated clean panel
	python -m mig_cement.data.preprocess

features:         ## clean panel -> feature matrix
	python -m mig_cement.features.build

pipeline:         ## full run: SQLite -> clean -> features -> train -> save model
	python -m mig_cement.pipeline

pipeline-dry:     ## same, but score without writing artefacts
	python -m mig_cement.pipeline --dry-run

api:              ## run the FastAPI service
	uvicorn mig_cement.api.main:app --reload --port 8000

dashboard:        ## run the Streamlit operations dashboard
	streamlit run DASHBOARD/app.py

test:             ## API contract tests
	pytest tests/ -q

mlflow:           ## tracking server + UI at http://localhost:5000
	mlflow server --backend-store-uri sqlite:///mlruns/mlflow.db \
	              --artifacts-destination mlruns/artifacts --serve-artifacts \
	              --host 127.0.0.1 --port 5000 --workers 1

up:               ## build and start api + dashboard
	docker compose -f docker/docker-compose.yml up --build

down:             ## stop and remove the stack
	docker compose -f docker/docker-compose.yml down

retrain:          ## run the training job in a container, writing to MODELS/
	docker compose -f docker/docker-compose.yml run --rm model

deploy:           ## phase 8 - refuses to ship a dirty tree
	@test -z "$$(git status --porcelain)" || (echo "ERROR: working tree dirty"; exit 1)
	@test "$$(git rev-parse --abbrev-ref HEAD)" = "main" || (echo "ERROR: not on main"; exit 1)
	@echo "TODO(phase-8): build, tag with $$(git rev-parse --short HEAD), push to ECR, update ECS"
