.PHONY: setup lint format clean-data pipeline features train api dashboard up down deploy

setup:            ## create venv, install deps, install package editable
	python -m venv .venv
	.venv/bin/pip install -U pip
	.venv/bin/pip install -r requirements.txt
	.venv/bin/pip install -e .
	.venv/bin/nbstripout --install

lint:
	ruff check src DASHBOARD
	black --check src DASHBOARD

format:
	ruff check --fix src DASHBOARD
	black src DASHBOARD

clean-data:       ## raw SQLite -> validated clean panel
	python -m mig_cement.data.preprocess

features:         ## clean panel -> feature matrix
	python -m mig_cement.features.build

pipeline:         ## full run: SQLite -> clean -> features -> train -> save model
	python -m mig_cement.pipeline

pipeline-dry:     ## same, but score without writing artefacts
	python -m mig_cement.pipeline --dry-run

train:            ## train + log to MLflow (MODEL=lightgbm)
	python -m mig_cement.models.train --model $(or $(MODEL),lightgbm)

api:
	uvicorn mig_cement.api.main:app --reload --port 8000

dashboard:
	python DASHBOARD/app.py

up:
	docker compose -f docker/docker-compose.yml up --build

down:
	docker compose -f docker/docker-compose.yml down

deploy:           ## phase 8 - refuses to ship a dirty tree
	@test -z "$$(git status --porcelain)" || (echo "ERROR: working tree dirty"; exit 1)
	@test "$$(git rev-parse --abbrev-ref HEAD)" = "main" || (echo "ERROR: not on main"; exit 1)
	@echo "TODO(phase-8): build, tag with $$(git rev-parse --short HEAD), push to ECR, update ECS"
