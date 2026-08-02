# MIG Cement Demand Forecasting

Forecasting daily cement demand across Midlands Infrastructure Group's 30 construction
sites, and turning those forecasts into reorder decisions.

**Full execution plan: [WORKFLOW.md](WORKFLOW.md)** — phases, owners, data-quality
findings and acceptance criteria. Read section 1.3 before writing any modelling code.

## Quickstart

```bash
make setup                 # venv + deps + editable install + nbstripout
cp .env.example .env
make test
make api                   # http://localhost:8000/docs
make dashboard             # http://localhost:8050
```

Windows: run `make` from Git Bash or WSL, or execute the commands in the Makefile
directly.

Full stack in containers:

```bash
make up                    # api + dashboard + mlflow
```

## Layout

| Path | Contents |
|---|---|
| `NOTEBOOKS/` | Analysis and justification, 01–06, mapped to the brief's seven steps |
| `src/mig_cement/` | The system: ingestion, features, models, inventory, API |
| `DASHBOARD/` | Plotly Dash application (calls the API, never loads models itself) |
| `DATA/raw/` | Source SQLite database (committed) |
| `DATA/interim`, `DATA/processed` | Derived data (gitignored) |
| `tests/` | Validator, leakage, inventory and API contract tests |
| `docker/`, `infra/` | Containers and AWS infrastructure |
| `REPORTS/` | Figures, decisions log, final documentation |

Notebooks import from `src/`. Never the reverse.

## Data

`DATA/raw/MIG_Cement_Records.db` — 30 sites × 3 cement types, 2022-01-01 to
2024-12-31, 32,880 rows across `Sites`, `CementTypes` and `Operations`.

Four findings that shape the approach:

- Series are **sparse** — 33.3% of the possible (date × site × type) grid is present
- **34.8%** of rows breach silo capacity; the raw inventory ledger needs repair
- **39.7%** of rows show consumption truncated by stock — the target is censored
- There is **no meaningful calendar seasonality**; signal lives in planned pours (r=0.78)

## Status

Scaffolding. Modules carry signatures, docstrings and `NotImplementedError`.
Phase 0 of WORKFLOW.md.
