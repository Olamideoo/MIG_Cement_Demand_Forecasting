# MIG Cement Demand Forecasting

Forecasting daily cement demand across Midlands Infrastructure Group's 30 construction
sites, and turning those forecasts into reorder decisions.

**Full execution plan: [WORKFLOW.md](WORKFLOW.md)** — phases, owners, data-quality
findings and acceptance criteria. Read §1.3 before writing any modelling code.

## Quickstart

```bash
make setup                 # venv + deps + editable install + nbstripout
cp .env.example .env
make clean-data            # raw SQLite -> DATA/interim/operations_clean.parquet
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
| `docker/`, `infra/` | Containers and AWS infrastructure |
| `REPORTS/` | Figures, decisions log, final documentation |

Notebooks import from `src/`. Never the reverse.

## Data

`DATA/raw/MIG_Cement_Records.db` — `Sites`, `CementTypes`, `Operations`,
2022-01-01 to 2024-12-31.

Four findings that shape the approach:

- The grain is **site-day and complete** — 30 sites × 1,096 days = 32,880 rows exactly,
  one cement type per site-day, one silo per site
- **34.8%** of raw rows breached silo capacity; the ledger is repaired by capping
  deliveries at available headroom
- **39.7%** of rows show consumption truncated by stock — the target is censored
- There is **no meaningful calendar seasonality**; signal lives in planned pours (r=0.78)

## Status

**Step 1 (Data Ingestion and Cleaning) complete.**

- `make clean-data` rebuilds `DATA/interim/operations_clean.parquet` from the raw database
- `NOTEBOOKS/01_Data_Ingestion_Cleaning.ipynb` carries the analysis with executed outputs
- Capacity breaches 11,439 → 0; consumption changed by −0.49%

Steps 2–7 are scaffolded: signatures, docstrings and `NotImplementedError`.
