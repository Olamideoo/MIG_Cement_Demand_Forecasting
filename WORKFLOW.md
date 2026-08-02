# MIG Cement Demand Forecasting — Execution Workflow

**Repo:** `github.com/Olamideoo/MIG_Cement_Demand_Forecasting`
**Team:** 2 collaborators (referred to below as **DS** and **MLE** — swap or split differently as you prefer)
**Scope:** SQLite → EDA → forecasting models → MLflow tracking → FastAPI service → Dash dashboard → Docker → AWS
**Date:** August 2026

---

## 1. Current State Audit

### 1.1 Repository

| Item | State |
|---|---|
| Git | `main` + `origin/olamide` branch, 1 commit ("Initial project setup") |
| `.gitignore` | Covers `.venv/`, `__pycache__`, `.ipynb_checkpoints`, `.env`, `.vscode` — modified but uncommitted |
| Notebooks | `01_Database_Exploration`, `02_Data_Preprocessing`, `03_EDA`, `07_Forecasting` in `NOTEBOOKS/`; `04_Feature_Engineering`, `05_Model_Training`, `06_Model_Evaluation` sitting loose in the repo root. **All 7 are 0 bytes** — not valid `.ipynb` JSON, they will not open in Jupyter. |
| `DASHBOARD`, `MODELS` | 0-byte **files**, not directories |
| `README.md` | Empty |
| `MIG_Cement_Records.db` | 3.1 MB, duplicated in repo root **and** `DATA/`; untracked and not gitignored |
| Missing entirely | `requirements.txt` / `environment.yml`, `src/`, MLflow config, API, Dockerfiles, IaC |

**Immediate fixes before any coding:**

1. Delete the 0-byte `DASHBOARD` and `MODELS` files; create them as directories with `.gitkeep`.
2. Delete all seven stub notebooks and recreate the six in §2.2 inside `NOTEBOOKS/`, as valid `.ipynb` with one empty cell. The current names don't map to the requirement flow (see §2.2) and two pairs duplicate each other.
3. Keep **one** copy of the DB, at `DATA/MIG_Cement_Records.db`. Delete the root copy.
4. Add `*.db`, `mlruns/`, `MODELS/*.pkl` to `.gitignore` — then decide storage (see §2.4). A 3 MB SQLite file is small enough to commit if you prefer reproducibility over cleanliness; if so, commit it once and never regenerate in place.
5. Commit `requirements.txt` pinned with `pip freeze` from a fresh venv.
6. Delete the stale `origin/olamide` branch or rebase it — start the team workflow from a clean `main`.

### 1.2 Database

Three tables, 2022-01-01 → 2024-12-31:

- **`Sites`** — 30 rows: `site_id`, `region` (North/South/East/West), `silo_capacity` (230–487 t), `behavior` (14 aggressive, 9 conservative, 7 chaotic)
- **`CementTypes`** — 3 rows: CEM_I, CEM_II, CEM_III
- **`Operations`** — 32,880 rows, 11 columns, no nulls, no duplicate `(date, site_id, cement_type)` keys

Note the schema differs from the brief's data dictionary: the brief describes a single flat `Cement_Demand` table; the actual DB is normalised into `Sites` / `CementTypes` / `Operations`, with `region` and `behavior` as **extra** site attributes not mentioned in the brief. Both are useful features. Update the brief's data dictionary to match reality.

### 1.3 Data quality findings — read this before modelling

These materially change the plan. Each is worth a section in `01_Data_Ingestion_Cleaning.ipynb`.

**(a) The grain is site-day, and the panel is complete.** *(corrected — the first version of this audit got it wrong)*

The original finding claimed the data was sparse: 30 sites × 3 cement types = 90 series, each with only ~365 rows over 1,096 days, therefore 33.3% coverage and a missing-date policy to decide. That reading was an artifact of assuming `(site_id, cement_type)` is the series key. It is not.

The arithmetic is exact: **30 sites × 1,096 consecutive days = 32,880 = the row count.** Every site has every date, with exactly one cement type recorded per site-day, rotating roughly evenly (CEM_I 33.2%, CEM_II 33.8%, CEM_III 33.1%). There are no gaps.

The inventory ledger confirms it. `opening_t == closing_{t-1}` holds on **100.0%** of rows at site grain and only **42.7%** at site-type grain — one silo per site, shared across cement grades, which also matches `silo_capacity` being constant per site.

Consequences:

- **No missing-date policy is needed.** Nothing to fill. `preprocess.check_calendar()` asserts completeness so the assumption fails loudly if a future extract breaks it.
- **Ledger repair runs per site**, chronologically, ignoring cement type.
- **Forecast at site-day grain with `cement_type` as a feature.** Splitting into 90 site×type series manufactures artificial gaps and destroys the ledger identity — it was the modelling approach implied by the original finding, and it would have been wrong.

Verified in `NOTEBOOKS/01_Data_Ingestion_Cleaning.ipynb` §3.

**(b) Inventory is physically impossible for conservative sites.**

| behavior | mean opening inventory | max | % rows where closing > silo capacity |
|---|---|---|---|
| aggressive | 13.5 t | 266 t | 0% |
| chaotic | 179.8 t | 922 t | 22.2% |
| conservative | 10,061 t | 20,646 t | **98.7%** |

**Resolved in Phase 1** — `preprocess.repair_ledger(mode="cap_deliveries")` caps deliveries at available headroom. Result: capacity breaches 11,439 → 0, at the cost of rejecting 19.2% of delivered volume but changing total consumption by only **−0.49%** and inducing a shortfall on 217 rows (0.7%). That asymmetry is the justification — the rejected volume was accumulation that was never going to be used. `mode="flag_only"` remains available if the team later concludes `silo_capacity` is the unreliable field.

Silo capacities are 230–487 t. Conservative sites accumulate 20,000+ tonnes in a 300-tonne silo — the data generator never caps inventory at capacity. Overall **11,439 of 32,880 rows (34.8%)** breach silo capacity. Options, in order of preference:

1. Re-cap inventory in preprocessing: recompute the inventory ledger site-by-site with `closing = min(opening + deliveries - consumed, silo_capacity)` and treat the excess as rejected/diverted deliveries. Keeps all sites, makes the inventory simulation and reorder-point work meaningful.
2. Model demand only, and run the inventory simulation on a re-derived ledger rather than the recorded one.
3. Exclude conservative sites — loses 30% of the panel, not recommended.

Whatever you choose, the 20% silo-utilisation and 30% write-off targets in the brief are only measurable against a repaired ledger.

**(c) The target is censored.**
In **39.7%** of rows `consumed_tonnes < planned_pour_tonnes` — consumption was truncated by available stock. So `consumed_tonnes` is *observed* demand, not *true* demand. Forecasting it directly bakes past stockouts into future forecasts and will systematically under-order. Recommended treatment: model `planned_pour_tonnes` (unconstrained demand signal) or an uncensoring-adjusted target, and use `consumed_tonnes` for accuracy reporting only. At minimum, add a `was_constrained` flag as a feature and discuss the bias in the report.

**(d) Almost no calendar seasonality.**
Mean daily consumption by weekday ranges 23.2–23.9 t; by month 23.0–24.5 t. There is no meaningful weekly or annual seasonal cycle at any grain. Consequences:

- SARIMA seasonal terms will contribute nothing. SARIMA**X** with exogenous regressors is still the right baseline — the signal is in the regressors, not the seasonality.
- The strongest predictor by far is `planned_pour_tonnes` (r = 0.78 with consumption). `rain_mm` is weakly negative (r = −0.18); `avg_temp_c` is negligible.
- This is a **regression-with-exogenous-features** problem more than a classical time-series problem. Frame the modelling accordingly, and be honest about it in the report — that's a finding, not a failure.

**(e) The MAPE ≤ 15% target is not achievable at daily site-level grain.**
MAPE is undefined when actuals are zero, and 12% of observed rows (more once you fill missing dates) are zero. Even excluding zeros, daily site-type volumes are small and noisy. Concretely:

- Report **WAPE** (weighted absolute percentage error) and **MASE** as primary metrics, with RMSE and bias.
- Report MAPE only at **weekly aggregated** grain, where zeros drop to ~9% of buckets, and at site-total / national level where it is well-defined.
- Set the ≤15% MAPE gate against the weekly, 8-week-horizon, site-level forecast — which is what the business actually orders on anyway. Get this renegotiation agreed with your "stakeholder" early and write it into the report as a scoping decision.

**(f) Minor.** 2 rows violate `opening + deliveries − consumed = closing` by more than 0.01 t. Trivially fixable; `validate.check_balance()` reports them on every run.

---

## 2. Target Repository Structure

```
MIG_Cement_Demand_Forecasting/
├── DATA/
│   ├── raw/MIG_Cement_Records.db
│   ├── interim/                # cleaned parquet
│   └── processed/              # feature matrices, train/val/test splits
├── NOTEBOOKS/                  # 01–06, exploration only, never imported by src/
│   ├── 01_Data_Ingestion_Cleaning.ipynb
│   ├── 02_EDA.ipynb
│   ├── 03_Feature_Engineering.ipynb
│   ├── 04_Model_Development.ipynb
│   ├── 05_Inventory_Simulation.ipynb
│   └── 06_Holdout_Validation.ipynb
├── src/mig_cement/
│   ├── __init__.py
│   ├── config.py               # pydantic-settings, reads .env
│   ├── data/
│   │   ├── load.py             # SQLite → DataFrame
│   │   ├── validate.py         # schema + balance + capacity checks
│   │   └── preprocess.py       # reindex, gap-fill, inventory ledger repair
│   ├── features/build.py       # lags, rolling stats, weather interactions
│   │                           # SAME function serves training and inference
│   ├── models/
│   │   ├── baseline.py         # seasonal naive, moving average
│   │   ├── sarimax.py
│   │   ├── ml.py               # RandomForest / LightGBM / XGBoost
│   │   ├── train.py            # entrypoint, logs to MLflow
│   │   └── evaluate.py         # WAPE/MASE/RMSE/MAPE, backtesting
│   ├── inventory/simulate.py   # silo projection, reorder points, safety stock
│   └── api/
│       ├── main.py             # FastAPI app
│       ├── schemas.py          # pydantic request/response
│       └── predict.py          # loads model from MLflow registry
├── DASHBOARD/
│   ├── app.py                  # Dash entrypoint
│   ├── pages/                  # overview, site drill-down, inventory, alerts
│   └── components/
├── MODELS/                     # local artifact cache (gitignored)
├── docker/
│   ├── api.Dockerfile
│   ├── dashboard.Dockerfile
│   └── docker-compose.yml      # api + dashboard + mlflow server
├── infra/                      # Terraform or CDK for AWS
├── REPORTS/                    # figures, final documentation
├── .env.example
├── requirements.txt            # or pyproject.toml + uv/poetry
├── Makefile                    # make setup / clean-data / train / serve / dashboard
└── README.md
```

**Rule:** notebooks import from `src/`, never the reverse. Anything a notebook proves out gets refactored into `src/` before the phase closes. This is what makes the FastAPI and Docker phases painless later.

### 2.2 Requirement steps → deliverables

The brief's seven execution steps map to six notebooks plus two code deliverables. Steps 6 and 7 are not analysis work — the dashboard is an application and deployment is infrastructure, neither belongs in a notebook.

| Brief step | Deliverable | Supporting `src/` module | Phase |
|---|---|---|---|
| 1. Data Ingestion & Cleaning | `01_Data_Ingestion_Cleaning.ipynb` | `data/load.py`, `data/validate.py`, `data/preprocess.py` | 1 |
| 2. Exploratory Data Analysis | `02_EDA.ipynb` | — | 2 |
| 3. Feature Engineering | `03_Feature_Engineering.ipynb` | `features/build.py` | 2 |
| 4. Model Development | `04_Model_Development.ipynb` | `models/baseline.py`, `sarimax.py`, `ml.py`, `train.py`, `evaluate.py` | 3 |
| 5. Inventory Simulation | `05_Inventory_Simulation.ipynb` | `inventory/simulate.py` | 4 |
| 6. Dashboard Application | `DASHBOARD/` (Dash app) | — | 6 |
| 7. Validation & Deployment | `06_Holdout_Validation.ipynb` **+** `src/api/`, `docker/`, `infra/` | `api/`, monitoring | 3, 5, 7–9 |

Three deliberate changes from the current stubs:

- **`01_Database_Exploration` + `02_Data_Preprocessing` merge into one.** Both are Step 1. Profiling and cleaning are iterative — you clean based on what profiling reveals, then re-profile. Splitting them across two notebooks means one imports the other's state.
- **`05_Model_Training` + `06_Model_Evaluation` merge into `04_Model_Development`.** Both are Step 4, and evaluation is not a separate pass — you compare candidates as you fit them. Split, the second notebook has to re-run the first to have anything to evaluate.
- **`07_Forecasting` becomes `06_Holdout_Validation`.** Generating forecasts is what the API does in production, not a notebook activity. What Step 7 actually asks for is validation against held-out data. Keeping this as its own notebook enforces a discipline worth having: **the test set is touched exactly once**, in one place, at the end. Model selection happens on validation data in `04`.

`05_Inventory_Simulation` is new — Step 5 currently has no notebook at all, despite being where the 98% pour-readiness, 20% utilisation and 30% write-off targets are actually evidenced.

### 2.3 Preventing train/serve skew

The FastAPI service must produce a feature vector identical to the one the model trained on. Two non-negotiables:

1. **One implementation.** `build_features()` accepts a DataFrame and returns features, parameterised to handle either a single site-type series or the full panel. `models/train.py` and `api/predict.py` both call it. Never a second copy inside the API.
2. **Version the transformer state with the model.** Log fitted encoders, scalers, and the exact feature column order as MLflow artifacts alongside the model, and have the API load them from the registry. If the API refits a categorical encoder independently, `CEM_II` can map to a different integer than it did in training — predictions degrade silently, with no error raised anywhere. This is the most common way MLflow projects fail in production.

**Serving pattern (still open — decide before Phase 5).** Step 7's "deploy forecasting pipeline" language suggests batch. Recommendation is hybrid: a scheduled job forecasts all 90 series and writes them to a table, the API serves those by lookup, and `POST /forecast/refresh` triggers on-demand recompute for a single site. Weekly-grain 8-week forecasts genuinely do not need real-time inference, and precomputing keeps heavy preprocessing off the request path — while the refresh endpoint still demonstrates the full live pipeline.

### 2.4 Data & artifact storage decisions (agree these on day 1)

| Asset | Local dev | Production |
|---|---|---|
| Raw DB | `DATA/raw/`, committed once (3 MB is acceptable) | S3 bucket, or RDS Postgres if you want to demo a real DB |
| Processed features | Parquet in `DATA/processed/`, gitignored | S3 |
| MLflow tracking | Local `mlruns/` or `mlflow server` via docker-compose | MLflow server on EC2/ECS with S3 artifact store + RDS backend |
| Trained models | MLflow Model Registry | Registry, pulled by API at container start |
| Secrets | `.env`, gitignored | AWS Secrets Manager / SSM Parameter Store |

---

## 3. Collaboration Model

**Branching:** trunk-based off `main`. Branch per unit of work: `feat/eda-weather-correlation`, `fix/inventory-ledger`, `chore/ci-setup`. PR into `main`, one review from the other collaborator, squash merge. Protect `main` (no direct pushes, one approving review). Delete the existing `origin/olamide` long-lived branch — long-lived personal branches with 2 people on shared notebooks is where merge hell comes from.

**Notebook conflicts:** notebooks diff terribly. Two mitigations, use both:

- One owner per notebook. Never edit the other person's notebook on your branch.
- Install `nbstripout` (`nbstripout --install`) so outputs are stripped on commit. Keep executed copies with outputs in `REPORTS/` if you need them for the write-up.

**Cadence:** twice-weekly 30-min sync — what merged, what's blocked, what changed in the shared interfaces (`src/features/build.py` signatures, the API schema, the model registry names). Keep a running decisions log in `REPORTS/decisions.md`; every judgment call in §1.3 goes in it.

**Split of ownership:**

- **DS** — data cleaning, EDA, feature engineering, model development, evaluation, inventory simulation logic, final report
- **MLE** — repo scaffolding, MLflow infrastructure, FastAPI, Dash, Docker, CI/CD, AWS, monitoring

Both review each other's PRs. The handoff points are the two contracts named in Phase 0.

---

## 4. Phased Workflow

Estimates assume two people working part-time; compress freely.

### Phase 0 — Foundation (Week 1) · owner: MLE, with DS pairing

| # | Task | Done when |
|---|---|---|
| 0.1 | Fix repo hygiene per §1.1 (dirs, notebooks, single DB, .gitignore) | `git status` clean, all notebooks open in Jupyter |
| 0.2 | Create venv, `requirements.txt`, `Makefile`, `.env.example` | `make setup` works on both machines from scratch |
| 0.3 | Scaffold `src/mig_cement/` package with `pip install -e .` | `import mig_cement` works from a notebook |
| 0.4 | Branch protection on `main`, PR template, `nbstripout` | Direct push to `main` rejected |
| 0.5 | `make lint` target, run locally before opening any PR | Both collaborators can run it; convention documented in README |
| 0.6 | **Agree the two contracts** (below) and write to `REPORTS/decisions.md` | Both sign off |

**Contract 1 — the feature matrix.** One row per `(date, site_id, cement_type)`, target column named `y`, all features numeric, no leakage. DS builds it, MLE consumes it. Frozen by end of Phase 2.
**Contract 2 — the API response.** `POST /forecast` takes `{site_id, cement_type, horizon_weeks}`, returns `{forecast: [{date, predicted_tonnes, lower, upper}], model_version, generated_at}`. Frozen by end of Phase 0 so Dash and API can be built in parallel against a stub.

### Phase 1 — Data understanding & cleaning (Week 2) · owner: DS

Brief Step 1 · `01_Data_Ingestion_Cleaning.ipynb` → `src/data/`

| # | Task | Status |
|---|---|---|
| 1.1 | Profile all three tables; confirm/refute each finding in §1.3 independently | **done** — §1.3(a) refuted and corrected |
| 1.2 | Establish the true grain and confirm calendar completeness | **done** — site-day, complete, no fill needed |
| 1.3 | Repair the inventory ledger — capacity cap, quantify rejected delivery volume | **done** — 11,439 breaches → 0, 19.2% of deliveries rejected |
| 1.4 | `data/validate.py`: schema, PK uniqueness, non-negativity, balance, capacity ceiling, ledger continuity | **done** |
| 1.5 | Write cleaned panel to `DATA/interim/operations_clean.parquet` | **done** — 32,880 rows, `make clean-data` |
| 1.6 | Ledger repair verified against hand-computed scenarios in notebook 01 | **done** |

**Exit criteria:** cleaned parquet reproducible from raw DB by one command; every validator passes on it; every rejected/modified row is counted and explained in the notebook. **Met.**

### Phase 2 — EDA & feature engineering (Week 3) · owner: DS

Brief Steps 2–3 · `02_EDA.ipynb` → `03_Feature_Engineering.ipynb` → `src/features/build.py`

| # | Task |
|---|---|
| 2.1 | Demand distributions by site, region, cement type, behavior; identify heterogeneity that argues for per-site vs. global models |
| 2.2 | Decompose the (weak) seasonality formally; ACF/PACF at daily and weekly grain to justify lag choices |
| 2.3 | Weather sensitivity: is rain's effect threshold-like (pours cancelled above X mm) rather than linear? Test it |
| 2.4 | Quantify the planned-vs-actual gap by behavior and by inventory position — this is the censoring story |
| 2.5 | Features: lags (1,7,14,28), rolling mean/std (7,14,28), planned pour lead-ahead sums, days-since-last-pour, rain flags, temp bins, site behavior/region/capacity, inventory cover days, calendar |
| 2.6 | **Leakage audit** — every feature computable at forecast time using only information available then. Rolling stats must be shift-1. Write this down explicitly |
| 2.7 | Emit Contract-1 feature matrix to `DATA/processed/` |

**Exit criteria:** feature matrix frozen and versioned; leakage audit documented; MLE unblocked to build training pipeline.

### Phase 3 — Modelling & MLflow (Weeks 4–5) · owner: DS on models, MLE on tracking infra

Brief Step 4 · `04_Model_Development.ipynb` → `src/models/` · final hold-out run lands in `06_Holdout_Validation.ipynb` (Brief Step 7)

| # | Task |
|---|---|
| 3.1 | **Split strategy first.** Time-based: train 2022-01→2024-06, validation 2024-07→2024-09, test 2024-10→2024-12 (held out, touched once). Rolling-origin backtest with 8-week horizons for honest error bars |
| 3.2 | Baselines: naive, seasonal naive, 28-day moving average, and "forecast = planned pour". **The last one is the real benchmark** — if your model can't beat planned pours, the project has no value. Say so up front |
| 3.3 | SARIMAX per series with exog regressors (planned pour, rain, temp). 90 series — script it, cache fits, expect some non-convergence and handle it |
| 3.4 | Global ML: RandomForest, then LightGBM/XGBoost, on the pooled panel with site/type as categoricals. Almost certainly the winner given weak seasonality |
| 3.5 | Optional stretch: quantile regression or conformal intervals for prediction intervals — needed for safety stock in Phase 4 |
| 3.6 | MLflow: every run logs params, metrics (WAPE/MASE/RMSE/MAPE-weekly/bias), the feature list, the split spec, and the model artifact. Tag runs by model family |
| 3.7 | Model selection on validation only; register winner to MLflow Model Registry as `mig-cement-forecaster`, promote to Staging |
| 3.8 | Single test-set evaluation → the headline numbers for the report |
| 3.9 | Error analysis: where does it fail? By behavior, by site size, by horizon week 1 vs week 8, by zero-vs-nonzero days |

**Exit criteria:** registered model beating the planned-pour benchmark on WAPE; test-set metrics recorded; MLflow runs reproducible by run ID.

### Phase 4 — Inventory optimisation (Week 6) · owner: DS

Brief Step 5 · `05_Inventory_Simulation.ipynb` → `src/inventory/simulate.py`

| # | Task |
|---|---|
| 4.1 | Forward-simulate silo levels from forecast + scheduled deliveries + opening position, per site-type |
| 4.2 | Safety stock from forecast uncertainty: `z × σ_forecast × √lead_time`, with the service level set to hit the 98% pour-readiness target. State the lead-time assumption explicitly (the data has none — pick a value, justify it, sensitivity-test it) |
| 4.3 | Reorder point = expected lead-time demand + safety stock, capped by silo capacity |
| 4.4 | Backtest the policy on held-out data: simulated stockout rate, mean utilisation, implied write-offs. **This is where you evidence the 98% / 20% / 30% targets** — a forecast alone doesn't prove any of them |
| 4.5 | Compare against a "current practice" counterfactual (reactive ordering) to quantify the improvement |

**Exit criteria:** simulated policy hits ≥98% pour readiness on test period, with utilisation and write-off deltas quantified against the counterfactual.

### Phase 5 — FastAPI service (Week 7) · owner: MLE

| # | Task |
|---|---|
| 5.1 | `POST /forecast` per Contract 2; `POST /forecast/batch`; `GET /health`; `GET /model/info` (version, registry stage, trained-at) |
| 5.2 | Load model from MLflow Registry at startup (lifespan handler), not per request |
| 5.3 | Pydantic validation: unknown `site_id` → 422 with a useful message; horizon bounds 1–8 weeks |
| 5.4 | `GET /inventory/alerts` returning current reorder recommendations by site |
| 5.5 | Structured logging (request ID, latency, model version), and log predictions for later drift monitoring |
| 5.7 | OpenAPI docs reviewed — this is a deliverable in itself |

**Exit criteria:** `uvicorn` serves real forecasts locally; `/docs` exercises every endpoint successfully; latency < 500 ms for a single-site 8-week request.

### Phase 6 — Dash dashboard (Week 7–8, parallel with 5) · owner: MLE, design input from DS

| # | Task |
|---|---|
| 6.1 | Multi-page layout: **Overview** (national demand, forecast vs actual, KPI tiles), **Site Drill-down** (per-site forecast with intervals, inventory projection), **Inventory & Alerts** (reorder table sorted by urgency, silo utilisation heatmap), **Model Performance** (accuracy by site/horizon, drift indicators) |
| 6.2 | Dashboard calls the FastAPI service — do not re-implement prediction in the Dash app |
| 6.3 | Filters: site, region, cement type, date range, forecast horizon |
| 6.4 | Alert logic surfaced clearly: red = projected stockout before next delivery, amber = below reorder point, green = healthy |
| 6.5 | Caching (`flask-caching`) so the 30-site view doesn't hammer the API |
| 6.6 | Export to CSV for the procurement team |

**Exit criteria:** operations manager can answer "which sites do I order for this week, and how much" in under 30 seconds on the dashboard.

### Phase 7 — Containerisation (Week 8) · owner: MLE

| # | Task |
|---|---|
| 7.1 | `docker/api.Dockerfile` — multi-stage, slim base, non-root user, healthcheck |
| 7.2 | `docker/dashboard.Dockerfile` — same pattern |
| 7.3 | `docker-compose.yml` — api + dashboard + mlflow server + (optional) postgres backend; shared network, env from `.env` |
| 7.4 | Image size discipline: `.dockerignore`, no notebooks/data in images, target < 1 GB |
| 7.5 | `docker compose up` reproduces the whole stack from a clean clone |

**Exit criteria:** full stack runs from `docker compose up` on the other collaborator's machine with no manual steps.

### Phase 8 — AWS deployment (Weeks 9–10) · owner: MLE

Pick one path and commit to it — don't half-build two.

**Path A — ECS Fargate (recommended for a portfolio project).**
ECR for images → ECS Fargate services for API and Dash → ALB with path routing (`/api/*` → API, `/*` → Dash) → S3 for MLflow artifacts + processed data → RDS Postgres (or SQLite-on-EFS, cheaper) for the MLflow backend → Secrets Manager → CloudWatch logs and alarms. Terraform in `infra/`. Costs roughly $40–80/month; tear down when not demoing.

**Path B — Lambda + API Gateway for the API, App Runner for Dash.** Cheaper at rest, cold-start pain, container size limits. Fine if budget matters more than latency.

| # | Task |
|---|---|
| 8.1 | IAM: least-privilege task roles; a dedicated deploy user with ECR push + ECS update only, credentials in `.env` (never committed) |
| 8.2 | Terraform (or CDK) for VPC, ECR, ECS, ALB, S3, RDS, Secrets. `terraform apply` from zero |
| 8.3 | `make deploy`: build, tag with the git SHA, push to ECR, update the ECS service, wait for stable. Run manually from a clean `main` only |
| 8.4 | HTTPS via ACM; restrict dashboard access (Cognito or at minimum basic auth) — do not leave it open |
| 8.5 | CloudWatch dashboard + alarms on 5xx rate, task health, prediction latency |
| 8.6 | Cost guardrail: budget alert, and a documented teardown command |

**Exit criteria:** public URL serving the dashboard, backed by the deployed API and registered model; `make deploy` from a clean `main` ships a new version reproducibly, and the image SHA on the running task matches the commit it came from.

### Phase 9 — Monitoring, docs, handover (Week 10) · owner: both

| # | Task |
|---|---|
| 9.1 | Drift monitoring: log predictions, join to actuals when they arrive, compute rolling WAPE; alarm when it exceeds the agreed threshold — this is the brief's "trigger model retraining" requirement |
| 9.2 | Scheduled retraining: EventBridge → ECS task running `src/models/train.py`, registering a new version (promote manually, not automatically) |
| 9.3 | `README.md`: problem, architecture diagram, quickstart, API examples, deployment steps |
| 9.4 | Final report in `REPORTS/`: methodology, every §1.3 data decision and its justification, model comparison table, error analysis, inventory simulation results vs the four targets, limitations, next steps |
| 9.5 | Model card for the registered model: intended use, training data window, known failure modes, the censoring caveat |
| 9.6 | Demo walkthrough / short recording |

---

## 5. Risk Register

| Risk | Likelihood | Mitigation |
|---|---|---|
| MAPE ≤ 15% unreachable at daily grain | **High** — see §1.3(e) | Renegotiate to weekly grain now; report WAPE/MASE alongside; document the reasoning |
| Censored target biases forecasts low | High | Model planned pour or unconstrained demand; flag constrained rows; discuss openly |
| Broken inventory ledger invalidates utilisation/write-off targets | High | Repair ledger in Phase 1 before any inventory work |
| Model barely beats "forecast = planned pour" | Medium | Benchmark it in Phase 3.2 explicitly; if true, pivot the value story to inventory optimisation and alerting, which still delivers |
| Notebook merge conflicts between collaborators | Medium | One owner per notebook + nbstripout, from day 1 |
| AWS costs drift | Medium | Budget alarm, Fargate Spot, documented teardown |
| Scope creep across 10 weeks | Medium | Phase exit criteria are gates — don't start the next phase until the current one passes |
| Model artifact/env mismatch between training and API | Low-Medium | MLflow logs the conda/pip env; API pulls from registry; pin versions |

---

## 6. First Week Checklist

**MLE**

- [ ] Repo hygiene fixes (§1.1, items 1–6)
- [ ] venv + `requirements.txt` + `Makefile` + editable `src/` package
- [ ] Branch protection, PR template, `nbstripout`
- [ ] MLflow running locally, one dummy run logged

**DS**

- [ ] Reproduce every finding in §1.3 independently and record agreement/disagreement in `REPORTS/decisions.md`
- [ ] Decide the missing-date policy
- [ ] Decide the inventory ledger repair approach
- [ ] Draft `01_Data_Ingestion_Cleaning.ipynb` end to end
- [ ] Propose the revised metric set to replace bare MAPE

**Together**

- [ ] Sign off Contract 1 and Contract 2
- [ ] Agree the 10-week phase calendar and the twice-weekly sync slot
- [ ] Agree the AWS path (A or B) and who owns the account/billing
