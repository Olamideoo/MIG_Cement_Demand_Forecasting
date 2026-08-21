"""Environment configuration.

Uses pydantic-settings BaseSettings (NOT BaseModel - that is for API payloads,
see api/schemas.py). Every path or URL that differs between laptop, container
and AWS belongs here, plus the business assumptions used by the inventory
simulation so they are not hardcoded across notebooks.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- data ---
    db_path: Path = REPO_ROOT / "DATA" / "raw" / "MIG_Cement_Records.db"
    interim_dir: Path = REPO_ROOT / "DATA" / "interim"
    processed_dir: Path = REPO_ROOT / "DATA" / "processed"
    models_dir: Path = REPO_ROOT / "MODELS"

    # --- mlflow ---
    # A tracking server, not a local store, and both halves of that are
    # deliberate.
    #
    # Not a file store: MLflow 3 puts the filesystem backend in maintenance mode
    # and raises on first use, and it cannot host the model registry.
    #
    # Not a local SQLite file either, though that fixes both of those. With any
    # local store, the artefact directory's absolute path is baked into the
    # experiment when it is created - C:\...\mlruns\artifacts from the host,
    # /app/mlruns/artifacts from a container. Whichever environment did not
    # create the experiment then cannot write artefacts to it. Talking HTTP to a
    # server that owns the store removes the question of paths altogether, and
    # is the same arrangement this would use on AWS with S3 behind it.
    #
    # Start it with `make mlflow` or `docker compose up mlflow`. When it is not
    # running, training still succeeds and simply logs nothing - see tracking.py.
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment: str = "mig-cement-forecasting"

    # Where the *server* keeps artefacts, relative to itself. Clients never use
    # this; they address artefacts through mlflow-artifacts:/ URIs that the
    # server resolves. Kept here so `make mlflow` and compose agree.
    mlflow_artifact_root: str = str(REPO_ROOT / "mlruns" / "artifacts")
    mlflow_backend_store: str = f"sqlite:///{(REPO_ROOT / 'mlruns' / 'mlflow.db').as_posix()}"

    model_name: str = "rf_demand_forecaster"
    model_stage: str = "Staging"

    # --- services ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_base_url: str = "http://localhost:8000"
    dashboard_port: int = 8050

    # --- modelling ---
    forecast_horizon_weeks: int = 8
    train_end: str = "2024-06-30"
    val_end: str = "2024-09-30"

    # --- inventory assumptions (see WORKFLOW.md phase 4) ---
    lead_time_days: int = 3
    service_level_z: float = 2.05  # ~98% service level


settings = Settings()
