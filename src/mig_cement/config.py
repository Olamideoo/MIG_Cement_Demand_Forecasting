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

    # --- mlflow ---
    mlflow_tracking_uri: str = f"file://{REPO_ROOT / 'mlruns'}"
    mlflow_experiment: str = "mig-cement-forecasting"
    model_name: str = "mig-cement-forecaster"
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
