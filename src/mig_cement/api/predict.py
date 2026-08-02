"""Inference. Loads the model and its transformer state from the MLflow registry.

Calls features.build.build_features - the SAME function used in training. If you
find yourself reimplementing feature logic here, stop: that is the train/serve
skew failure described in WORKFLOW.md section 2.3.
"""

from __future__ import annotations

import pandas as pd


class Forecaster:
    """Holds the loaded model, encoders and feature column order."""

    def __init__(self) -> None:
        self.model = None
        self.feature_columns: list[str] = []
        self.model_version: str = "unloaded"

    def load(self) -> None:
        """Load model + transformer artifacts from the registry. Called once at
        app startup via the FastAPI lifespan handler, never per request."""
        raise NotImplementedError

    def predict(self, site_id: str, cement_type: str,
                horizon_weeks: int) -> pd.DataFrame:
        """Recursive multi-step forecast for one series."""
        raise NotImplementedError

    def predict_batch(self, site_ids: list[str] | None,
                      cement_types: list[str] | None,
                      horizon_weeks: int) -> pd.DataFrame:
        raise NotImplementedError


forecaster = Forecaster()
