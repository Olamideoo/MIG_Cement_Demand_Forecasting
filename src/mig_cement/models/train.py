"""Training entrypoint. Run: python -m mig_cement.models.train

Logs params, metrics, the feature list, the split spec, the fitted transformer
state and the model itself to MLflow. The transformer artifacts are what the API
loads at startup - without them, encoders refit independently and predictions
degrade silently.
"""

from __future__ import annotations

import argparse


def time_split(panel, train_end: str, val_end: str):
    """Time-based split. Never random - this is a forecasting problem."""
    raise NotImplementedError


def run(model_family: str = "lightgbm", register: bool = False) -> str:
    """Train, evaluate on validation, log to MLflow. Returns the run id."""
    raise NotImplementedError


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="lightgbm",
                   choices=["naive", "planned_pour", "sarimax", "rf", "lightgbm"])
    p.add_argument("--register", action="store_true")
    args = p.parse_args()
    run(args.model, args.register)


if __name__ == "__main__":
    main()
