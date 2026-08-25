"""Training pipeline: SQLite -> clean -> features -> model artefact.

    python -m mig_cement.pipeline          # train, save, log the run
    python -m mig_cement.pipeline --dry-run
    python -m mig_cement.pipeline --no-tracking

Writes MODELS/rf_demand_forecaster.joblib, its metadata sidecar, and the hold-out
forecasts consumed by Step 5. Each run is also recorded to MLflow - parameters,
both sets of metrics, the fitted model and its artefacts - so runs can be
compared rather than overwritten. See tracking.py; logging never fails a run.

Frozen configuration: RandomForest(300), weekly per site, 8-week horizon, 8
schedule-and-site features. Feature construction lives in `features/build.py` so the
API can reuse it unchanged.
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from mig_cement import tracking
from mig_cement.config import settings
from mig_cement.data import load, preprocess, validate
from mig_cement.features.build import CATEGORICAL_COLS, FEATURES, TARGET, build_weekly_panel

TRAIN_END, VAL_END = "2024-06-30", "2024-09-30"
HORIZON_WEEKS = 8
N_ESTIMATORS, RANDOM_STATE = 300, 42


def build_model() -> Pipeline:
    """Encoder and estimator together, so serving cannot drift from training."""
    numeric = [f for f in FEATURES if f not in CATEGORICAL_COLS]
    return Pipeline([
        ("preprocessor", ColumnTransformer([
            ("num", "passthrough", numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS)])),
        ("model", RandomForestRegressor(n_estimators=N_ESTIMATORS,
                                        random_state=RANDOM_STATE, n_jobs=-1)),
    ])


def score(y_true, y_pred) -> dict[str, float]:
    """MAPE on non-zero actuals; the rest over everything."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    nz = y_true != 0
    return {
        "MAPE": float(np.mean(np.abs((y_true[nz] - y_pred[nz]) / y_true[nz]))),
        "RMSE": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "WAPE": float(np.abs(y_true - y_pred).sum() / np.abs(y_true).sum()),
        "MAE": float(np.abs(y_true - y_pred).mean()),
        "bias": float((y_pred - y_true).mean()),
    }


def run(db_path: Path | None = None, save: bool = True, track: bool = True) -> dict:
    started = datetime.now(timezone.utc)
    run_name = f"rf{N_ESTIMATORS}-{started:%Y%m%d-%H%M%S}"

    with tracking.start(run_name, enabled=track and save) as mlrun:
        # 1. data -------------------------------------------------------------
        raw = load.load_panel(db_path)
        clean, report = preprocess.build_clean_panel(raw, mode="cap_deliveries")
        validate.validate_clean(clean)
        weekly, dropped = build_weekly_panel(clean)
        print(f"panel: {weekly.shape} | {weekly.date.nunique()} weeks x "
              f"{weekly.site_id.nunique()} sites | {dropped} partial weeks dropped")

        # 2. split ------------------------------------------------------------
        d = weekly["date"]
        train = weekly[d <= TRAIN_END]
        val = weekly[(d > TRAIN_END) & (d <= VAL_END)].groupby("site_id").head(HORIZON_WEEKS)
        test = weekly[d > VAL_END].groupby("site_id").head(HORIZON_WEEKS)
        print(f"split: train {len(train):,} | val {len(val)} | test {len(test)}")

        mlrun.log_params({
            "estimator": "RandomForestRegressor",
            "n_estimators": N_ESTIMATORS,
            "random_state": RANDOM_STATE,
            "preprocess_mode": "cap_deliveries",
            "train_end": TRAIN_END,
            "val_end": VAL_END,
            "horizon_weeks": HORIZON_WEEKS,
            "n_features": len(FEATURES),
            "features": ",".join(FEATURES),
            "categorical_features": ",".join(CATEGORICAL_COLS),
            "grain": "weekly per site",
            "rows_train": len(train),
            "rows_val": len(val),
            "rows_test": len(test),
            "n_sites": int(weekly.site_id.nunique()),
            "partial_weeks_dropped": dropped,
        })
        mlrun.set_tags({
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "git_commit": tracking.git_revision(),
            "target_mape": 0.15,
            "dataset": Path(db_path or settings.db_path).name,
        })

        # 3. validation fit - reproduces the selection figure, catches drift ---
        model = build_model().fit(train[FEATURES], train[TARGET])
        val_metrics = score(val[TARGET], np.clip(model.predict(val[FEATURES]), 0, None))
        print(f"validation: MAPE {100*val_metrics['MAPE']:.2f}%  "
              f"RMSE {val_metrics['RMSE']:.2f} t")

        # 4. final fit on train + validation - the deployable model ------------
        refit = weekly[d <= VAL_END]
        model = build_model().fit(refit[FEATURES], refit[TARGET])

        test_pred = np.clip(model.predict(test[FEATURES]), 0, None)
        test_metrics = score(test[TARGET], test_pred)
        met = test_metrics["MAPE"] <= 0.15
        print(f"hold-out:   MAPE {100*test_metrics['MAPE']:.2f}%  "
              f"RMSE {test_metrics['RMSE']:.2f} t  bias {test_metrics['bias']:+.2f} t  "
              f"target <= 15% {'PASS' if met else 'FAIL'}")

    
        mlrun.log_metrics({
            **{f"val_{k}": v for k, v in val_metrics.items()},
            **{f"test_{k}": v for k, v in test_metrics.items()},
            "target_met": float(met),
            "rows_refit": float(len(refit)),
        })

        if not save:
            print("dry run - nothing written")
            return {"validation": val_metrics, "test": test_metrics}

        # 5. persist ----------------------------------------------------------
        settings.models_dir.mkdir(parents=True, exist_ok=True)
        settings.processed_dir.mkdir(parents=True, exist_ok=True)
        model_path = settings.models_dir / "rf_demand_forecaster.joblib"
        joblib.dump(model, model_path, compress=3)

        meta_path = settings.models_dir / "rf_demand_forecaster_meta.json"
        meta_path.write_text(json.dumps({
            "name": "rf_demand_forecaster",
            "created_utc": started.isoformat(timespec="seconds"),
            "estimator": f"RandomForestRegressor(n_estimators={N_ESTIMATORS}, "
                         f"random_state={RANDOM_STATE})",
            "features": FEATURES,                  # exact column order for inference
            "categorical_features": CATEGORICAL_COLS,
            "target": "y (consumed_tonnes, summed to the week)",
            "grain": "weekly, per site, weeks labelled by Monday start",
            "horizon_weeks": HORIZON_WEEKS,
            "trained_on": {"rows": int(len(refit)), "from": str(refit.date.min().date()),
                           "to": str(refit.date.max().date()),
                           "sites": int(refit.site_id.nunique())},
            "validation_metrics": {k: round(v, 4) for k, v in val_metrics.items()},
            "holdout_metrics": {k: round(v, 4) for k, v in test_metrics.items()},
            "project_target": {"MAPE": 0.15, "met": bool(met)},
            "versions": {"python": platform.python_version(),
                         "scikit_learn": sklearn.__version__,
                         "pandas": pd.__version__, "numpy": np.__version__},
        }, indent=2))

        forecasts = test[["site_id", "date", "planned_pour_tonnes", "silo_capacity"]].copy()
        forecasts["actual_tonnes"] = test[TARGET].values
        forecasts["forecast_tonnes"] = test_pred
        forecasts["horizon_week"] = forecasts.groupby("site_id").cumcount() + 1
        forecasts_path = settings.processed_dir / "test_forecasts.parquet"
        forecasts.to_parquet(forecasts_path, index=False)

        # reloading is the only proof the artefact actually works
        diff = float(np.abs(np.clip(joblib.load(model_path).predict(test[FEATURES]), 0, None)
                            - test_pred).max())
        assert diff < 1e-9, f"reloaded model does not reproduce predictions ({diff})"
        print(f"saved: {model_path.name} ({model_path.stat().st_size/1e6:.1f} MB), "
              f"metadata, {len(forecasts)} forecasts | reload verified ({diff:.0e})")

        # 6. log the run ------------------------------------------------------
        mlrun.log_metrics({
            "model_size_mb": model_path.stat().st_size / 1e6,
            "reload_max_abs_diff": diff,
        })
        mlrun.log_artifact(meta_path)
        mlrun.log_artifact(forecasts_path)
        mlrun.log_model(model,
                        sample_input=refit[FEATURES].head(5),
                        sample_output=test_pred[:5],
                        name="rf_demand_forecaster")

        return {"validation": val_metrics, "test": test_metrics,
                "model_path": model_path}


def main() -> None:
    p = argparse.ArgumentParser(description="Train and persist the demand forecaster")
    p.add_argument("--db", type=Path, default=None)
    p.add_argument("--dry-run", action="store_true",
                   help="train and score, write nothing, log nothing")
    p.add_argument("--no-tracking", action="store_true",
                   help="skip MLflow logging (the artefact is still written)")
    args = p.parse_args()
    run(args.db, save=not args.dry_run, track=not args.no_tracking)


if __name__ == "__main__":
    main()
