"""MLflow run tracking for the training pipeline.

Without this, a training run leaves behind a model file and some numbers printed
to a terminal. Run it again next week with different hyperparameters and the
previous figures are gone - overwritten, with no way to compare. This module
makes each run a durable, comparable record: parameters in, metrics out,
artefacts attached, timestamped.

Two design points worth stating, because both are deliberate.

**Tracking never fails a training run.** The pipeline's job is to produce a
model artefact. If mlflow is not installed, or the tracking directory is
read-only, or a log call raises, the run should still finish and still save the
model - so every method here swallows its exceptions and reports once. A
training job that dies because its logging died is worse than one that logs
nothing.

**`import mlflow` happens in exactly one place.** It is a heavy dependency,
pulling in Flask, SQLAlchemy and Alembic. Confining it here means the API and
the dashboard - which have no reason to carry it - never import it even though
it sits in the shared requirements.txt. The import is inside `start()` rather
than at module scope so that merely importing `mig_cement.tracking` costs
nothing.

Usage:

    with tracking.start("nightly") as run:
        run.log_params({"n_estimators": 300})
        run.log_metrics({"test_MAPE": 0.1277})
        run.log_model(model, sample_input=X.head())
"""

from __future__ import annotations

import contextlib
import os
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from mig_cement.config import settings


def git_revision() -> str | None:
    """Short commit hash, with a `-dirty` suffix when the tree has edits.

    Tagging the run with this is what makes it reproducible: metrics without a
    commit tell you what happened but not which code produced it. Returns None
    outside a repository, which is the normal case inside a container.
    """
    def _git(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args], capture_output=True, text=True, timeout=5,
                cwd=Path(__file__).resolve().parent)
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    sha = _git("rev-parse", "--short", "HEAD")
    if sha is None:
        return None
    return f"{sha}-dirty" if _git("status", "--porcelain") else sha


class _NullRun:
    """What callers get when tracking is off or unavailable.

    Every method accepts the same arguments as the real thing and does nothing,
    so `pipeline.py` needs no `if tracking_enabled:` branches around its logging.
    """

    active = False

    def log_params(self, params: dict[str, Any]) -> None: ...
    def log_metrics(self, metrics: dict[str, float]) -> None: ...
    def set_tags(self, tags: dict[str, Any]) -> None: ...
    def log_artifact(self, path: Path | str, artifact_path: str | None = None) -> None: ...
    def log_model(self, model: Any, sample_input: Any = None,
                  sample_output: Any = None, name: str = "model") -> None: ...


class _MlflowRun(_NullRun):
    """A live run. Each call is individually guarded.

    Guarding per call rather than per run matters: if logging the 26 MB model
    fails on a full disk, the params and metrics already recorded should survive
    rather than the whole run being lost.
    """

    active = True

    def __init__(self, mlflow: Any) -> None:
        self._mlflow = mlflow

    @staticmethod
    @contextlib.contextmanager
    def _guard(what: str) -> Iterator[None]:
        try:
            yield
        except Exception as exc:                        # noqa: BLE001 - see module docstring
            print(f"  mlflow: could not log {what} ({exc.__class__.__name__})")

    def log_params(self, params: dict[str, Any]) -> None:
        with self._guard("params"):
            # Values are stringified by mlflow anyway, and long ones are
            # truncated at 6000 characters. The feature list is well inside that.
            self._mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        with self._guard("metrics"):
            self._mlflow.log_metrics({k: float(v) for k, v in metrics.items()})

    def set_tags(self, tags: dict[str, Any]) -> None:
        with self._guard("tags"):
            self._mlflow.set_tags({k: v for k, v in tags.items() if v is not None})

    def log_artifact(self, path: Path | str, artifact_path: str | None = None) -> None:
        path = Path(path)
        if not path.exists():
            return
        with self._guard(path.name):
            self._mlflow.log_artifact(str(path), artifact_path=artifact_path)

    def log_model(self, model: Any, sample_input: Any = None,
                  sample_output: Any = None, name: str = "model") -> None:
        """Log the fitted estimator with a schema.

        The signature is the point of doing this rather than logging the joblib
        as a plain file. It records the exact column names, order and dtypes the
        model expects, so loading it later with the wrong feature order fails
        loudly at load time instead of silently returning wrong numbers - which
        is the failure mode this project spent effort avoiding elsewhere by
        sharing `features/build.py` between training and serving.

        Registering is what turns a pile of runs into a versioned model line:
        each training run adds a new version under one name, so "which model is
        live" has an answer that is not a filename. It needs the database
        backend, which is why config.py points at SQLite.
        """
        with self._guard("model"):
            sample_input = _widen_integers(sample_input)

            signature = None
            if sample_input is not None and sample_output is not None:
                from mlflow.models import infer_signature
                signature = infer_signature(sample_input, sample_output)

            import mlflow.sklearn

            # mlflow renamed this argument in 2.x and removed the old name in
            # 3.x, so try the current spelling first and fall back.
            kwargs = dict(sk_model=model, signature=signature,
                          input_example=sample_input,
                          registered_model_name=settings.model_name)
            try:
                mlflow.sklearn.log_model(name=name, **kwargs)
            except TypeError:
                mlflow.sklearn.log_model(artifact_path=name, **kwargs)


def _widen_integers(sample: Any) -> Any:
    """Declare integer feature columns as float64 in the logged signature.

    Several features - the forward pour sums, days since last delivery - are
    integers in the training panel, so an inferred schema pins them as int64.
    NumPy integers cannot hold a missing value, so the moment a caller passes a
    frame where one of those columns has a NaN, pandas silently promotes it to
    float and MLflow's schema enforcement rejects the request as a type
    mismatch. The model itself is indifferent - a random forest splits on the
    numeric value either way - so the strictness buys nothing and costs a
    confusing runtime failure.

    Widening at signature time is MLflow's own recommended fix. Non-numeric
    columns (site_id, region, behavior) are untouched: they are one-hot encoded
    inside the pipeline and their dtypes carry real meaning.
    """
    if sample is None or not hasattr(sample, "select_dtypes"):
        return sample
    integer_cols = sample.select_dtypes(include="integer").columns
    return sample.astype({c: "float64" for c in integer_cols}) if len(integer_cols) else sample


def _ensure_store(uri: str) -> None:
    """Create the directory a local SQLite store will live in.

    SQLAlchemy will happily create the database file, but not the folder holding
    it, so a first run on a clean checkout would otherwise fail with a bare
    "unable to open database file".
    """
    if uri.startswith("sqlite:///"):
        db = Path(uri.removeprefix("sqlite:///"))
        db.parent.mkdir(parents=True, exist_ok=True)


def _ensure_experiment(mlflow: Any, name: str, uri: str) -> None:
    """Select the experiment, creating it if this is the first run.

    Against a tracking server the artefact location is left unset on purpose.
    The server assigns an `mlflow-artifacts:/` URI it resolves itself, which is
    precisely what lets a run from the host and a run from a container share one
    experiment - neither client ever names a filesystem path.

    Against a local store there is no server to do that, and MLflow would
    otherwise resolve the default location against the current working
    directory, scattering 26 MB models wherever the pipeline happened to be
    launched from. So the location is pinned at creation time instead.
    """
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    if client.get_experiment_by_name(name) is None:
        kwargs: dict[str, Any] = {}
        if not uri.startswith(("http://", "https://")):
            root = Path(settings.mlflow_artifact_root)
            root.mkdir(parents=True, exist_ok=True)
            kwargs["artifact_location"] = root.as_uri()
        with contextlib.suppress(Exception):
            # Suppressed rather than guarded: two runs starting together can
            # both see "missing" and race, and losing that race is harmless -
            # set_experiment below picks up whichever won.
            client.create_experiment(name, **kwargs)
    mlflow.set_experiment(name)


@contextlib.contextmanager
def start(run_name: str | None = None, enabled: bool = True) -> Iterator[_NullRun]:
    """Open a tracked run, or a no-op stand-in if that is not possible.

    `enabled=False` is how `--dry-run` and `--no-tracking` avoid writing history
    for a run nobody intends to keep.
    """
    if not enabled:
        yield _NullRun()
        return

    try:
        import mlflow
    except ImportError:
        print("  mlflow: not installed - run not tracked "
              "(pip install -r requirements.txt)")
        yield _NullRun()
        return

    # An explicit MLFLOW_TRACKING_URI wins, so compose or CI can redirect runs
    # to a server without touching config. Otherwise the local SQLite store.
    uri = os.environ.get("MLFLOW_TRACKING_URI") or settings.mlflow_tracking_uri

    try:
        _ensure_store(uri)
        mlflow.set_tracking_uri(uri)
        _ensure_experiment(mlflow, settings.mlflow_experiment, uri)
        run = mlflow.start_run(run_name=run_name)
    except Exception as exc:                            # noqa: BLE001
        # The exception text is worth the extra line. "MlflowException" alone
        # cannot distinguish a server that is down from one that is up and
        # returning 403, and those need different fixes.
        detail = " ".join(str(exc).split())[:160]
        print(f"  mlflow: {uri} unavailable - run not tracked\n"
              f"    {exc.__class__.__name__}: {detail}")
        if uri.startswith("http"):
            print("    start the server with `make mlflow`, or "
                  "`docker compose -f docker/docker-compose.yml up -d mlflow`")
        yield _NullRun()
        return

    with run:
        print(f"  mlflow: logging to {settings.mlflow_experiment} "
              f"(run {run.info.run_id[:8]})")
        yield _MlflowRun(mlflow)
