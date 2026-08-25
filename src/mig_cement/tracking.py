# MLflow run tracking for the training pipeline.


from __future__ import annotations

import contextlib
import os
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from mig_cement.config import settings


def git_revision() -> str | None:
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

    active = False

    def log_params(self, params: dict[str, Any]) -> None: ...
    def log_metrics(self, metrics: dict[str, float]) -> None: ...
    def set_tags(self, tags: dict[str, Any]) -> None: ...
    def log_artifact(self, path: Path | str, artifact_path: str | None = None) -> None: ...
    def log_model(self, model: Any, sample_input: Any = None,
                  sample_output: Any = None, name: str = "model") -> None: ...


class _MlflowRun(_NullRun):

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
        
        with self._guard("model"):
            sample_input = _widen_integers(sample_input)

            signature = None
            if sample_input is not None and sample_output is not None:
                from mlflow.models import infer_signature
                signature = infer_signature(sample_input, sample_output)

            import mlflow.sklearn

            kwargs = dict(sk_model=model, signature=signature,
                          input_example=sample_input,
                          registered_model_name=settings.model_name)
            try:
                mlflow.sklearn.log_model(name=name, **kwargs)
            except TypeError:
                mlflow.sklearn.log_model(artifact_path=name, **kwargs)


def _widen_integers(sample: Any) -> Any:
    if sample is None or not hasattr(sample, "select_dtypes"):
        return sample
    integer_cols = sample.select_dtypes(include="integer").columns
    return sample.astype({c: "float64" for c in integer_cols}) if len(integer_cols) else sample


def _server_reachable(uri: str, timeout: float = 2.0) -> bool:
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(f"{uri.rstrip('/')}/health", timeout=timeout) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


def _ensure_store(uri: str) -> None:
    if uri.startswith("sqlite:///"):
        db = Path(uri.removeprefix("sqlite:///"))
        db.parent.mkdir(parents=True, exist_ok=True)


def _ensure_experiment(mlflow: Any, name: str, uri: str) -> None:
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    if client.get_experiment_by_name(name) is None:
        kwargs: dict[str, Any] = {}
        if not uri.startswith(("http://", "https://")):
            root = Path(settings.mlflow_artifact_root)
            root.mkdir(parents=True, exist_ok=True)
            kwargs["artifact_location"] = root.as_uri()
        with contextlib.suppress(Exception):
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

    uri = os.environ.get("MLFLOW_TRACKING_URI") or settings.mlflow_tracking_uri

    if uri.startswith(("http://", "https://")) and not _server_reachable(uri):
        print(f"  mlflow: no tracking server at {uri} - run not tracked\n"
              "    start it with `make mlflow`, or "
              "`docker compose -f docker/docker-compose.yml up -d mlflow`")
        yield _NullRun()
        return

    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")

    try:
        _ensure_store(uri)
        mlflow.set_tracking_uri(uri)
        _ensure_experiment(mlflow, settings.mlflow_experiment, uri)
        run = mlflow.start_run(run_name=run_name)
    except Exception as exc:                            # noqa: BLE001
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
