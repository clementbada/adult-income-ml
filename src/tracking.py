"""Experiment tracking with MLflow (project requirement 7).

MLflow is our logbook: every training run records
- its parameters (model settings, preprocessing options),
- its metrics (mean and standard deviation over the 5 folds, and each fold),
- its artifacts (files: per-fold table, figures).
Runs are stored locally in the folder `mlruns/` (not committed, see .gitignore).
To browse them:  mlflow ui   then open http://127.0.0.1:5000 in a browser.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd

from src.data import resolve
from src.evaluate import METRIC_NAMES, summarize

logger = logging.getLogger(__name__)


def setup_mlflow(cfg: dict[str, Any]) -> None:
    """Point MLflow to the local `mlruns/` folder and select the project experiment."""
    tcfg = cfg["tracking"]
    mlflow.set_tracking_uri(resolve(tcfg["tracking_dir"]).as_uri())
    mlflow.set_experiment(tcfg["experiment_name"])


def log_cv_run(
    run_name: str,
    params: dict[str, Any],
    per_fold: pd.DataFrame,
    tags: dict[str, str] | None = None,
    figures: list[str | Path] | None = None,
) -> str:
    """Log one cross-validated model as one MLflow run. Returns the run id."""
    summary = summarize(per_fold)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(tags or {})
        mlflow.log_params(params)
        for m in METRIC_NAMES:
            mlflow.log_metric(f"cv_{m}_mean", float(summary.loc[m, "mean"]))
            mlflow.log_metric(f"cv_{m}_std", float(summary.loc[m, "std"]))
            for fold, value in per_fold[m].items():
                mlflow.log_metric(f"cv_{m}", float(value), step=int(fold))   # one point per fold
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cv_per_fold.csv"
            per_fold.to_csv(path)
            mlflow.log_artifact(str(path))
        for fig in figures or []:
            mlflow.log_artifact(str(fig), artifact_path="figures")
        logger.info("MLflow run '%s' logged (id %s)", run_name, run.info.run_id)
        return run.info.run_id
