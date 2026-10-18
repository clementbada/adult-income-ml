"""STEP 3 - Baseline model: L2-regularized logistic regression inside the full Pipeline.

Why this baseline (Sessions 1 and 2):
- logistic regression models P(Y=1 | x) = sigmoid(x'beta) and is trained by minimizing log-loss;
- the L2 (Ridge) penalty shrinks coefficients and keeps them stable when predictors are correlated
  (e.g. marital-status and relationship). In scikit-learn, C = 1 / lambda: smaller C = stronger penalty;
- regularization requires scaled features: the StandardScaler sits inside the Pipeline (STEP 2),
  fitted on the training folds only.
C is fixed at its default (1.0) here; choosing it is a model-selection question, done with Optuna
in STEP 5.

Run from the project root:  python -m src.models
(cross-validates the baseline on the development set and logs the run in MLflow)
"""

from __future__ import annotations

import logging
from typing import Any

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.features import build_preprocessor

logger = logging.getLogger(__name__)


def build_logistic(
    cfg: dict[str, Any], class_weight: str | None = None, smote: bool = False, **overrides: Any,
) -> Pipeline:
    """Full Pipeline: STEP 2 'linear' preprocessor + logistic regression.

    STEP 4 options (class-imbalance remedies, Session 6):
    - class_weight="balanced": errors on the minority class (>50K) weigh n_neg / n_pos ~ 3.2 times
      more in the training loss;
    - smote=True: SMOTE creates synthetic >50K rows. It sits INSIDE an imblearn Pipeline, after the
      preprocessing, so it only ever runs on the training folds (validation and test keep the real
      class distribution).
    """
    params = {**cfg["models"]["logistic"], **overrides}
    model = LogisticRegression(
        penalty=params["penalty"],
        C=params["C"],
        max_iter=params["max_iter"],
        class_weight=class_weight,
        random_state=cfg["seed"],
    )
    preprocess = build_preprocessor(cfg, family="linear")
    if smote:
        # imblearn does not accept a nested Pipeline as a step: the two preprocessing steps
        # (feature builder, column transformer) are inserted one by one.
        return ImbPipeline([*preprocess.steps,
                            ("smote", SMOTE(random_state=cfg["seed"])),
                            ("model", model)])
    return Pipeline([("preprocess", preprocess), ("model", model)])


def main() -> None:
    """Cross-validate the baseline on the development set and log it in MLflow."""
    from src.data import get_cv, load_config, load_splits
    from src.evaluate import cross_validate_oof, format_summary
    from src.tracking import log_cv_run, setup_mlflow

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    X_dev, _, y_dev, _ = load_splits(cfg)            # the test set is not used in STEP 3
    threshold = cfg["evaluation"]["default_threshold"]

    pipe = build_logistic(cfg)
    per_fold, _ = cross_validate_oof(pipe, X_dev, y_dev, get_cv(cfg), threshold)
    for metric, value in format_summary(per_fold).items():
        logger.info("%-10s %s", metric, value)

    setup_mlflow(cfg)
    log_cv_run(
        run_name="step3_logistic_baseline",
        params={**{f"model_{k}": v for k, v in cfg["models"]["logistic"].items()},
                "capital": cfg["features"]["capital"], "bands": ",".join(cfg["features"]["bands"]),
                "threshold": threshold, "n_folds": cfg["split"]["n_folds"], "seed": cfg["seed"]},
        per_fold=per_fold,
        tags={"step": "3", "model": "logistic_regression", "family": "linear"},
    )


if __name__ == "__main__":
    main()
