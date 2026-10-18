"""STEP 3 and STEP 5 - The three models, each one inside a single full Pipeline (requirement 5).

STEP 3 - Baseline model: L2-regularized logistic regression inside the full Pipeline.

Why this baseline (Sessions 1 and 2):
- logistic regression models P(Y=1 | x) = sigmoid(x'beta) and is trained by minimizing log-loss;
- the L2 (Ridge) penalty shrinks coefficients and keeps them stable when predictors are correlated
  (e.g. marital-status and relationship). In scikit-learn, C = 1 / lambda: smaller C = stronger penalty;
- regularization requires scaled features: the StandardScaler sits inside the Pipeline (STEP 2),
  fitted on the training folds only.
C is fixed at its default (1.0) here; choosing it is a model-selection question, done with Optuna
in STEP 5.

STEP 5 - Two tree ensembles (Sessions 3 and 4), to compare with the baseline (requirement 4):
- Random Forest: many deep trees, each trained on a bootstrap sample and a random subset of columns,
  then averaged. Averaging reduces the variance of single trees (Session 4);
- LightGBM: gradient boosting, small trees added one after the other, each one correcting the errors
  of the previous ones (Session 4). Histogram-based and fast, with native categorical support.
Both trees learn non-linear shapes (age bell curve) and interactions by themselves, so they get no
age / hours bands (decision 13). Their hyperparameters are tuned with Optuna in src/tuning.py.

Run from the project root:  python -m src.models
(cross-validates the baseline on the development set and logs the run in MLflow)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.features import build_preprocessor, model_inputs

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


def build_random_forest(cfg: dict[str, Any], encoding: str = "onehot", **overrides: Any) -> Pipeline:
    """Full Pipeline: preprocessor (no bands) + Random Forest.

    The categorical encoding is itself a choice we let Optuna make (decision 33):
    - "ordinal": the codes of the "tree" preprocessor. scikit-learn's Random Forest has no
      categorical mode, so it reads the codes as ordered numbers (e.g. occupation 3 < occupation 7),
      an order that does not exist; a split can still isolate a category, but needs several cuts;
    - "onehot": the "linear" preprocessor without bands, one yes/no column per category. Its
      StandardScaler changes nothing for a tree (a split "age <= 40" becomes "scaled age <= 0.12",
      same people on each side), so we reuse it instead of writing a third preprocessor.
    A first check with default settings (notebook 05, Part B) gave no clear winner: the answer
    depends on the other hyperparameters, hence the choice inside the Optuna search.
    """
    params = {**cfg["models"]["random_forest"], **overrides}
    model = RandomForestClassifier(random_state=cfg["seed"], **params)
    if encoding == "onehot":
        preprocess = build_preprocessor(cfg, family="linear", bands=())
    elif encoding == "ordinal":
        preprocess = build_preprocessor(cfg, family="tree")
    else:
        raise ValueError(f"encoding must be 'onehot' or 'ordinal', got {encoding!r}")
    return Pipeline([("preprocess", preprocess), ("model", model)])


class ToCategory(BaseEstimator, TransformerMixin):
    """Stateless step: turn the ordinal-coded categorical columns into pandas 'category' columns.

    Why: LightGBM handles categorical columns natively (it groups categories at each split instead of
    reading the codes as ordered numbers), but only if it knows which columns are categorical. With
    pandas 'category' columns, LightGBM detects them by itself (categorical_feature="auto"), so the
    full Pipeline can be fitted, cross-validated and cloned like any other model, without passing an
    extra argument to fit(). This replaces the `categorical_mask` route planned in STEP 2.
    A category never seen in training gets the code -1, which LightGBM treats as a missing value:
    the prediction still works instead of raising an error.
    """

    def __init__(self, columns: tuple[str, ...] = ()):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: Any = None) -> "ToCategory":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for col in self.columns:
            out[col] = out[col].astype("category")
        return out

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.asarray(input_features, dtype=object)


def build_lightgbm(cfg: dict[str, Any], **overrides: Any) -> Pipeline:
    """Full Pipeline: "tree" preprocessor (ordinal codes, no scaling, no bands) + LightGBM.

    The preprocessor outputs a pandas DataFrame (set_output) so that the column names and the
    'category' types reach LightGBM (see ToCategory).
    """
    params = {**cfg["models"]["lightgbm"], **overrides}
    model = LGBMClassifier(random_state=cfg["seed"], **params)
    preprocess = build_preprocessor(cfg, family="tree").set_output(transform="pandas")
    capital = cfg.get("features", {}).get("capital", "indicators")
    _, cat_cols = model_inputs("tree", capital, ())
    return Pipeline([*preprocess.steps, ("to_category", ToCategory(tuple(cat_cols))), ("model", model)])


# One entry per model: used by src/tuning.py so that the three models go through the same code.
MODEL_BUILDERS = {
    "logistic": build_logistic,
    "random_forest": build_random_forest,
    "lightgbm": build_lightgbm,
}


def build_model(name: str, cfg: dict[str, Any], **overrides: Any) -> Pipeline:
    """Build one of the three models by name, with optional hyperparameter overrides."""
    if name not in MODEL_BUILDERS:
        raise ValueError(f"model must be one of {list(MODEL_BUILDERS)}, got {name!r}")
    return MODEL_BUILDERS[name](cfg, **overrides)


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
