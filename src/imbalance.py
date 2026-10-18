"""STEP 4 - Class imbalance: compare the Session 6 remedies on the same folds and choose one.

Remedy hierarchy (Session 6), all on the STEP 3 logistic regression, same 5 folds, same grader:
  A. baseline at the default threshold 0.5 (STEP 3 reference);
  B. threshold tuning: same fitted model, the cut-off is chosen to minimise the business cost;
  C. class weights (class_weight="balanced") + tuned threshold;
  D. SMOTE inside the training folds only (imblearn Pipeline) + tuned threshold.

Business cost (decision 27): cost = cost_fn x (wealthy prospects missed) + cost_fp x (useless calls),
with cost_fn = 5 and cost_fp = 1 (config.yaml, section `imbalance`). Every threshold is chosen on
OUT-OF-FOLD predictions of the development set; the test set is not used (STEP 6).

Run from the project root:  python -m src.imbalance
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator

from src.evaluate import best_threshold, cross_validate_oof, threshold_table
from src.models import build_logistic

logger = logging.getLogger(__name__)

# name -> options passed to build_logistic
REMEDIES: dict[str, dict[str, Any]] = {
    "threshold_only": {},
    "class_weight": {"class_weight": "balanced"},
    "smote": {"smote": True},
}


def run_remedies(
    cfg: dict[str, Any], X: pd.DataFrame, y: pd.Series, cv: BaseCrossValidator,
) -> dict[str, tuple[pd.DataFrame, np.ndarray]]:
    """Cross-validate the logistic regression once per remedy: {name: (per-fold metrics, OOF proba)}."""
    out = {}
    for name, options in REMEDIES.items():
        logger.info("Remedy: %s", name)
        out[name] = cross_validate_oof(build_logistic(cfg, **options), X, y, cv)
    return out


def remedy_summary(
    y: pd.Series, results: dict[str, tuple[pd.DataFrame, np.ndarray]], cv: BaseCrossValidator,
    X: pd.DataFrame, cost_fn: float, cost_fp: float = 1.0,
) -> pd.DataFrame:
    """One row per remedy: ranking metrics (unchanged by the threshold) and the business operating
    point at the cost-minimising threshold, with the cost per fold (mean, std) at that threshold."""
    folds = [va for _, va in cv.split(X, y)]
    rows = []
    for name, (per_fold, oof) in results.items():
        t = best_threshold(y, oof, cost_fn, cost_fp)
        op = threshold_table(y, oof, cost_fn, cost_fp, thresholds=np.array([t])).iloc[0]
        fold_cost = [threshold_table(y.iloc[va], oof[va], cost_fn, cost_fp, np.array([t]))["cost"].iloc[0]
                     for va in folds]
        fold_best_t = [best_threshold(y.iloc[va], oof[va], cost_fn, cost_fp) for va in folds]
        rows.append({
            "remedy": name,
            "pr_auc": per_fold["pr_auc"].mean(), "pr_auc_std": per_fold["pr_auc"].std(),
            "roc_auc": per_fold["roc_auc"].mean(), "log_loss": per_fold["log_loss"].mean(),
            "threshold": t, "calls": int(op["calls"]), "found": int(op["tp"]), "missed": int(op["fn"]),
            "useless_calls": int(op["fp"]), "precision": op["precision"], "recall": op["recall"],
            "cost": op["cost"], "fold_cost_mean": float(np.mean(fold_cost)),
            "fold_cost_std": float(np.std(fold_cost, ddof=1)),
            "fold_best_thresholds": ", ".join(f"{v:.2f}" for v in fold_best_t),
        })
    return pd.DataFrame(rows).set_index("remedy")


def sensitivity_table(
    y: pd.Series, proba: np.ndarray, cost_fns: list[float], cost_fp: float = 1.0,
) -> pd.DataFrame:
    """Robustness of the business assumption: operating point for each cost ratio X, and the price of
    each step (extra calls needed per extra wealthy prospect found)."""
    rows = []
    for x in cost_fns:
        t = best_threshold(y, proba, x, cost_fp)
        op = threshold_table(y, proba, x, cost_fp, thresholds=np.array([t])).iloc[0]
        rows.append({"X": x, "threshold": t, "calls": int(op["calls"]),
                     "share_called": op["calls"] / len(y), "found": int(op["tp"]),
                     "missed": int(op["fn"]), "recall": op["recall"], "precision": op["precision"]})
    table = pd.DataFrame(rows).set_index("X")
    table["extra_calls"] = table["calls"].diff()
    table["extra_found"] = table["found"].diff()
    table["calls_per_extra_found"] = table["extra_calls"] / table["extra_found"]
    return table


def cross_validated_threshold(
    y: pd.Series, proba: np.ndarray, cv: BaseCrossValidator, X: pd.DataFrame,
    cost_fn: float, cost_fp: float = 1.0,
) -> pd.DataFrame:
    """Honest estimate of the threshold choice (same logic as cross-validation / an out-of-sample
    backtest): for each fold, the threshold is chosen on the OTHER 4 folds' out-of-fold predictions,
    then the cost is measured on this fold, which played no part in the choice.

    Compared with choosing the threshold on all predictions and measuring on the same predictions
    (slightly optimistic: the threshold is picked partly because it was lucky on these people)."""
    folds = [va for _, va in cv.split(X, y)]
    pooled_t = best_threshold(y, proba, cost_fn, cost_fp)
    rows = []
    for k, va in enumerate(folds, start=1):
        others = np.setdiff1d(np.arange(len(y)), va)
        t_k = best_threshold(y.iloc[others], proba[others], cost_fn, cost_fp)   # chosen without fold k
        honest = threshold_table(y.iloc[va], proba[va], cost_fn, cost_fp, np.array([t_k])).iloc[0]
        in_sample = threshold_table(y.iloc[va], proba[va], cost_fn, cost_fp, np.array([pooled_t])).iloc[0]
        rows.append({"fold": k, "threshold_chosen_without_fold": t_k,
                     "cost_honest": honest["cost"], "recall_honest": honest["recall"],
                     "precision_honest": honest["precision"],
                     "threshold_all_data": pooled_t, "cost_in_sample": in_sample["cost"]})
    return pd.DataFrame(rows).set_index("fold")


def main() -> None:
    """Compare the remedies, print the decision tables and log one MLflow run per remedy."""
    from src.data import get_cv, load_config, load_splits
    from src.tracking import log_cv_run, setup_mlflow

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    icfg = cfg["imbalance"]
    X_dev, _, y_dev, _ = load_splits(cfg)            # the test set is not used in STEP 4
    cv = get_cv(cfg)

    results = run_remedies(cfg, X_dev, y_dev, cv)
    summary = remedy_summary(y_dev, results, cv, X_dev, icfg["cost_fn"], icfg["cost_fp"])
    sens = sensitivity_table(y_dev, results["threshold_only"][1], icfg["cost_fn_sensitivity"], icfg["cost_fp"])
    honest = cross_validated_threshold(y_dev, results["threshold_only"][1], cv, X_dev,
                                       icfg["cost_fn"], icfg["cost_fp"])
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        logger.info("Remedies at X = %s:\n%s", icfg["cost_fn"], summary.round(3))
        logger.info("Sensitivity (threshold only):\n%s", sens.round(3))
        logger.info("Cross-validated threshold (honest estimate):\n%s", honest.round(3))

    setup_mlflow(cfg)
    for name, (per_fold, _) in results.items():
        row = summary.loc[name]
        run_id = log_cv_run(
            run_name=f"step4_logistic_{name}",
            params={"remedy": name, **{f"model_{k}": v for k, v in cfg["models"]["logistic"].items()},
                    "cost_fn": icfg["cost_fn"], "cost_fp": icfg["cost_fp"], "seed": cfg["seed"]},
            per_fold=per_fold,
            tags={"step": "4", "model": "logistic_regression", "family": "linear"},
        )
        import mlflow
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics({"chosen_threshold": row["threshold"], "business_cost": row["cost"],
                                "recall_at_threshold": row["recall"],
                                "precision_at_threshold": row["precision"]})


if __name__ == "__main__":
    main()
