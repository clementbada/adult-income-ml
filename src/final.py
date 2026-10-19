"""STEP 6 - Final model: freeze the choices, refit on the development set, evaluate ONCE on the test set.

Course rule (Session 5, "CV tuning + test set"): tune with cross-validation -> freeze the modelling
choices -> evaluate once on the untouched test set. Everything that is chosen (model, hyperparameters,
threshold) is read from the development-set results of STEPS 4 and 5 BEFORE the test set is opened:
- final model: LightGBM (decision 35), hyperparameters in reports/tuning/best_params.yaml;
- threshold: the business-cost rule of STEP 4 (X = 5), recomputed on the model's OWN out-of-fold
  predictions (reports/tuning/oof_predictions.csv), never on the test set;
- the test set is used once, only to MEASURE. Nothing is changed after looking at it.

Two references are measured on the test set at the same time, for comparison only (they cannot
replace the final model, see decision 39):
- the tuned logistic regression (baseline of STEP 3, tuned in STEP 5);
- LightGBM with the exact capital amounts (decision 7), to report what removing them costs.

The threshold is part of the delivered model: `predict_decision` returns 1 when P(>50K) >= threshold.
Without it, scikit-learn's predict() would answer at 0.5 (Session 6).

Run from the project root:  python -m src.final   (about 1 minute)
Writes reports/final/ (tables, test probabilities for STEPS 7 and 9, frozen settings) and one
MLflow run per model.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.pipeline import Pipeline

from src.data import get_cv, load_config, load_splits, resolve
from src.evaluate import best_threshold, compute_metrics
from src.models import build_model
from src.tracking import setup_mlflow
from src.tuning import evaluate_tuned

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------------------------
# 1. Frozen choices (read from the development-set results, before the test set is opened)
# ----------------------------------------------------------------------------------------------
def load_best_params(cfg: dict[str, Any], model: str) -> dict[str, Any]:
    """Hyperparameters chosen by Optuna in STEP 5 (development set only)."""
    path = resolve(cfg["tuning"]["results_dir"]) / "best_params.yaml"
    with open(path) as f:
        return yaml.safe_load(f)[model]


def oof_threshold(cfg: dict[str, Any], model: str, y_dev: pd.Series) -> float:
    """STEP 4 rule on the model's own out-of-fold predictions of STEP 5 (never on the test set)."""
    path = resolve(cfg["tuning"]["results_dir"]) / "oof_predictions.csv"
    oof = pd.read_csv(path, index_col=0)
    oof = oof.loc[y_dev.index]                       # same rows, same order as the development set
    if not (oof["target"].values == y_dev.values).all():
        raise ValueError("oof_predictions.csv does not match the development set: rerun python -m src.tuning")
    icfg = cfg["imbalance"]
    return best_threshold(y_dev, oof[model].values, icfg["cost_fn"], icfg["cost_fp"])


def with_capital(cfg: dict[str, Any], capital: str) -> dict[str, Any]:
    """Copy of the configuration with another capital option (decision 18): same code, other input."""
    out = copy.deepcopy(cfg)
    out["features"]["capital"] = capital
    return out


# ----------------------------------------------------------------------------------------------
# 2. Fit on the whole development set, predict with the threshold
# ----------------------------------------------------------------------------------------------
def fit_on_development(cfg: dict[str, Any], model: str, params: dict[str, Any],
                       X_dev: pd.DataFrame, y_dev: pd.Series) -> Pipeline:
    """Full Pipeline (preprocessing + model) refitted on the 80% development set.
    The 5 models of the cross-validation each saw 4/5 of it; the final model uses all of it."""
    return build_model(model, cfg, **params).fit(X_dev, y_dev)


def predict_decision(pipe: Pipeline, X: pd.DataFrame, threshold: float) -> np.ndarray:
    """Decision of the delivered model: 1 = call the prospect, when P(>50K) >= threshold."""
    return (pipe.predict_proba(X)[:, 1] >= threshold).astype(int)


def business_cost(metrics: dict[str, float], cfg: dict[str, Any]) -> float:
    """STEP 4 cost: cost_fn per missed wealthy prospect + cost_fp per useless call."""
    icfg = cfg["imbalance"]
    return icfg["cost_fn"] * metrics["fn"] + icfg["cost_fp"] * metrics["fp"]


def measure_on_test(name: str, y_test: pd.Series, proba: np.ndarray, threshold: float,
             cfg: dict[str, Any]) -> dict[str, Any]:
    """Same metrics as in STEPS 3-5 (src/evaluate.py) + business view, on the test set."""
    m = compute_metrics(y_test, proba, threshold)
    n = len(y_test)
    return {"model": name, "threshold": threshold, **m,
            "calls": m["tp"] + m["fp"], "calls_share": (m["tp"] + m["fp"]) / n,
            "cost": business_cost(m, cfg), "cost_per_1000": 1000 * business_cost(m, cfg) / n}


def dev_vs_test(cfg: dict[str, Any], test: pd.DataFrame) -> pd.DataFrame:
    """Development (5-fold CV of STEP 5, mean and std) next to the single test measure.
    If the test value lies within about 2 std of the CV mean, the CV estimate was honest."""
    comp = pd.read_csv(resolve(cfg["tuning"]["results_dir"]) / "comparison.csv").set_index("model")
    rows = []
    for name in ("lightgbm", "logistic"):
        cv_mean, cv_std = comp.loc[name, "pr_auc_mean"], comp.loc[name, "pr_auc_std"]
        t = test.set_index("model").loc[name, "pr_auc"]
        rows.append({"model": name, "cv_pr_auc_mean": cv_mean, "cv_pr_auc_std": cv_std,
                     "test_pr_auc": t, "gap_in_std": (t - cv_mean) / cv_std,
                     "cv_recall": comp.loc[name, "recall"],
                     "test_recall": test.set_index("model").loc[name, "recall"]})
    return pd.DataFrame(rows).set_index("model")


# ----------------------------------------------------------------------------------------------
# 3. Run everything
# ----------------------------------------------------------------------------------------------
def run_final(cfg: dict[str, Any]) -> dict[str, Any]:
    fcfg = cfg["final"]
    X_dev, X_test, y_dev, y_test = load_splits(cfg)
    final, ref = fcfg["model"], fcfg["reference"]

    # Frozen choices: hyperparameters and thresholds, from development results only
    params = {m: load_best_params(cfg, m) for m in (final, ref)}
    thresholds = {m: oof_threshold(cfg, m, y_dev) for m in (final, ref)}
    logger.info("Frozen: %s %s, threshold %.2f", final, params[final], thresholds[final])

    # "Exact amounts" reference: same LightGBM settings, threshold from its OWN out-of-fold predictions
    cfg_amounts = with_capital(cfg, "amounts")
    r_amounts = evaluate_tuned(final, params[final], cfg_amounts, X_dev, y_dev, get_cv(cfg))
    amounts_name = f"{final}_amounts"
    thresholds[amounts_name] = r_amounts["threshold"]

    # Refit on the whole development set, then the ONE look at the test set
    pipes = {final: fit_on_development(cfg, final, params[final], X_dev, y_dev),
             ref: fit_on_development(cfg, ref, params[ref], X_dev, y_dev),
             amounts_name: fit_on_development(cfg_amounts, final, params[final], X_dev, y_dev)}
    probas = {name: p.predict_proba(X_test)[:, 1] for name, p in pipes.items()}
    test = pd.DataFrame([measure_on_test(n, y_test, probas[n], thresholds[n], cfg) for n in pipes])

    return {"params": params, "thresholds": thresholds, "test": test, "probas": probas,
            "pipes": pipes, "y_test": y_test, "X_test": X_test, "dev_vs_test": dev_vs_test(cfg, test)}


def save_results(cfg: dict[str, Any], r: dict[str, Any]) -> Path:
    """Tables + test probabilities (STEPS 7 and 9 reuse them) + the frozen settings."""
    out = resolve(cfg["final"]["results_dir"])
    out.mkdir(parents=True, exist_ok=True)
    r["test"].to_csv(out / "test_results.csv", index=False)
    r["dev_vs_test"].to_csv(out / "dev_vs_test.csv")
    pd.DataFrame({**{n: np.round(p, 5) for n, p in r["probas"].items()}, "target": r["y_test"].values},
                 index=r["y_test"].index).to_csv(out / "test_predictions.csv")
    final = cfg["final"]["model"]
    frozen = {"model": final, "capital": cfg["features"]["capital"],
              "threshold": float(r["thresholds"][final]), "params": r["params"][final],
              "rule": "predict 1 (call) when P(>50K) >= threshold; cost_fn = 5, cost_fp = 1"}
    with open(out / "final_model.yaml", "w") as f:
        yaml.safe_dump(frozen, f, sort_keys=False)
    return out


def log_to_mlflow(cfg: dict[str, Any], r: dict[str, Any]) -> None:
    """One MLflow run per model measured on the test set (requirement 7)."""
    setup_mlflow(cfg)
    for _, row in r["test"].iterrows():
        name = row["model"]
        base = name.replace("_amounts", "")
        with mlflow.start_run(run_name=f"step6_test_{name}"):
            mlflow.set_tags({"step": "6", "model": base, "set": "test",
                             "role": "final" if name == cfg["final"]["model"] else "reference"})
            mlflow.log_params({**{f"model_{k}": v for k, v in r["params"][base].items()},
                               "threshold": row["threshold"],
                               "capital": "amounts" if name.endswith("_amounts") else cfg["features"]["capital"]})
            for m in ["pr_auc", "roc_auc", "log_loss", "precision", "recall", "f1", "cost", "cost_per_1000"]:
                mlflow.log_metric(f"test_{m}", float(row[m]))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    r = run_final(cfg)
    out = save_results(cfg, r)
    log_to_mlflow(cfg, r)
    cols = ["model", "threshold", "pr_auc", "roc_auc", "log_loss", "recall", "precision", "cost"]
    logger.info("Test set results:\n%s", r["test"][cols].round(3).to_string(index=False))
    logger.info("Saved in %s", out)


if __name__ == "__main__":
    main()
