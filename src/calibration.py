"""STEP 7 - Calibration diagnostics of the final model (Session 7, requirement 9).

Question (Session 7): when the model says "30% chance of earning >50K", is it really about 30%?
Ranking (PR-AUC, ROC-AUC) and calibration are different questions: a model can rank well and
still give probabilities that are too extreme (over-confident) or not extreme enough (under-confident).

Tools, all from Session 7:
- reliability diagram: 10 equal-width bins, mean predicted probability vs observed frequency,
  always shown with the number of people per bin (a point built on few people is weak evidence);
- Brier score: mean of (p - y)^2, lower is better; mixes calibration and ranking;
- ECE: sum over bins of (share of people in the bin) x |observed - predicted|;
- recalibration, learned on predictions the base model did NOT train on ("held-out or out-of-fold
  predictions", Session 7): Platt scaling p_cal = sigmoid(A s + B), with s = log(p / (1 - p)) the
  model's score, and isotonic regression (a non-decreasing step function).

Which data for what (Session 7, "calibrate on unseen predictions"):
- the calibration maps are learned and compared on the DEVELOPMENT out-of-fold predictions of
  STEP 5 (each person predicted by a model that never saw them), with the same 5 folds;
- the TEST probabilities of the final model (saved in STEP 6) are only used to measure, once,
  "before" and "after". The test set never chooses anything.

Run from the project root:  python -m src.calibration   (a few seconds, no model is refitted)
"""

from __future__ import annotations

import logging
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from src.data import get_cv, load_config, load_splits, resolve
from src.tracking import setup_mlflow

logger = logging.getLogger(__name__)
EPS = 1e-6                     # keeps p away from 0 and 1 before taking log(p / (1 - p))
METHODS = ("raw", "platt", "isotonic")


# ----------------------------------------------------------------------------------------------
# 1. Diagnostics: reliability table, ECE, Brier
# ----------------------------------------------------------------------------------------------
def reliability_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """One row per non-empty equal-width bin: mean predicted, observed frequency, gap, count."""
    y, p = np.asarray(y), np.asarray(p)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    rows = []
    for b in range(n_bins):
        in_bin = bins == b
        if in_bin.any():
            rows.append({"bin": f"{b / n_bins:.1f}-{(b + 1) / n_bins:.1f}", "n": int(in_bin.sum()),
                         "mean_predicted": p[in_bin].mean(), "observed": y[in_bin].mean()})
    table = pd.DataFrame(rows)
    table["gap"] = table["observed"] - table["mean_predicted"]   # > 0: under-confident, < 0: over
    return table


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """ECE = sum over bins of (|B_m| / n) x |observed(B_m) - predicted(B_m)| (Session 7)."""
    t = reliability_table(y, p, n_bins)
    return float((t["n"] / t["n"].sum() * t["gap"].abs()).sum())


def calibration_metrics(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> dict[str, float]:
    """Probability quality (Brier, ECE) next to ranking quality (PR-AUC, ROC-AUC), as in Session 7."""
    return {"brier": brier_score_loss(y, p), "ece": expected_calibration_error(y, p, n_bins),
            "pr_auc": average_precision_score(y, p), "roc_auc": roc_auc_score(y, p),
            "mean_predicted": float(np.mean(p)), "observed_rate": float(np.mean(y))}


# ----------------------------------------------------------------------------------------------
# 2. Recalibration maps (Session 7): learned on unseen predictions only
# ----------------------------------------------------------------------------------------------
def score(p: np.ndarray) -> np.ndarray:
    """Model score s = log(p / (1 - p)). For LightGBM it is the raw boosting output."""
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def fit_platt(p: np.ndarray, y: np.ndarray) -> LogisticRegression:
    """Platt scaling: a logistic regression on the single score s, i.e. p_cal = sigmoid(A s + B).
    No penalty (very large C): only two parameters to estimate. A = 1 and B = 0 = nothing to correct;
    A < 1 = the model was over-confident (too extreme), A > 1 = under-confident."""
    return LogisticRegression(C=1e6, max_iter=1000).fit(score(p).reshape(-1, 1), y)


def platt_parameters(platt: LogisticRegression) -> tuple[float, float]:
    return float(platt.coef_[0][0]), float(platt.intercept_[0])


def fit_isotonic(p: np.ndarray, y: np.ndarray) -> IsotonicRegression:
    """Isotonic regression: the non-decreasing step function g minimising sum (y - g(p))^2.
    Same calibrator as CalibratedClassifierCV(method="isotonic") of the Session 7 notebook."""
    return IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(p, y)


def fit_map(method: str, p: np.ndarray, y: np.ndarray) -> Any:
    if method == "raw":
        return None
    if method == "platt":
        return fit_platt(p, y)
    if method == "isotonic":
        return fit_isotonic(p, y)
    raise ValueError(f"method must be one of {METHODS}, got {method!r}")


def apply_map(method: str, fitted: Any, p: np.ndarray) -> np.ndarray:
    if method == "raw":
        return np.asarray(p, dtype=float)
    if method == "platt":
        return fitted.predict_proba(score(p).reshape(-1, 1))[:, 1]
    return fitted.predict(p)


# ----------------------------------------------------------------------------------------------
# 3. Choose the calibrator on the development set (cross-validated, same 5 folds)
# ----------------------------------------------------------------------------------------------
def cross_validated_calibration(y: pd.Series, oof: np.ndarray, folds: list[tuple[np.ndarray, np.ndarray]],
                                n_bins: int = 10) -> pd.DataFrame:
    """For each fold: learn the map on the out-of-fold predictions of the 4 other folds, measure it
    on this fold. The map is never measured on the predictions it learned from (Session 7, rule 4)."""
    y = np.asarray(y)
    rows = []
    for k, (tr, va) in enumerate(folds, start=1):
        for method in METHODS:
            fitted = fit_map(method, oof[tr], y[tr])
            rows.append({"fold": k, "method": method,
                         **calibration_metrics(y[va], apply_map(method, fitted, oof[va]), n_bins)})
    return pd.DataFrame(rows)


def choose_method(cv_table: pd.DataFrame, min_folds_better: int = 4) -> str:
    """Keep the raw probabilities unless a calibrator lowers the Brier score on at least 4 of the 5
    folds (a clear, regular gain); among those, the lowest mean Brier wins."""
    wide = cv_table.pivot(index="fold", columns="method", values="brier")
    candidates = [m for m in ("platt", "isotonic") if (wide[m] < wide["raw"]).sum() >= min_folds_better]
    if not candidates:
        return "raw"
    return min(candidates, key=lambda m: wide[m].mean())


def bayes_threshold(cfg: dict[str, Any]) -> float:
    """Session 7: with calibrated probabilities, calling costs cost_fp x (1 - p) and not calling
    costs cost_fn x p, so we call when p >= cost_fp / (cost_fp + cost_fn) = 1 / 6 for X = 5."""
    icfg = cfg["imbalance"]
    return icfg["cost_fp"] / (icfg["cost_fp"] + icfg["cost_fn"])


# ----------------------------------------------------------------------------------------------
# 4. Run everything
# ----------------------------------------------------------------------------------------------
def run_calibration(cfg: dict[str, Any]) -> dict[str, Any]:
    ccfg, model = cfg["calibration"], cfg["final"]["model"]
    n_bins = ccfg["n_bins"]
    X_dev, X_test, y_dev, y_test = load_splits(cfg)

    # Development: out-of-fold probabilities of STEP 5 (unseen by the model that produced them)
    oof = pd.read_csv(resolve(cfg["tuning"]["results_dir"]) / "oof_predictions.csv", index_col=0).loc[y_dev.index]
    p_dev = oof[model].values
    folds = list(get_cv(cfg).split(X_dev, y_dev))
    cv_table = cross_validated_calibration(y_dev, p_dev, folds, n_bins)
    chosen = choose_method(cv_table, ccfg["min_folds_better"])
    platt_ab = [platt_parameters(fit_platt(p_dev[tr], y_dev.values[tr])) for tr, _ in folds]

    # Test: probabilities of the final model (STEP 6), measured before / after each map.
    # Every map is learned on ALL development out-of-fold predictions, never on the test set.
    test = pd.read_csv(resolve(cfg["final"]["results_dir"]) / "test_predictions.csv", index_col=0)
    p_test, y_t = test[model].values, test["target"].values
    maps = {m: fit_map(m, p_dev, y_dev.values) for m in METHODS}
    p_test_by = {m: apply_map(m, maps[m], p_test) for m in METHODS}
    test_table = pd.DataFrame([{"method": m, **calibration_metrics(y_t, p_test_by[m], n_bins)} for m in METHODS])

    # Segments (Session 7): overall calibration can hide a group that is systematically off
    sex = X_test.loc[test.index, "sex"].astype(str).values
    seg = pd.DataFrame([{"sex": s, "n": int((sex == s).sum()),
                         **calibration_metrics(y_t[sex == s], p_test[sex == s], n_bins)}
                        for s in ("Male", "Female")])

    return {"model": model, "n_bins": n_bins, "cv_table": cv_table, "chosen": chosen,
            "platt_ab_per_fold": platt_ab, "platt_ab_all": platt_parameters(maps["platt"]),
            "dev_reliability": reliability_table(y_dev.values, p_dev, n_bins),
            "test_reliability": reliability_table(y_t, p_test, n_bins),
            "test_table": test_table, "segments": seg, "p_test_by": p_test_by, "y_test": y_t,
            "p_dev": p_dev, "y_dev": y_dev.values, "bayes_threshold": bayes_threshold(cfg)}


def save_results(cfg: dict[str, Any], r: dict[str, Any]) -> None:
    out = resolve(cfg["calibration"]["results_dir"])
    out.mkdir(parents=True, exist_ok=True)
    r["cv_table"].to_csv(out / "cv_calibration.csv", index=False)
    r["dev_reliability"].to_csv(out / "reliability_dev.csv", index=False)
    r["test_reliability"].to_csv(out / "reliability_test.csv", index=False)
    r["test_table"].to_csv(out / "test_calibration.csv", index=False)
    r["segments"].to_csv(out / "calibration_by_sex.csv", index=False)


def log_to_mlflow(cfg: dict[str, Any], r: dict[str, Any]) -> None:
    setup_mlflow(cfg)
    with mlflow.start_run(run_name="step7_calibration"):
        mlflow.set_tags({"step": "7", "model": r["model"], "set": "development + test"})
        a, b = r["platt_ab_all"]
        mlflow.log_params({"n_bins": r["n_bins"], "chosen_method": r["chosen"], "platt_A": round(a, 4),
                           "platt_B": round(b, 4), "bayes_threshold": round(r["bayes_threshold"], 4)})
        for _, row in r["test_table"].iterrows():
            for m in ("brier", "ece", "pr_auc"):
                mlflow.log_metric(f"test_{row['method']}_{m}", float(row[m]))
        cv_mean = r["cv_table"].groupby("method")[["brier", "ece"]].mean()
        for method, row in cv_mean.iterrows():
            mlflow.log_metric(f"cv_{method}_brier", float(row["brier"]))
            mlflow.log_metric(f"cv_{method}_ece", float(row["ece"]))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    r = run_calibration(cfg)
    save_results(cfg, r)
    log_to_mlflow(cfg, r)
    logger.info("Development (cross-validated) mean per method:\n%s",
                r["cv_table"].groupby("method")[["brier", "ece", "pr_auc"]].mean().round(4).to_string())
    logger.info("Chosen on development: %s | Platt A, B = %.3f, %.3f", r["chosen"], *r["platt_ab_all"])
    logger.info("Test set:\n%s", r["test_table"][["method", "brier", "ece", "pr_auc"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
