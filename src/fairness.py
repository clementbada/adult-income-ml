"""STEP 9 - Fairness and bias audit of the final model (Session 9, requirement 10).

Session 9 order: measure by group -> choose a fairness definition -> mitigate -> re-audit on held-out data.

1. Measure (Fairlearn MetricFrame, Session 9): for each sex, at the business threshold, the four
   quantities behind the three families of definitions:
   - selection rate (share of people called)           -> demographic parity;
   - TPR = recall (share of wealthy people called)     -> equal opportunity;
   - TPR and FPR (share of non-wealthy people called)  -> equalized odds;
   - PPV = precision (share of calls that are wealthy) -> predictive parity.
   `race` is reported the same way (with group sizes), without mitigation (decision 10).
2. Definition, chosen from the cost structure BEFORE looking at the model (Session 9):
   a missed wealthy prospect costs 5 calls, so the harm is "being a wealthy person the bank never
   contacts" -> EQUAL OPPORTUNITY (equal recall for wealthy women and wealthy men).
3. Mitigation, in-processing (Session 9, slide "in-processing"): the training loss is reweighted by
   label AND group, w(y, a). We give a weight `lam` to wealthy women (y = 1, a = Female) and 1 to
   everyone else. `sex` is used ONLY to build these training weights: it is still not a model input,
   and it is not needed at decision time (one single threshold for everybody, decision 8).
   `lam` is chosen on the DEVELOPMENT set, with the same 5 folds as before: out-of-fold predictions
   for each `lam`, threshold re-chosen with the STEP 4 business rule, gap measured out-of-fold.
   Rule fixed in advance: the smallest `lam` whose out-of-fold recall gap is <= max_tpr_gap.
4. Re-audit: the mitigated model is refitted on the whole development set and measured ONCE on the
   untouched test set, next to the STEP 6 model (before / after).

Run from the project root:  python -m src.fairness   (about 2 minutes)
"""

from __future__ import annotations

import logging
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from fairlearn.metrics import (MetricFrame, count, false_positive_rate, selection_rate,
                               true_positive_rate)
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, precision_score
from sklearn.pipeline import Pipeline

from src.calibration import expected_calibration_error
from src.data import get_cv, load_config, load_splits, resolve
from src.evaluate import best_threshold
from src.final import fit_on_development, load_best_params
from src.models import build_model
from src.tracking import setup_mlflow

logger = logging.getLogger(__name__)
MARRIED = ["Married-civ-spouse", "Married-AF-spouse"]


# ----------------------------------------------------------------------------------------------
# 1. Measure by group (Session 9: Fairlearn MetricFrame)
# ----------------------------------------------------------------------------------------------
def _base_rate(y_true, y_pred) -> float:
    return float(np.mean(y_true))


def _ppv(y_true, y_pred) -> float:
    return float(precision_score(y_true, y_pred, zero_division=0))


def _wealthy(y_true, y_pred) -> int:
    return int(np.sum(y_true))


FAIRNESS_METRICS = {"n": count, "wealthy": _wealthy, "base_rate": _base_rate,
                    "selection_rate": selection_rate, "tpr": true_positive_rate,
                    "fpr": false_positive_rate, "ppv": _ppv}


def group_metrics(y: np.ndarray, decision: np.ndarray, groups: np.ndarray) -> MetricFrame:
    """Size, base rate, selection rate, TPR (recall), FPR, PPV (precision) for each group."""
    return MetricFrame(metrics=FAIRNESS_METRICS, y_true=np.asarray(y), y_pred=np.asarray(decision),
                       sensitive_features=pd.Series(np.asarray(groups), name="group"))


def gap_summary(mf: MetricFrame) -> dict[str, float]:
    """One number per fairness definition (Session 9). Differences are max - min across groups."""
    g = mf.by_group
    sel = g["selection_rate"]
    return {"tpr_gap": float(g["tpr"].max() - g["tpr"].min()),                    # equal opportunity
            "fpr_gap": float(g["fpr"].max() - g["fpr"].min()),
            "equalized_odds_gap": float(max(g["tpr"].max() - g["tpr"].min(),     # equalized odds
                                            g["fpr"].max() - g["fpr"].min())),
            "selection_rate_ratio": float(sel.min() / sel.max()),                 # four-fifths rule
            "ppv_gap": float(g["ppv"].max() - g["ppv"].min())}                   # predictive parity


def recall_within_strata(y: np.ndarray, decision: np.ndarray, sex: np.ndarray,
                         marital: pd.Series) -> pd.DataFrame:
    """Recall of wealthy women and wealthy men INSIDE each marital-status group (married / not married).
    If the gap disappears inside each group, the overall gap comes from the different composition of
    the two sexes (a proxy effect, Session 9), not from different treatment of similar people."""
    married = np.where(marital.astype(str).isin(MARRIED).values, "married", "not married")
    mf = MetricFrame(metrics={"wealthy": _wealthy, "tpr": true_positive_rate},
                     y_true=np.asarray(y), y_pred=np.asarray(decision),
                     sensitive_features=pd.DataFrame({"marital status": married, "sex": np.asarray(sex)}))
    return mf.by_group


def bootstrap_tpr_gap(y: np.ndarray, decision: np.ndarray, sex: np.ndarray, n_boot: int,
                      seed: int) -> tuple[float, float]:
    """95% interval of the recall gap (men - women): resample the test people with replacement
    (the bootstrap idea of Session 4) and recompute the gap each time."""
    rng = np.random.default_rng(seed)
    y, decision, sex = np.asarray(y), np.asarray(decision), np.asarray(sex)
    gaps = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        yb, db, sb = y[i], decision[i], sex[i]
        tpr = {s: db[(sb == s) & (yb == 1)].mean() for s in ("Male", "Female")}
        gaps[b] = tpr["Male"] - tpr["Female"]
    return float(np.percentile(gaps, 2.5)), float(np.percentile(gaps, 97.5))


def business_cost(y: np.ndarray, decision: np.ndarray, cfg: dict[str, Any]) -> float:
    """STEP 4 cost: cost_fn per missed wealthy prospect + cost_fp per useless call."""
    y, d = np.asarray(y), np.asarray(decision)
    icfg = cfg["imbalance"]
    return float(icfg["cost_fn"] * ((d == 0) & (y == 1)).sum() + icfg["cost_fp"] * ((d == 1) & (y == 0)).sum())


# ----------------------------------------------------------------------------------------------
# 2. Mitigation: reweight the loss by (label, group), choose the weight on the development set
# ----------------------------------------------------------------------------------------------
def group_label_weights(y: np.ndarray, sex: np.ndarray, lam: float, group: str) -> np.ndarray:
    """Session 9 in-processing weights w(y, a): `lam` for (y = 1, a = group), 1 for everyone else.
    lam = 1 gives back the STEP 6 model exactly."""
    return np.where((np.asarray(y) == 1) & (np.asarray(sex) == group), float(lam), 1.0)


def fit_weighted(cfg: dict[str, Any], params: dict[str, Any], X: pd.DataFrame, y: pd.Series,
                 weights: np.ndarray) -> Pipeline:
    """Same Pipeline and hyperparameters as the final model; only the training weights change."""
    return build_model(cfg["final"]["model"], cfg, **params).fit(X, y, model__sample_weight=weights)


def weighted_oof(cfg: dict[str, Any], params: dict[str, Any], X: pd.DataFrame, y: pd.Series,
                 weights: np.ndarray, folds: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Out-of-fold probabilities with the same 5 folds as STEPS 3-5 (each person predicted by a
    model that never saw them). Weights are used for training only, never for measuring."""
    oof = np.zeros(len(y))
    for tr, va in folds:
        pipe = fit_weighted(cfg, params, X.iloc[tr], y.iloc[tr], weights[tr])
        oof[va] = pipe.predict_proba(X.iloc[va])[:, 1]
    return oof


def tradeoff_curve(cfg: dict[str, Any], params: dict[str, Any], X_dev: pd.DataFrame, y_dev: pd.Series,
                   folds: list[tuple[np.ndarray, np.ndarray]]) -> tuple[pd.DataFrame, dict[float, np.ndarray]]:
    """Session 9 fairness-accuracy trade-off, on the development set only: one row per weight `lam`.
    For each `lam`: out-of-fold probabilities, threshold re-chosen with the STEP 4 business rule,
    then recall by sex, recall gap, business cost, PR-AUC and log-loss (all out-of-fold)."""
    fcfg, icfg = cfg["fairness"], cfg["imbalance"]
    sex = X_dev[fcfg["sensitive"]].astype(str).values
    y = y_dev.values
    rows, oofs = [], {}
    for lam in fcfg["lambdas"]:
        w = group_label_weights(y, sex, lam, fcfg["group"])
        oof = weighted_oof(cfg, params, X_dev, y_dev, w, folds)
        t = best_threshold(y_dev, oof, icfg["cost_fn"], icfg["cost_fp"])
        d = (oof >= t).astype(int)
        g = group_metrics(y, d, sex).by_group
        rows.append({"lam": lam, "threshold": t, "tpr_female": g.loc["Female", "tpr"],
                     "tpr_male": g.loc["Male", "tpr"], "tpr_gap": g.loc["Male", "tpr"] - g.loc["Female", "tpr"],
                     "cost": business_cost(y, d, cfg), "pr_auc": average_precision_score(y, oof),
                     "log_loss": log_loss(y, oof), "calls": int(d.sum())})
        oofs[lam] = oof
        logger.info("lam %.1f: threshold %.2f, recall gap %.3f, cost %.0f", lam, t, rows[-1]["tpr_gap"], rows[-1]["cost"])
    curve = pd.DataFrame(rows).set_index("lam")
    curve["cost_vs_unweighted"] = curve["cost"] / curve["cost"].iloc[0] - 1
    return curve, oofs


def choose_lambda(curve: pd.DataFrame, max_tpr_gap: float) -> float:
    """Rule fixed before the test set: the smallest weight whose out-of-fold recall gap is within
    `max_tpr_gap` (the least intervention that reaches the target, as in the S6 hierarchy).
    If no weight reaches it, the weight with the smallest gap."""
    ok = curve[curve["tpr_gap"].abs() <= max_tpr_gap]
    return float(ok.index.min()) if len(ok) else float(curve["tpr_gap"].abs().idxmin())


# ----------------------------------------------------------------------------------------------
# 3. Run everything
# ----------------------------------------------------------------------------------------------
def audit(name: str, y: np.ndarray, proba: np.ndarray, threshold: float, sex: np.ndarray,
          cfg: dict[str, Any]) -> tuple[MetricFrame, dict[str, Any]]:
    """Audit one model on the test set: metrics by sex + one summary row (performance + gaps)."""
    d = (proba >= threshold).astype(int)
    mf = group_metrics(y, d, sex)
    low, high = bootstrap_tpr_gap(y, d, sex, cfg["fairness"]["bootstrap"], cfg["seed"])
    row = {"model": name, "threshold": threshold, "pr_auc": average_precision_score(y, proba),
           "log_loss": log_loss(y, proba), "brier": brier_score_loss(y, proba),
           "ece": expected_calibration_error(y, proba, cfg["calibration"]["n_bins"]),
           "recall": float(d[y == 1].mean()), "precision": _ppv(y, d), "calls": int(d.sum()),
           "cost": business_cost(y, d, cfg), **gap_summary(mf),
           "tpr_gap_ci_low": low, "tpr_gap_ci_high": high}
    return mf, row


def run_fairness(cfg: dict[str, Any]) -> dict[str, Any]:
    fcfg = cfg["fairness"]
    final = cfg["final"]["model"]
    X_dev, X_test, y_dev, y_test = load_splits(cfg)
    params = load_best_params(cfg, final)
    folds = list(get_cv(cfg).split(X_dev, y_dev))
    sex_dev = X_dev[fcfg["sensitive"]].astype(str).values
    sex_test = X_test[fcfg["sensitive"]].astype(str).values
    y_t = y_test.values

    # BEFORE: the STEP 6 model (test probabilities saved in STEP 6, threshold 0.15)
    test_pred = pd.read_csv(resolve(cfg["final"]["results_dir"]) / "test_predictions.csv", index_col=0).loc[y_test.index]
    p_before = test_pred[final].values
    t_before = float(pd.read_csv(resolve(cfg["final"]["results_dir"]) / "test_results.csv")
                     .set_index("model").loc[final, "threshold"])
    mf_before, row_before = audit("before (STEP 6 model)", y_t, p_before, t_before, sex_test, cfg)
    d_before = (p_before >= t_before).astype(int)
    strata_before = recall_within_strata(y_t, d_before, sex_test, X_test["marital-status"])

    # Development-set view of the same question (out-of-fold, STEP 5): is the gap stable?
    oof_dev = pd.read_csv(resolve(cfg["tuning"]["results_dir"]) / "oof_predictions.csv", index_col=0).loc[y_dev.index, final].values
    d_dev = (oof_dev >= t_before).astype(int)
    mf_dev = group_metrics(y_dev.values, d_dev, sex_dev)
    strata_dev = recall_within_strata(y_dev.values, d_dev, sex_dev, X_dev["marital-status"])

    # Secondary attribute: race, before only, with group sizes (decision 10)
    mf_race = group_metrics(y_t, d_before, X_test[fcfg["secondary"]].astype(str).values)

    # MITIGATION: choose the weight on the development set (out-of-fold), never on the test set
    curve, _ = tradeoff_curve(cfg, params, X_dev, y_dev, folds)
    lam = choose_lambda(curve, fcfg["max_tpr_gap"])
    t_after = float(curve.loc[lam, "threshold"])

    # AFTER: refit on the whole development set with the chosen weights, ONE look at the test set
    w = group_label_weights(y_dev.values, sex_dev, lam, fcfg["group"])
    pipe_after = fit_weighted(cfg, params, X_dev, y_dev, w)
    p_after = pipe_after.predict_proba(X_test)[:, 1]
    mf_after, row_after = audit(f"after (lam = {lam:g})", y_t, p_after, t_after, sex_test, cfg)
    strata_after = recall_within_strata(y_t, (p_after >= t_after).astype(int), sex_test, X_test["marital-status"])

    # Sanity check: lam = 1 must give back the STEP 6 model (same Pipeline, same data, weights all 1)
    pipe_check = fit_on_development(cfg, final, params, X_dev, y_dev)
    reproduces_step6 = bool(np.allclose(pipe_check.predict_proba(X_test)[:, 1], p_before, atol=1e-4))

    comparison = pd.DataFrame([row_before, row_after]).set_index("model")
    return {"lam": lam, "threshold_before": t_before, "threshold_after": t_after, "curve": curve,
            "mf_before": mf_before, "mf_after": mf_after, "mf_dev": mf_dev, "mf_race": mf_race,
            "strata_before": strata_before, "strata_after": strata_after, "strata_dev": strata_dev,
            "comparison": comparison, "p_before": p_before, "p_after": p_after, "y_test": y_t,
            "sex_test": sex_test, "reproduces_step6": reproduces_step6}


def save_results(cfg: dict[str, Any], r: dict[str, Any]) -> None:
    out = resolve(cfg["fairness"]["results_dir"])
    out.mkdir(parents=True, exist_ok=True)
    by_sex = pd.concat({"before": r["mf_before"].by_group, "after": r["mf_after"].by_group}, names=["model"])
    by_sex.to_csv(out / "metrics_by_sex_test.csv")
    r["mf_dev"].by_group.to_csv(out / "metrics_by_sex_dev_oof.csv")
    r["mf_race"].by_group.to_csv(out / "metrics_by_race_test.csv")
    strata = pd.concat({"test before": r["strata_before"], "test after": r["strata_after"],
                        "dev out-of-fold": r["strata_dev"]}, names=["set"])
    strata.to_csv(out / "recall_by_sex_within_marital_status.csv")
    r["curve"].to_csv(out / "tradeoff_curve_dev.csv")
    r["comparison"].to_csv(out / "before_after_test.csv")


def log_to_mlflow(cfg: dict[str, Any], r: dict[str, Any]) -> None:
    """One run per weight of the trade-off curve (development) + one run per model on the test set."""
    setup_mlflow(cfg)
    for lam, row in r["curve"].iterrows():
        with mlflow.start_run(run_name=f"step9_dev_lam_{lam:g}"):
            mlflow.set_tags({"step": "9", "set": "development (out-of-fold)", "mitigation": "label-group reweighting"})
            mlflow.log_params({"lam": lam, "group": cfg["fairness"]["group"], "threshold": row["threshold"]})
            mlflow.log_metrics({k: float(row[k]) for k in ["tpr_female", "tpr_male", "tpr_gap", "cost", "pr_auc", "log_loss"]})
    for name, row in r["comparison"].iterrows():
        with mlflow.start_run(run_name=f"step9_test_{'before' if name.startswith('before') else 'after'}"):
            mlflow.set_tags({"step": "9", "set": "test", "model": name})
            mlflow.log_params({"lam": 1.0 if name.startswith("before") else r["lam"], "threshold": row["threshold"]})
            mlflow.log_metrics({k: float(row[k]) for k in ["pr_auc", "recall", "precision", "cost", "tpr_gap",
                                                           "fpr_gap", "selection_rate_ratio", "ppv_gap"]})


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    r = run_fairness(cfg)
    save_results(cfg, r)
    log_to_mlflow(cfg, r)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        logger.info("Trade-off curve (development, out-of-fold):\n%s", r["curve"].round(3))
        logger.info("Chosen weight: %g (threshold %.2f)", r["lam"], r["threshold_after"])
        logger.info("Test, by sex, before:\n%s", r["mf_before"].by_group.round(3))
        logger.info("Test, by sex, after:\n%s", r["mf_after"].by_group.round(3))
        logger.info("Test, before / after:\n%s", r["comparison"].round(3).T)
        logger.info("lam = 1 reproduces the STEP 6 model: %s", r["reproduces_step6"])


if __name__ == "__main__":
    main()
