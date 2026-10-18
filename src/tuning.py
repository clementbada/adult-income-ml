"""STEP 5 - Hyperparameter tuning with Optuna, experiment tracking with MLflow, model comparison.

Requirements covered:
- 4: three models (logistic regression, Random Forest, LightGBM), the same metrics for all three;
- 6: Optuna tuning with cross-validation, and a nested cross-validation so that the reported score
     is not optimistic (Session 5);
- 7: every tuning trial of every model is logged in MLflow (parameters, metrics per fold, artifacts).

What Optuna does (Session 5). A hyperparameter is a setting chosen before training (C, number of
trees, depth...). Optuna proposes a set of values (a "trial"), we score it with the 5-fold CV of
STEP 1, and Optuna uses the scores of the previous trials to propose better values next time
(TPE sampler: it learns which regions of the search space give good scores). It is smarter than a
grid search, which tries every combination blindly.

How a model is tuned and compared (same rules for the three models):
1. Objective = mean PR-AUC over the 5 folds of the development set (main metric, decision 1).
   We tune PR-AUC and NOT the business cost: the cost depends on the threshold, and the threshold is
   chosen after the model, on its out-of-fold predictions (STEP 4 rule, decision 28).
2. Pruning: after each fold the running mean is compared with the other trials at the same fold; a
   trial clearly below the median is stopped early (MedianPruner) to save time.
3. The best hyperparameters are refitted with the same folds to get out-of-fold predictions, then
   the STEP 4 rule is applied: best threshold for the business cost (X = 5) and its cost per fold.
4. Selection bias: the best of N trials is partly lucky on these 5 folds, so its score is a bit
   optimistic. Nested cross-validation (Session 5) measures the score of the whole procedure
   "tune with Optuna, then fit": for each outer fold, Optuna runs on the other 4 folds only (inner
   CV), and the chosen model is scored on the outer fold, which played no part in the choice.
The test set is never used here (STEP 6).

Run from the project root:  python -m src.tuning
Options: --models lightgbm random_forest   (subset of models)
         --n-trials 5                      (override the number of trials, for a quick check)
         --skip-nested                     (skip the nested cross-validation, the slowest part)
Results are saved in reports/tuning/ (read by notebooks/05_models_tuning.ipynb).
"""

from __future__ import annotations

import argparse
import logging
import tempfile
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import optuna
import pandas as pd
import yaml
from sklearn.model_selection import BaseCrossValidator, StratifiedKFold

from src.data import resolve
from src.evaluate import (
    METRIC_NAMES,
    best_threshold,
    compute_metrics,
    cross_validate_oof,
    summarize,
    threshold_table,
)
from src.imbalance import cross_validated_threshold
from src.models import MODEL_BUILDERS, build_model
from src.tracking import log_cv_run

logger = logging.getLogger(__name__)

MODEL_NAMES = list(MODEL_BUILDERS)


# ----------------------------------------------------------------------------------------------
# 1. Search spaces: read from config.yaml (section tuning.search_spaces)
# ----------------------------------------------------------------------------------------------
def suggest_params(trial: optuna.Trial, space: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Ask Optuna for one value per hyperparameter, following the search space of config.yaml.

    Each entry has a type:
    - float: a number between low and high (log: true = explored on a log scale, for values that
      span several orders of magnitude, such as C from 0.001 to 100);
    - int: a whole number between low and high (optional step, optional log scale);
    - categorical: one value among a list (choices).
    """
    params = {}
    for name, spec in space.items():
        kind = spec["type"]
        if kind == "float":
            params[name] = trial.suggest_float(name, float(spec["low"]), float(spec["high"]),
                                               log=spec.get("log", False))
        elif kind == "int":
            params[name] = trial.suggest_int(name, int(spec["low"]), int(spec["high"]),
                                             step=spec.get("step", 1), log=spec.get("log", False))
        elif kind == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError(f"Unknown type {kind!r} for hyperparameter {name!r}")
    return params


def n_trials_for(cfg: dict[str, Any], model: str, nested: bool = False) -> int:
    """Number of Optuna trials for a model (config.yaml, section tuning)."""
    tcfg = cfg["tuning"]
    return int((tcfg["nested"]["n_trials"] if nested else tcfg["n_trials"])[model])


def make_study(cfg: dict[str, Any], name: str) -> optuna.Study:
    """An Optuna study that maximises PR-AUC.

    - TPESampler with a fixed seed: the same trials are proposed every time we run (reproducible);
      the first `n_startup_trials` are random, to explore before TPE starts to learn;
    - MedianPruner: from the fold `pruner_warmup_folds` on, a trial whose running mean PR-AUC is
      below the median of the previous trials at the same fold is stopped.
    """
    tcfg = cfg["tuning"]
    sampler = optuna.samplers.TPESampler(seed=tcfg["sampler_seed"],
                                         n_startup_trials=tcfg["n_startup_trials"])
    pruner = optuna.pruners.MedianPruner(n_startup_trials=tcfg["n_startup_trials"],
                                         n_warmup_steps=tcfg["pruner_warmup_folds"])
    return optuna.create_study(study_name=name, direction="maximize", sampler=sampler, pruner=pruner)


# ----------------------------------------------------------------------------------------------
# 2. One trial = one cross-validation of one set of hyperparameters
# ----------------------------------------------------------------------------------------------
def cross_validate_trial(
    trial: optuna.Trial, pipe: Any, X: pd.DataFrame, y: pd.Series, cv: BaseCrossValidator,
) -> pd.DataFrame:
    """Same logic as `cross_validate_oof` (STEP 3), fold by fold, with pruning.

    After each fold the running mean PR-AUC is reported to Optuna. If the pruner decides the trial
    is hopeless, optuna.TrialPruned is raised; the folds already done are attached to the exception
    so that the pruned trial can still be logged in MLflow.
    """
    from sklearn.base import clone

    rows = []
    for k, (tr, va) in enumerate(cv.split(X, y), start=1):
        fitted = clone(pipe).fit(X.iloc[tr], y.iloc[tr])
        proba = fitted.predict_proba(X.iloc[va])[:, 1]
        rows.append({"fold": k, **compute_metrics(y.iloc[va], proba)})
        running_mean = float(np.mean([r["pr_auc"] for r in rows]))
        trial.report(running_mean, step=k)
        if trial.should_prune():
            exc = optuna.TrialPruned(f"pruned after fold {k} (running PR-AUC {running_mean:.4f})")
            exc.per_fold = pd.DataFrame(rows).set_index("fold")
            raise exc
    return pd.DataFrame(rows).set_index("fold")


def make_objective(
    model: str, cfg: dict[str, Any], X: pd.DataFrame, y: pd.Series, cv: BaseCrossValidator,
    log_to_mlflow: bool = True,
):
    """Build the function Optuna maximises: hyperparameters -> mean PR-AUC over the folds.

    With log_to_mlflow=True, every trial (completed or pruned) becomes one MLflow child run under
    the parent run of the model (requirement 7: "log every run ... across your tuning trials").
    """
    space = cfg["tuning"]["search_spaces"][model]

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, space)
        pipe = build_model(model, cfg, **params)
        try:
            per_fold = cross_validate_trial(trial, pipe, X, y, cv)
            state = "complete"
        except optuna.TrialPruned as exc:
            per_fold, state = exc.per_fold, "pruned"
            if log_to_mlflow:
                _log_trial(model, trial, params, per_fold, state)
            raise
        if log_to_mlflow:
            _log_trial(model, trial, params, per_fold, state)
        return float(per_fold["pr_auc"].mean())

    return objective


def _log_trial(model: str, trial: optuna.Trial, params: dict[str, Any], per_fold: pd.DataFrame,
               state: str) -> None:
    """One MLflow child run per Optuna trial: hyperparameters, metrics per fold, trial state."""
    log_cv_run(
        run_name=f"{model}_trial_{trial.number:03d}",
        params={**params, "trial_number": trial.number},
        per_fold=per_fold,
        tags={"step": "5", "model": model, "optuna_state": state,
              "folds_done": str(len(per_fold))},
        nested=True,
    )


# ----------------------------------------------------------------------------------------------
# 3. Tune one model (Optuna + MLflow)
# ----------------------------------------------------------------------------------------------
def tune_model(
    model: str, cfg: dict[str, Any], X: pd.DataFrame, y: pd.Series, cv: BaseCrossValidator,
    n_trials: int | None = None, log_to_mlflow: bool = True,
) -> optuna.Study:
    """Run the Optuna study of one model on the development set and return it.

    MLflow structure: one parent run "step5_tune_<model>" holding the search space, the best trial
    and the table of all trials (artifact); one child run per trial.
    """
    n_trials = n_trials or n_trials_for(cfg, model)
    study = make_study(cfg, f"step5_{model}")
    objective = make_objective(model, cfg, X, y, cv, log_to_mlflow=log_to_mlflow)
    if not log_to_mlflow:
        study.optimize(objective, n_trials=n_trials)
        return study

    with mlflow.start_run(run_name=f"step5_tune_{model}"):
        mlflow.set_tags({"step": "5", "model": model, "role": "optuna_study"})
        mlflow.log_params({"n_trials": n_trials, "objective": "mean PR-AUC, 5-fold CV",
                           "sampler": "TPE", "pruner": "MedianPruner",
                           "sampler_seed": cfg["tuning"]["sampler_seed"]})
        study.optimize(objective, n_trials=n_trials)
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metric("best_cv_pr_auc", study.best_value)
        mlflow.log_metric("n_pruned", sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f"trials_{model}.csv"
            trials_table(study).to_csv(path)
            mlflow.log_artifact(str(path))
            space_path = Path(tmp) / "search_space.yaml"
            space_path.write_text(yaml.safe_dump(cfg["tuning"]["search_spaces"][model]))
            mlflow.log_artifact(str(space_path))
    logger.info("%s: best mean PR-AUC %.4f with %s", model, study.best_value, study.best_params)
    return study


def trials_table(study: optuna.Study) -> pd.DataFrame:
    """One row per trial: number, state, score, hyperparameters (readable version of the study)."""
    rows = [{"trial": t.number, "state": t.state.name, "pr_auc": t.value,
             "folds_done": len(t.intermediate_values), **t.params} for t in study.trials]
    return pd.DataFrame(rows).set_index("trial")


# ----------------------------------------------------------------------------------------------
# 4. Compare the tuned models with the same metrics and the STEP 4 business rule
# ----------------------------------------------------------------------------------------------
def evaluate_tuned(
    model: str, params: dict[str, Any], cfg: dict[str, Any], X: pd.DataFrame, y: pd.Series,
    cv: BaseCrossValidator,
) -> dict[str, Any]:
    """Refit the best hyperparameters on the 5 folds and apply the STEP 4 rule.

    Returns the per-fold metrics, the out-of-fold probabilities, the cost-minimising threshold and
    the cost per fold at that threshold, plus the honest (cross-validated) threshold check of STEP 4.
    """
    icfg = cfg["imbalance"]
    per_fold, oof = cross_validate_oof(build_model(model, cfg, **params), X, y, cv)
    t = best_threshold(y, oof, icfg["cost_fn"], icfg["cost_fp"])
    folds = [va for _, va in cv.split(X, y)]
    fold_cost = np.array([threshold_table(y.iloc[va], oof[va], icfg["cost_fn"], icfg["cost_fp"],
                                          np.array([t]))["cost"].iloc[0] for va in folds])
    honest = cross_validated_threshold(y, oof, cv, X, icfg["cost_fn"], icfg["cost_fp"])
    return {"per_fold": per_fold, "oof": oof, "threshold": t, "fold_cost": fold_cost,
            "honest": honest}


def comparison_table(results: dict[str, dict[str, Any]], y: pd.Series, cfg: dict[str, Any]) -> pd.DataFrame:
    """The same metrics for every model (requirement 4): one row per model."""
    icfg = cfg["imbalance"]
    rows = []
    for model, r in results.items():
        s = summarize(r["per_fold"])
        op = threshold_table(y, r["oof"], icfg["cost_fn"], icfg["cost_fp"],
                             thresholds=np.array([r["threshold"]])).iloc[0]
        rows.append({
            "model": model,
            **{f"{m}_mean": s.loc[m, "mean"] for m in METRIC_NAMES[:3]},
            "pr_auc_std": s.loc["pr_auc", "std"],
            "threshold": r["threshold"], "calls": int(op["calls"]), "found": int(op["tp"]),
            "missed": int(op["fn"]), "useless_calls": int(op["fp"]),
            "recall": op["recall"], "precision": op["precision"], "cost": op["cost"],
            "fold_cost_mean": r["fold_cost"].mean(), "fold_cost_std": r["fold_cost"].std(ddof=1),
            "cost_honest": r["honest"]["cost_honest"].sum(),
        })
    return pd.DataFrame(rows).set_index("model")


def paired_differences(
    results: dict[str, dict[str, Any]], reference: str = "logistic",
) -> pd.DataFrame:
    """Paired comparison with the reference model, fold by fold.

    The models are evaluated on the SAME folds, so the right question is "on each fold, by how much
    does model B beat model A?" (paired), not "do the two means differ by more than the
    fold-to-fold spread?" (unpaired, too pessimistic: some folds are harder for every model).
    """
    ref = results[reference]
    rows = []
    for model, r in results.items():
        if model == reference:
            continue
        d_pr = r["per_fold"]["pr_auc"].to_numpy() - ref["per_fold"]["pr_auc"].to_numpy()
        d_cost = r["fold_cost"] - ref["fold_cost"]
        rows.append({
            "model": model, "vs": reference,
            "pr_auc_diff_mean": d_pr.mean(), "pr_auc_diff_std": d_pr.std(ddof=1),
            "folds_better_pr_auc": f"{int((d_pr > 0).sum())}/{len(d_pr)}",
            "cost_diff_mean": d_cost.mean(), "cost_diff_std": d_cost.std(ddof=1),
            "folds_lower_cost": f"{int((d_cost < 0).sum())}/{len(d_cost)}",
            "pr_auc_diff_per_fold": ", ".join(f"{v:+.4f}" for v in d_pr),
            "cost_diff_per_fold": ", ".join(f"{v:+.0f}" for v in d_cost),
        })
    return pd.DataFrame(rows).set_index("model")


def log_final_run(model: str, params: dict[str, Any], r: dict[str, Any], cfg: dict[str, Any]) -> None:
    """One MLflow run per tuned model with the STEP 4 operating point (threshold, business cost)."""
    run_id = log_cv_run(
        run_name=f"step5_final_{model}",
        params={**params, "cost_fn": cfg["imbalance"]["cost_fn"], "cost_fp": cfg["imbalance"]["cost_fp"]},
        per_fold=r["per_fold"],
        tags={"step": "5", "model": model, "role": "tuned_model"},
    )
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics({"chosen_threshold": r["threshold"],
                            "business_cost_fold_mean": float(r["fold_cost"].mean()),
                            "business_cost_honest": float(r["honest"]["cost_honest"].sum())})


# ----------------------------------------------------------------------------------------------
# 5. LightGBM and class imbalance: is_unbalance (= scale_pos_weight), the open point of STEP 4
# ----------------------------------------------------------------------------------------------
def lightgbm_imbalance_check(
    params: dict[str, Any], cfg: dict[str, Any], X: pd.DataFrame, y: pd.Series,
    cv: BaseCrossValidator, base: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Tuned LightGBM with and without class reweighting, each with its own best threshold.

    is_unbalance=True makes LightGBM weigh the positive class by n_negative / n_positive, computed on
    the training folds only (the same idea as scale_pos_weight, Session 6, without hard-coding 3.2).
    Same decision rule as STEP 4: keep "threshold only" unless reweighting clearly lowers the cost.
    """
    base = base or evaluate_tuned("lightgbm", params, cfg, X, y, cv)
    weighted = evaluate_tuned("lightgbm", {**params, "is_unbalance": True}, cfg, X, y, cv)
    table = comparison_table({"lightgbm_threshold_only": base, "lightgbm_is_unbalance": weighted}, y, cfg)
    d_cost = weighted["fold_cost"] - base["fold_cost"]
    table["cost_diff_vs_threshold_only_per_fold"] = ["-", ", ".join(f"{v:+.0f}" for v in d_cost)]
    return table, weighted


# ----------------------------------------------------------------------------------------------
# 6. Nested cross-validation: an unbiased score for the whole "tune, then fit" procedure
# ----------------------------------------------------------------------------------------------
def nested_cv(
    model: str, cfg: dict[str, Any], X: pd.DataFrame, y: pd.Series, outer_cv: BaseCrossValidator,
    n_trials: int | None = None,
) -> pd.DataFrame:
    """For each outer fold: Optuna on the other folds (inner CV), refit, score on the outer fold.

    The outer folds are the 5 folds of STEP 1. The inner CV has `inner_folds` folds (3 by default,
    fewer than 5 to keep the cost reasonable: 5 outer x n_trials x 3 inner fits). Optuna never sees
    the outer fold, so its PR-AUC is an honest estimate (Session 5: "the reported performance comes
    from the outer folds, not from the inner CV scores used for tuning").
    The inner trials are not logged one by one in MLflow (they only serve the estimate); the outer
    scores are logged as one run per model by `main`.
    """
    tcfg = cfg["tuning"]
    n_trials = n_trials or n_trials_for(cfg, model, nested=True)
    rows = []
    for k, (tr, va) in enumerate(outer_cv.split(X, y), start=1):
        X_tr, y_tr = X.iloc[tr], y.iloc[tr]
        inner_cv = StratifiedKFold(n_splits=tcfg["nested"]["inner_folds"], shuffle=True,
                                   random_state=cfg["seed"])
        study = make_study(cfg, f"nested_{model}_outer{k}")
        study.optimize(make_objective(model, cfg, X_tr, y_tr, inner_cv, log_to_mlflow=False),
                       n_trials=n_trials)
        fitted = build_model(model, cfg, **study.best_params).fit(X_tr, y_tr)
        proba = fitted.predict_proba(X.iloc[va])[:, 1]
        outer = compute_metrics(y.iloc[va], proba)
        rows.append({"outer_fold": k, "inner_best_pr_auc": study.best_value,
                     "outer_pr_auc": outer["pr_auc"], "outer_roc_auc": outer["roc_auc"],
                     "outer_log_loss": outer["log_loss"], "best_params": str(study.best_params)})
        logger.info("Nested %s, outer fold %d: inner %.4f -> outer %.4f", model, k,
                    study.best_value, outer["pr_auc"])
    return pd.DataFrame(rows).set_index("outer_fold")


# ----------------------------------------------------------------------------------------------
# 7. Everything, from the command line
# ----------------------------------------------------------------------------------------------
def main() -> None:
    """Tune the three models, compare them, run the LightGBM imbalance check and the nested CV.
    Save every table in reports/tuning/ and log everything in MLflow."""
    from src.data import get_cv, load_config, load_splits
    from src.tracking import setup_mlflow

    parser = argparse.ArgumentParser(description="STEP 5: Optuna tuning and model comparison")
    parser.add_argument("--models", nargs="+", default=MODEL_NAMES, choices=MODEL_NAMES)
    parser.add_argument("--n-trials", type=int, default=None, help="override every n_trials (quick check)")
    parser.add_argument("--skip-nested", action="store_true", help="skip the nested cross-validation")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    pd.set_option("display.width", 250)                         # print the tables in full
    pd.set_option("display.max_columns", 30)
    optuna.logging.set_verbosity(optuna.logging.WARNING)       # our own log lines are enough
    cfg = load_config()
    X_dev, _, y_dev, _ = load_splits(cfg)                       # the test set is not used in STEP 5
    cv = get_cv(cfg)
    out_dir = resolve(cfg["tuning"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_mlflow(cfg)

    # 1. Tuning (one Optuna study per model, every trial in MLflow)
    best_params = {}
    for model in args.models:
        study = tune_model(model, cfg, X_dev, y_dev, cv, n_trials=args.n_trials)
        best_params[model] = study.best_params
        trials_table(study).to_csv(out_dir / f"trials_{model}.csv")
    (out_dir / "best_params.yaml").write_text(yaml.safe_dump(best_params, sort_keys=False))

    # 2. Comparison with the same metrics and the STEP 4 business rule
    results = {m: evaluate_tuned(m, best_params[m], cfg, X_dev, y_dev, cv) for m in args.models}
    for m, r in results.items():
        log_final_run(m, best_params[m], r, cfg)
        r["per_fold"].to_csv(out_dir / f"per_fold_{m}.csv")
    pd.DataFrame({m: r["oof"] for m, r in results.items()}, index=y_dev.index).round(5) \
        .assign(target=y_dev).to_csv(out_dir / "oof_predictions.csv")    # used for the plots of notebook 05
    comparison = comparison_table(results, y_dev, cfg)
    comparison.to_csv(out_dir / "comparison.csv")
    logger.info("Comparison of the tuned models:\n%s", comparison.round(4))
    if "logistic" in results and len(results) > 1:
        paired = paired_differences(results, reference="logistic")
        paired.to_csv(out_dir / "paired_vs_logistic.csv")
        logger.info("Paired differences vs logistic regression:\n%s", paired.round(4))

    # 3. LightGBM with / without class reweighting (open point of STEP 4)
    if "lightgbm" in results:
        lgbm_table, _ = lightgbm_imbalance_check(best_params["lightgbm"], cfg, X_dev, y_dev, cv,
                                                 base=results["lightgbm"])
        lgbm_table.to_csv(out_dir / "lightgbm_imbalance.csv")
        logger.info("LightGBM, threshold only vs is_unbalance:\n%s", lgbm_table.round(4))

    # 4. Nested cross-validation (honest score of the tuning procedure)
    if not args.skip_nested:
        nested_rows = []
        for model in args.models:
            n_nested = max(2, args.n_trials // 2) if args.n_trials else n_trials_for(cfg, model, True)
            table = nested_cv(model, cfg, X_dev, y_dev, cv, n_trials=n_nested)
            table.to_csv(out_dir / f"nested_{model}.csv")
            nested_rows.append({"model": model,
                                "tuning_cv_pr_auc": results[model]["per_fold"]["pr_auc"].mean(),
                                "nested_pr_auc_mean": table["outer_pr_auc"].mean(),
                                "nested_pr_auc_std": table["outer_pr_auc"].std(ddof=1)})
            with mlflow.start_run(run_name=f"step5_nested_{model}"):
                mlflow.set_tags({"step": "5", "model": model, "role": "nested_cv"})
                mlflow.log_params({"inner_folds": cfg["tuning"]["nested"]["inner_folds"],
                                   "n_trials_per_outer_fold": n_nested})
                for k, v in table["outer_pr_auc"].items():
                    mlflow.log_metric("outer_pr_auc", float(v), step=int(k))
                mlflow.log_metric("nested_pr_auc_mean", float(table["outer_pr_auc"].mean()))
        nested = pd.DataFrame(nested_rows).set_index("model")
        nested["optimism"] = nested["tuning_cv_pr_auc"] - nested["nested_pr_auc_mean"]
        nested.to_csv(out_dir / "nested_summary.csv")
        logger.info("Nested cross-validation:\n%s", nested.round(4))


if __name__ == "__main__":
    main()
