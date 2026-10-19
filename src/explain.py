"""STEP 8 - Diagnostics and explainability of the final model (Session 8, requirement 8).

Session 8 asks two questions after a good global score:
1. Where does the model fail?  -> slice-based evaluation on meaningful groups, with size and
   prevalence next to each metric, then a look at the errors of the weakest slice.
2. What drives its predictions? -> three tools, each answering a different question:
   - gain importance (LightGBM's own statistic) vs permutation importance (drop of PR-AUC on the
     test set when one column is shuffled);
   - SHAP: global (mean |SHAP| per feature, beeswarm) and local (waterfall for one person);
   - partial dependence (PDP): the average relationship the model learned for one feature.

Everything is computed on the FROZEN final model of STEP 6 (LightGBM, threshold 0.15), rebuilt with
`fit_on_development`. Nothing is chosen here: the test set is only used to describe the model.

Scale (Session 8 warning): for LightGBM, SHAP explains the RAW score (log-odds), not the probability.
prediction = sigmoid(base value + sum of the SHAP values).

Run from the project root:  python -m src.explain   (about 2 minutes, mostly permutation importance)
"""

from __future__ import annotations

import logging
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.metrics import average_precision_score, precision_score, recall_score
from sklearn.pipeline import Pipeline

from src.data import load_config, load_splits, resolve
from src.features import CATEGORICAL_COLUMNS
from src.final import fit_on_development, load_best_params
from src.tracking import setup_mlflow

logger = logging.getLogger(__name__)

# Education levels for the slices: grouped from `education-num` (1-16) into readable diplomas
EDUCATION_LEVELS = [(1, 8, "no high-school diploma"), (9, 9, "high-school graduate"),
                    (10, 12, "some college / associate"), (13, 13, "bachelor"), (14, 16, "master / doctorate")]
MARRIED = ["Married-civ-spouse", "Married-AF-spouse"]


# ----------------------------------------------------------------------------------------------
# 0. The frozen final model and its inputs
# ----------------------------------------------------------------------------------------------
def final_pipeline(cfg: dict[str, Any], X_dev: pd.DataFrame, y_dev: pd.Series) -> Pipeline:
    """Same model as STEP 6: LightGBM, Optuna hyperparameters, refitted on the development set."""
    model = cfg["final"]["model"]
    return fit_on_development(cfg, model, load_best_params(cfg, model), X_dev, y_dev)


def model_inputs(pipe: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """What LightGBM actually receives: the raw columns passed through every preprocessing step."""
    Z = X
    for _, step in pipe.steps[:-1]:
        Z = step.transform(Z)
    return Z


def readable_inputs(pipe: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Same columns as `model_inputs`, but with category names instead of ordinal codes (for plots)."""
    Z = model_inputs(pipe, X).copy()
    built = pipe.named_steps["features"].transform(X)
    for col in CATEGORICAL_COLUMNS:
        if col in Z.columns:
            Z[col] = built[col].fillna("Missing").astype(str).values
    return Z


# ----------------------------------------------------------------------------------------------
# 1. Slice-based evaluation (Session 8, Part II of the notebook)
# ----------------------------------------------------------------------------------------------
def slice_masks(X: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, np.ndarray]:
    """Meaningful, decision-relevant slices (Session 8: no search over thousands of subgroups)."""
    f = cfg["features"]
    masks: dict[str, np.ndarray] = {"All": np.ones(len(X), dtype=bool)}
    age = pd.cut(X["age"], bins=list(map(float, f["age_bins"])), labels=f["age_labels"]).astype(str).values
    for label in f["age_labels"]:
        masks[f"age {label}"] = age == label
    edu = X["education-num"].values
    for lo, hi, label in EDUCATION_LEVELS:
        masks[f"education: {label}"] = (edu >= lo) & (edu <= hi)
    married = X["marital-status"].astype(str).isin(MARRIED).values
    masks["married"], masks["not married"] = married, ~married
    hours = pd.cut(X["hours-per-week"], bins=list(map(float, f["hours_bins"])),
                   labels=f["hours_labels"]).astype(str).values
    for label in f["hours_labels"]:
        masks[f"hours {label}"] = hours == label
    sex = X["sex"].astype(str).values
    masks["sex: Male"], masks["sex: Female"] = sex == "Male", sex == "Female"
    return masks


def slice_table(y: np.ndarray, proba: np.ndarray, threshold: float, masks: dict[str, np.ndarray]) -> pd.DataFrame:
    """Session 8 audit table: size, prevalence, confusion counts, precision, recall, PR-AUC per slice."""
    y, proba = np.asarray(y), np.asarray(proba)
    decision = (proba >= threshold).astype(int)
    rows = []
    for name, m in masks.items():
        ys, ds, ps = y[m], decision[m], proba[m]
        rows.append({"slice": name, "n": int(m.sum()), "prevalence": ys.mean(),
                     "wealthy": int(ys.sum()), "found": int((ds & ys).sum()), "missed": int(((1 - ds) & ys).sum()),
                     "useless_calls": int((ds & (1 - ys)).sum()),
                     "precision": precision_score(ys, ds, zero_division=0), "recall": recall_score(ys, ds),
                     "pr_auc": average_precision_score(ys, ps)})
    return pd.DataFrame(rows).set_index("slice")


def weakest_slice(table: pd.DataFrame, min_wealthy: int) -> str:
    """Weakest slice for OUR decision: a missed wealthy prospect costs 5 calls, so recall is the metric
    that matters (Session 8, "the weakest segment is metric-dependent"). Slices with too few wealthy
    people are left out (Session 8: tiny slices give noisy estimates)."""
    candidates = table.drop(index="All")
    candidates = candidates[candidates["wealthy"] >= min_wealthy]
    return str(candidates["recall"].idxmin())


def error_analysis(X: pd.DataFrame, y: np.ndarray, proba: np.ndarray, threshold: float,
                   mask: np.ndarray) -> pd.DataFrame:
    """Inside one slice: wealthy people FOUND vs wealthy people MISSED (Session 8, "inspect errors")."""
    y = np.asarray(y)
    wealthy = mask & (y == 1)
    found, missed = wealthy & (proba >= threshold), wealthy & (proba < threshold)
    desc = pd.DataFrame({
        "mean age": X["age"], "mean years of education": X["education-num"],
        "mean hours per week": X["hours-per-week"],
        "share with a capital gain": (X["capital-gain"] > 0).astype(float),
        "share manager or professional": X["occupation"].astype(str).isin(["Exec-managerial", "Prof-specialty"]).astype(float),
        "share women": (X["sex"].astype(str) == "Female").astype(float),
        "mean predicted probability": proba,
    }, index=X.index)
    table = pd.DataFrame({"wealthy found": desc[found].mean(), "wealthy missed": desc[missed].mean()}).T
    table.insert(0, "n", [int(found.sum()), int(missed.sum())])
    return table


# ----------------------------------------------------------------------------------------------
# 2. Global importance: gain vs permutation (Session 8, Part III)
# ----------------------------------------------------------------------------------------------
def gain_importance(pipe: Pipeline) -> pd.Series:
    """LightGBM's native gain importance (share of the total gain), per model input."""
    booster = pipe.named_steps["model"].booster_
    gain = pd.Series(booster.feature_importance(importance_type="gain"), index=booster.feature_name())
    return (gain / gain.sum()).sort_values(ascending=False)


def permutation_table(pipe: Pipeline, X: pd.DataFrame, y: pd.Series, cfg: dict[str, Any]) -> pd.DataFrame:
    """Drop of PR-AUC on the test set when one RAW column is shuffled (the rest unchanged).
    Columns the Pipeline drops (sex, race, fnlwgt, education) must give exactly 0."""
    ecfg = cfg["explain"]
    r = permutation_importance(pipe, X, y, n_repeats=ecfg["permutation_repeats"], random_state=cfg["seed"],
                               scoring="average_precision")
    return (pd.DataFrame({"pr_auc_drop": r.importances_mean, "std": r.importances_std}, index=X.columns)
            .sort_values("pr_auc_drop", ascending=False))


# ----------------------------------------------------------------------------------------------
# 3. SHAP (Session 8, Part IV)
# ----------------------------------------------------------------------------------------------
def shap_values(pipe: Pipeline, X: pd.DataFrame) -> tuple[np.ndarray, float, pd.DataFrame]:
    """SHAP values of LightGBM on the log-odds scale, one row per person, one column per model input.
    Check: base value + sum of the row = the model's raw score for that person."""
    Z = model_inputs(pipe, X)
    explainer = shap.TreeExplainer(pipe.named_steps["model"])
    values = explainer.shap_values(Z)
    values = values[1] if isinstance(values, list) else values      # positive class only
    base = explainer.expected_value
    base = float(base[1] if np.ndim(base) else base)
    return np.asarray(values), base, Z


def shap_global(values: np.ndarray, columns: list[str]) -> pd.Series:
    """Session 8 global SHAP importance: mean of |SHAP| over people."""
    return pd.Series(np.abs(values).mean(axis=0), index=columns).sort_values(ascending=False)


def shap_by_category(values: np.ndarray, readable: pd.DataFrame, column: str) -> pd.DataFrame:
    """Direction for a categorical feature: mean SHAP value of each category (log-odds)."""
    j = list(readable.columns).index(column)
    df = pd.DataFrame({column: readable[column].values, "shap": values[:, j]})
    return (df.groupby(column)["shap"].agg(["mean", "size"]).rename(columns={"mean": "mean SHAP", "size": "n"})
            .sort_values("mean SHAP", ascending=False))


def local_cases(y: np.ndarray, proba: np.ndarray, threshold: float, weak_mask: np.ndarray) -> dict[str, int]:
    """Three people to explain, picked by a fixed rule (no cherry-picking):
    - a clear call: a wealthy person predicted close to 90%;
    - a borderline call: the person just above the threshold;
    - a missed wealthy person from the weakest slice, at the median probability of those misses."""
    y = np.asarray(y)
    pos = np.arange(len(y))
    clear = pos[y == 1][np.argmin(np.abs(proba[y == 1] - 0.90))]
    above = pos[proba >= threshold]
    border = above[np.argmin(proba[above] - threshold)]
    missed = pos[weak_mask & (y == 1) & (proba < threshold)]
    median_miss = missed[np.argmin(np.abs(proba[missed] - np.median(proba[missed])))]
    return {"clear call (wealthy, ~90%)": int(clear), "borderline call (just above 0.15)": int(border),
            "missed wealthy person (weakest slice)": int(median_miss)}


def local_table(values: np.ndarray, base: float, readable: pd.DataFrame, i: int, top: int = 10) -> pd.DataFrame:
    """One person: value of each input and its SHAP contribution, sorted by |SHAP|."""
    t = pd.DataFrame({"value": readable.iloc[i].astype(str).values, "shap": values[i]}, index=readable.columns)
    return t.reindex(t["shap"].abs().sort_values(ascending=False).index).head(top)


# ----------------------------------------------------------------------------------------------
# 4. Partial dependence (Session 8, Part V)
# ----------------------------------------------------------------------------------------------
def pdp_table(pipe: Pipeline, X_dev: pd.DataFrame, feature: str, cfg: dict[str, Any]) -> pd.DataFrame:
    """Average predicted probability when `feature` is set to each grid value for everyone
    (development sample, as the Session 8 notebook uses the training data)."""
    sample = X_dev.sample(cfg["explain"]["pdp_sample"], random_state=cfg["seed"])
    r = partial_dependence(pipe, sample, [feature], kind="average")
    return pd.DataFrame({feature: r["grid_values"][0], "average predicted probability": r["average"][0]})


# ----------------------------------------------------------------------------------------------
# 5. Run everything
# ----------------------------------------------------------------------------------------------
def run_explain(cfg: dict[str, Any]) -> dict[str, Any]:
    X_dev, X_test, y_dev, y_test = load_splits(cfg)
    pipe = final_pipeline(cfg, X_dev, y_dev)
    threshold = float(pd.read_csv(resolve(cfg["final"]["results_dir"]) / "test_results.csv")
                      .set_index("model").loc[cfg["final"]["model"], "threshold"])
    proba = pipe.predict_proba(X_test)[:, 1]
    y = y_test.values

    masks = slice_masks(X_test, cfg)
    slices = slice_table(y, proba, threshold, masks)
    weak = weakest_slice(slices, cfg["explain"]["min_wealthy_per_slice"])
    errors = error_analysis(X_test, y, proba, threshold, masks[weak])

    gain = gain_importance(pipe)
    perm = permutation_table(pipe, X_test, y_test, cfg)

    values, base, Z = shap_values(pipe, X_test)
    readable = readable_inputs(pipe, X_test)
    raw_score = pipe.named_steps["model"].predict(Z, raw_score=True)
    additivity_error = float(np.abs(base + values.sum(axis=1) - raw_score).max())
    shap_imp = shap_global(values, list(Z.columns))
    cases = local_cases(y, proba, threshold, masks[weak])

    pdps = {f: pdp_table(pipe, X_dev, f, cfg) for f in cfg["explain"]["pdp_features"]}
    return {"pipe": pipe, "threshold": threshold, "proba": proba, "y": y, "X_test": X_test, "masks": masks,
            "slices": slices, "weakest": weak, "errors": errors, "gain": gain, "perm": perm,
            "shap_values": values, "base": base, "Z": Z, "readable": readable, "shap_importance": shap_imp,
            "additivity_error": additivity_error, "cases": cases, "pdps": pdps}


def save_results(cfg: dict[str, Any], r: dict[str, Any]) -> None:
    out = resolve(cfg["explain"]["results_dir"])
    out.mkdir(parents=True, exist_ok=True)
    r["slices"].to_csv(out / "slices_test.csv")
    r["errors"].to_csv(out / "error_analysis_weakest_slice.csv")
    imp = pd.DataFrame({"gain_share": r["gain"], "mean_abs_shap": r["shap_importance"]})
    imp.sort_values("mean_abs_shap", ascending=False).to_csv(out / "importance_gain_shap.csv")
    r["perm"].to_csv(out / "permutation_importance_test.csv")
    for f, t in r["pdps"].items():
        t.to_csv(out / f"pdp_{f}.csv", index=False)


def log_to_mlflow(cfg: dict[str, Any], r: dict[str, Any]) -> None:
    setup_mlflow(cfg)
    with mlflow.start_run(run_name="step8_explainability"):
        mlflow.set_tags({"step": "8", "model": cfg["final"]["model"], "set": "test (description only)",
                         "weakest_slice": r["weakest"]})
        w = r["slices"].loc[r["weakest"]]
        mlflow.log_metrics({"weakest_slice_recall": float(w["recall"]), "weakest_slice_pr_auc": float(w["pr_auc"]),
                            "shap_additivity_error": r["additivity_error"]})
        for feat, v in r["shap_importance"].items():
            mlflow.log_metric(f"mean_abs_shap_{feat.replace('-', '_')}", float(v))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    r = run_explain(cfg)
    save_results(cfg, r)
    log_to_mlflow(cfg, r)
    cols = ["n", "prevalence", "wealthy", "missed", "precision", "recall", "pr_auc"]
    logger.info("Slices (test, threshold %.2f):\n%s", r["threshold"], r["slices"][cols].round(3).to_string())
    logger.info("Weakest slice (recall): %s", r["weakest"])
    logger.info("Mean |SHAP| (log-odds):\n%s", r["shap_importance"].round(3).to_string())
    logger.info("Permutation importance (PR-AUC drop):\n%s", r["perm"].round(4).to_string())
    logger.info("SHAP additivity check, max error: %.1e", r["additivity_error"])


if __name__ == "__main__":
    main()
