"""Tests for src/evaluate.py, src/models.py and src/tracking.py (small synthetic data, no internet)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from src.data import load_config
from src.evaluate import compute_metrics, cross_validate_oof, summarize
from src.models import build_logistic


def _raw_frame(n: int = 600, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """Raw rows with the 14 Adult columns; the target depends on age and education-num."""
    rng = np.random.default_rng(seed)
    cat = lambda values: pd.Categorical(rng.choice(values, n))
    X = pd.DataFrame({
        "age": rng.integers(17, 91, n), "workclass": cat(["Private", "Self-emp-inc", "State-gov"]),
        "fnlwgt": rng.integers(10_000, 1_000_000, n), "education": cat(["Bachelors", "HS-grad"]),
        "education-num": rng.integers(1, 17, n), "marital-status": cat(["Married-civ-spouse", "Never-married"]),
        "occupation": cat(["Sales", "Exec-managerial", "Craft-repair"]),
        "relationship": cat(["Husband", "Wife", "Own-child"]), "race": cat(["White", "Black"]),
        "sex": cat(["Male", "Female"]), "capital-gain": rng.choice([0, 0, 0, 5000], n),
        "capital-loss": rng.choice([0, 0, 0, 1900], n), "hours-per-week": rng.integers(1, 99, n),
        "native-country": cat(["United-States", "Mexico"]),
    })
    logit = 0.05 * (X["age"] - 40) + 0.4 * (X["education-num"] - 10)
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int))
    return X, y


def test_metrics_perfect_ranking():
    y = np.array([0, 0, 1, 1])
    m = compute_metrics(y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert m["pr_auc"] == pytest.approx(1.0) and m["roc_auc"] == pytest.approx(1.0)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 0, 0, 2)
    assert m["prevalence"] == 0.5


def test_metrics_threshold_changes_precision_recall_not_auc():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.4, 0.6, 0.3, 0.7, 0.9])
    low, high = compute_metrics(y, p, threshold=0.2), compute_metrics(y, p, threshold=0.65)
    assert low["recall"] >= high["recall"]
    assert low["pr_auc"] == high["pr_auc"] and low["roc_auc"] == high["roc_auc"]


def test_logistic_pipeline_runs_on_raw_data_and_ignores_sensitive_columns():
    cfg = load_config()
    X, y = _raw_frame()
    pipe = build_logistic(cfg).fit(X, y)
    proba = pipe.predict_proba(X)[:, 1]
    assert proba.shape == (len(X),) and ((proba >= 0) & (proba <= 1)).all()
    names = pipe.named_steps["preprocess"].get_feature_names_out()
    assert not any(n.startswith(("sex", "race", "fnlwgt", "education_")) for n in names)


def test_cross_validation_gives_one_oof_prediction_per_row():
    cfg = load_config()
    X, y = _raw_frame()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    per_fold, oof = cross_validate_oof(build_logistic(cfg), X, y, cv)
    assert len(per_fold) == 5
    assert not np.isnan(oof).any() and oof.shape == (len(y),)
    assert set(summarize(per_fold).columns) == {"mean", "std"}


def test_mlflow_run_is_logged(tmp_path):
    import mlflow
    from src.tracking import log_cv_run

    mlflow.set_tracking_uri(tmp_path.as_uri())
    mlflow.set_experiment("test")
    per_fold = pd.DataFrame({m: [0.5, 0.6] for m in
                             ["pr_auc", "roc_auc", "log_loss", "precision", "recall", "f1"]},
                            index=pd.Index([1, 2], name="fold"))
    run_id = log_cv_run("unit_test", {"C": 1.0}, per_fold, tags={"step": "test"})
    run = mlflow.get_run(run_id)
    assert run.data.params["C"] == "1.0"
    assert run.data.metrics["cv_pr_auc_mean"] == pytest.approx(0.55)


# ---------------------------------------------------------------------------------------------
# STEP 5: Random Forest and LightGBM
# ---------------------------------------------------------------------------------------------
from src.models import build_lightgbm, build_model, build_random_forest  # noqa: E402


@pytest.mark.parametrize("builder,options", [
    (build_random_forest, {"encoding": "onehot"}),
    (build_random_forest, {"encoding": "ordinal"}),
    (build_lightgbm, {}),
])
def test_tree_pipelines_run_on_raw_data(builder, options):
    cfg = load_config()
    X, y = _raw_frame()
    pipe = builder(cfg, n_estimators=20, **options).fit(X, y)
    proba = pipe.predict_proba(X)[:, 1]
    assert proba.shape == (len(X),) and ((proba >= 0) & (proba <= 1)).all()


def test_tree_pipelines_tolerate_an_unknown_category():
    cfg = load_config()
    X, y = _raw_frame()
    new = X.iloc[[0, 1]].copy()
    new["native-country"] = pd.Categorical(["Atlantis", "Atlantis"])
    for name in ["random_forest", "lightgbm"]:
        pipe = build_model(name, cfg, n_estimators=20).fit(X, y)
        assert pipe.predict_proba(new).shape == (2, 2)


def test_lightgbm_receives_categorical_columns_and_no_sensitive_column():
    cfg = load_config()
    X, y = _raw_frame()
    pipe = build_lightgbm(cfg, n_estimators=20).fit(X, y)
    Xt = X
    for _, step in pipe.steps[:-1]:          # everything before the model (already fitted)
        Xt = step.transform(Xt)
    assert not any(c in Xt.columns for c in ["sex", "race", "fnlwgt", "education"])
    cat_cols = [c for c in Xt.columns if str(Xt[c].dtype) == "category"]
    assert cat_cols == ["workclass", "marital-status", "occupation", "relationship", "native-country"]
    # LightGBM really treats them as categorical: its trees contain categorical splits ("==")
    dump = pipe.named_steps["model"].booster_.dump_model()
    split_types = set()

    def walk(node):
        if "split_feature" in node:
            split_types.add((dump["feature_names"][node["split_feature"]], node["decision_type"]))
            walk(node["left_child"])
            walk(node["right_child"])

    for tree in dump["tree_info"]:
        walk(tree["tree_structure"])
    categorical_splits = {name for name, kind in split_types if kind == "=="}
    assert categorical_splits and categorical_splits <= set(cat_cols)


def test_lightgbm_one_row_gives_the_same_prediction_as_the_batch():
    """Pandas categories are rebuilt for every input: the codes must keep the same meaning."""
    cfg = load_config()
    X, y = _raw_frame()
    pipe = build_lightgbm(cfg, n_estimators=20).fit(X, y)
    batch = pipe.predict_proba(X.iloc[:30])[:, 1]
    one_by_one = np.array([pipe.predict_proba(X.iloc[[i]])[0, 1] for i in range(30)])
    assert np.allclose(batch, one_by_one)


def test_build_model_rejects_unknown_names():
    with pytest.raises(ValueError):
        build_model("svm", load_config())
