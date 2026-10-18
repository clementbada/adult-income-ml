"""Tests for src/tuning.py (STEP 5): Optuna search spaces, trials, MLflow logging, comparison tables.
Small synthetic data and very small budgets, so the tests run in seconds and without internet."""

import copy

import mlflow
import numpy as np
import optuna
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from src.data import load_config
from src.tuning import (
    cross_validate_trial,
    make_study,
    nested_cv,
    paired_differences,
    suggest_params,
    trials_table,
    tune_model,
)
from src.models import build_model
from test_models import _raw_frame   # shared synthetic raw rows (tests/test_models.py)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _small_cfg() -> dict:
    """Project config with small search spaces, so that each trial is fast on 600 rows."""
    cfg = copy.deepcopy(load_config())
    cfg["tuning"]["n_startup_trials"] = 2
    spaces = cfg["tuning"]["search_spaces"]
    spaces["random_forest"]["n_estimators"] = {"type": "int", "low": 10, "high": 30, "step": 10}
    spaces["lightgbm"]["n_estimators"] = {"type": "int", "low": 10, "high": 30, "step": 10}
    return cfg


CV = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)


def test_suggested_values_stay_inside_the_search_space():
    cfg = _small_cfg()
    for model, space in cfg["tuning"]["search_spaces"].items():
        study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
        for _ in range(20):
            params = suggest_params(study.ask(), space)
            for name, spec in space.items():
                if spec["type"] == "categorical":
                    assert params[name] in spec["choices"]
                else:
                    assert spec["low"] <= params[name] <= spec["high"]
            build_model(model, cfg, **params)        # every proposal gives a valid Pipeline


def test_unknown_type_in_search_space_is_rejected():
    study = optuna.create_study()
    with pytest.raises(ValueError):
        suggest_params(study.ask(), {"C": {"type": "decimal", "low": 0, "high": 1}})


@pytest.mark.parametrize("model", ["logistic", "random_forest", "lightgbm"])
def test_tuning_returns_a_best_trial_for_each_model(model):
    cfg = _small_cfg()
    X, y = _raw_frame()
    study = tune_model(model, cfg, X, y, CV, n_trials=3, log_to_mlflow=False)
    assert 0 <= study.best_value <= 1
    assert set(study.best_params) == set(cfg["tuning"]["search_spaces"][model])
    table = trials_table(study)
    assert len(table) == 3 and "pr_auc" in table.columns


def test_same_seed_gives_the_same_trials():
    cfg = _small_cfg()
    X, y = _raw_frame()
    a = tune_model("logistic", cfg, X, y, CV, n_trials=3, log_to_mlflow=False)
    b = tune_model("logistic", cfg, X, y, CV, n_trials=3, log_to_mlflow=False)
    assert [t.params for t in a.trials] == [t.params for t in b.trials]


def test_pruned_trial_keeps_the_folds_already_done():
    """A pruner that stops every trial after the first fold: the exception carries that fold."""
    X, y = _raw_frame()
    study = optuna.create_study(direction="maximize",
                                pruner=optuna.pruners.ThresholdPruner(lower=1.1))   # always prunes
    trial = study.ask()
    with pytest.raises(optuna.TrialPruned) as info:
        cross_validate_trial(trial, build_model("logistic", load_config()), X, y, CV)
    assert len(info.value.per_fold) == 1


def test_every_trial_is_logged_in_mlflow_under_one_parent_run(tmp_path):
    cfg = _small_cfg()
    X, y = _raw_frame()
    mlflow.set_tracking_uri(tmp_path.as_uri())
    mlflow.set_experiment("test_tuning")
    tune_model("logistic", cfg, X, y, CV, n_trials=3, log_to_mlflow=True)
    runs = mlflow.search_runs(experiment_names=["test_tuning"])
    parents = runs[runs["tags.mlflow.runName"] == "step5_tune_logistic"]
    children = runs[runs["tags.mlflow.parentRunId"] == parents["run_id"].iloc[0]]
    assert len(parents) == 1 and len(children) == 3
    assert children["params.C"].notna().all() and children["metrics.cv_pr_auc_mean"].notna().all()
    assert parents["metrics.best_cv_pr_auc"].iloc[0] == pytest.approx(children["metrics.cv_pr_auc_mean"].max())


def test_paired_differences_are_computed_fold_by_fold():
    fold = pd.Index([1, 2, 3], name="fold")
    results = {
        "logistic": {"per_fold": pd.DataFrame({"pr_auc": [0.70, 0.72, 0.74]}, index=fold),
                     "fold_cost": np.array([100.0, 110.0, 120.0])},
        "lightgbm": {"per_fold": pd.DataFrame({"pr_auc": [0.71, 0.74, 0.73]}, index=fold),
                     "fold_cost": np.array([95.0, 100.0, 125.0])},
    }
    row = paired_differences(results).loc["lightgbm"]
    assert row["pr_auc_diff_mean"] == pytest.approx((0.01 + 0.02 - 0.01) / 3)
    assert row["folds_better_pr_auc"] == "2/3" and row["folds_lower_cost"] == "2/3"
    assert row["cost_diff_mean"] == pytest.approx((-5 - 10 + 5) / 3)


def test_nested_cv_scores_every_outer_fold():
    cfg = _small_cfg()
    cfg["tuning"]["nested"]["inner_folds"] = 2
    X, y = _raw_frame()
    table = nested_cv("logistic", cfg, X, y, CV, n_trials=2)
    assert list(table.index) == [1, 2, 3]
    assert table["outer_pr_auc"].between(0, 1).all()


def test_study_maximises():
    assert make_study(_small_cfg(), "s").direction == optuna.study.StudyDirection.MAXIMIZE
