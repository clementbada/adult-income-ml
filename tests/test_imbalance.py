"""Tests for the STEP 4 additions (business cost, threshold choice, class weights, SMOTE)."""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.data import load_config
from src.evaluate import best_threshold, cross_validate_oof, threshold_table
from src.imbalance import sensitivity_table
from src.models import build_logistic
from test_models import _raw_frame   # shared synthetic raw rows (tests/test_models.py)


def test_threshold_table_counts_and_cost():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.3, 0.6, 0.1])
    row = threshold_table(y, p, cost_fn=5, cost_fp=1, thresholds=np.array([0.5])).iloc[0]
    assert (row["tp"], row["fn"], row["fp"], row["calls"]) == (1, 1, 1, 2)
    assert row["cost"] == 5 * 1 + 1 * 1


def test_higher_cost_of_missing_gives_lower_threshold():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    p = np.clip(0.5 * y + rng.normal(0.25, 0.2, 2000), 0, 1)
    assert best_threshold(y, p, cost_fn=10) <= best_threshold(y, p, cost_fn=2)


def test_sensitivity_table_is_monotone_in_x():
    rng = np.random.default_rng(1)
    y = pd.Series(rng.integers(0, 2, 2000))
    p = np.clip(0.5 * y.values + rng.normal(0.25, 0.2, 2000), 0, 1)
    table = sensitivity_table(y, p, [2, 3, 5, 10])
    assert table["calls"].is_monotonic_increasing and table["missed"].is_monotonic_decreasing


def test_class_weight_option_reaches_the_model():
    cfg = load_config()
    pipe = build_logistic(cfg, class_weight="balanced")
    assert pipe.named_steps["model"].class_weight == "balanced"


def test_smote_runs_on_training_folds_only():
    cfg = load_config()
    X, y = _raw_frame()
    pipe = build_logistic(cfg, smote=True)
    assert "smote" in pipe.named_steps
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    per_fold, oof = cross_validate_oof(pipe, X, y, cv)
    # one prediction per real row: no synthetic row reaches the validation folds
    assert oof.shape == (len(y),) and not np.isnan(oof).any()


def test_cross_validated_threshold_never_uses_the_measured_fold():
    rng = np.random.default_rng(2)
    y = pd.Series(rng.integers(0, 2, 1500))
    p = np.clip(0.5 * y.values + rng.normal(0.25, 0.2, 1500), 0, 1)
    X = pd.DataFrame({"a": rng.random(1500)})
    from src.imbalance import cross_validated_threshold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    table = cross_validated_threshold(y, p, cv, X, cost_fn=5)
    assert len(table) == 5
    for k, (_, va) in enumerate(cv.split(X, y), start=1):
        others = np.setdiff1d(np.arange(len(y)), va)
        assert table.loc[k, "threshold_chosen_without_fold"] == best_threshold(y.iloc[others], p[others], 5)
