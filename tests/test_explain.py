"""Tests for src/explain.py (STEP 8). Small synthetic data from tests/test_models.py, no download."""

import numpy as np
import pandas as pd
import pytest

from src.data import load_config
from src.explain import (gain_importance, local_cases, model_inputs, readable_inputs, shap_global,
                         shap_values, slice_masks, slice_table, weakest_slice)
from src.models import build_model
from tests.test_models import _raw_frame


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def fitted(cfg):
    X, y = _raw_frame()
    pipe = build_model("lightgbm", cfg, n_estimators=30, num_leaves=8, min_child_samples=5).fit(X, y)
    return pipe, X, y


def test_shap_values_add_up_to_the_raw_score(fitted):
    """Session 8: base value + sum of the SHAP values = model output on the explained scale (log-odds)."""
    pipe, X, _ = fitted
    values, base, Z = shap_values(pipe, X)
    raw = pipe.named_steps["model"].predict(Z, raw_score=True)
    assert values.shape == Z.shape
    assert np.allclose(base + values.sum(axis=1), raw, atol=1e-6)


def test_sensitive_columns_never_reach_the_model(fitted):
    pipe, X, _ = fitted
    cols = set(model_inputs(pipe, X).columns)
    assert not cols & {"sex", "race", "fnlwgt", "education", "capital-gain", "capital-loss"}


def test_readable_inputs_show_category_names(fitted):
    pipe, X, _ = fitted
    r = readable_inputs(pipe, X)
    assert "Spouse" in set(r["relationship"])                 # names, not ordinal codes
    assert list(r.columns) == list(model_inputs(pipe, X).columns)


def test_gain_importance_sums_to_one(fitted):
    pipe, _, _ = fitted
    assert gain_importance(pipe).sum() == pytest.approx(1.0)


def test_shap_global_is_mean_absolute_value():
    values = np.array([[1.0, -2.0], [-3.0, 0.0]])
    imp = shap_global(values, ["a", "b"])
    assert imp["a"] == pytest.approx(2.0) and imp["b"] == pytest.approx(1.0)


def test_slice_table_counts(cfg):
    X, _ = _raw_frame()
    y = np.array([1, 0] * (len(X) // 2))
    proba = np.where(y == 1, 0.6, 0.1)
    proba[0] = 0.05                                          # one wealthy person missed
    t = slice_table(y, proba, 0.15, slice_masks(X, cfg))
    assert t.loc["All", "missed"] == 1 and t.loc["All", "found"] == y.sum() - 1
    assert t.loc["married", "n"] + t.loc["not married", "n"] == len(X)
    assert t.loc["sex: Male", "n"] + t.loc["sex: Female", "n"] == len(X)


def test_weakest_slice_ignores_tiny_slices():
    t = pd.DataFrame({"wealthy": [500, 20, 300], "recall": [0.9, 0.1, 0.6]}, index=["All", "tiny", "big"])
    assert weakest_slice(t, min_wealthy=100) == "big"


def test_local_cases_follow_the_rule():
    y = np.array([1, 1, 0, 1, 0, 1])
    proba = np.array([0.88, 0.40, 0.16, 0.05, 0.02, 0.10])
    weak = np.array([False, False, False, True, False, True])
    cases = local_cases(y, proba, 0.15, weak)
    assert cases["clear call (wealthy, ~90%)"] == 0
    assert cases["borderline call (just above 0.15)"] == 2
    assert proba[cases["missed wealthy person (weakest slice)"]] < 0.15
