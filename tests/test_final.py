"""Tests for src/final.py (STEP 6). Small synthetic data, no internet, no test-set access."""

import numpy as np
import pandas as pd
import pytest

from src.data import load_config
from src.final import business_cost, fit_on_development, predict_decision, measure_on_test, with_capital
from tests.test_models import _raw_frame


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def test_final_settings_are_frozen_in_config(cfg):
    """The final model is written in config.yaml before the test set is used (decision 39)."""
    assert cfg["final"]["model"] == "lightgbm"
    assert cfg["final"]["reference"] == "logistic"


def test_with_capital_does_not_change_the_original_config(cfg):
    other = with_capital(cfg, "amounts")
    assert other["features"]["capital"] == "amounts"
    assert cfg["features"]["capital"] == "indicators"


def test_business_cost_matches_step4_rule(cfg):
    m = {"fn": 10, "fp": 30}
    assert business_cost(m, cfg) == cfg["imbalance"]["cost_fn"] * 10 + cfg["imbalance"]["cost_fp"] * 30


def test_predict_decision_uses_the_threshold_not_0_5(cfg):
    """A person at P = 0.30 is called with the threshold 0.15, not with the default 0.5."""
    X, y = _raw_frame()
    pipe = fit_on_development(cfg, "logistic", {"C": 1.0}, X, y)
    proba = pipe.predict_proba(X)[:, 1]
    low = predict_decision(pipe, X, 0.15)
    assert (low == (proba >= 0.15)).all()
    assert low.sum() >= pipe.predict(X).sum()          # lower threshold -> at least as many calls


def test_measure_on_test_counts_are_consistent(cfg):
    y = pd.Series([0, 0, 1, 1, 1])
    proba = np.array([0.1, 0.3, 0.2, 0.6, 0.9])
    row = measure_on_test("demo", y, proba, 0.25, cfg)
    assert row["calls"] == row["tp"] + row["fp"] == 3          # 0.3, 0.6 and 0.9 are >= 0.25
    assert row["tp"] + row["fn"] == 3                   # every >50K person is either found or missed
    assert row["cost"] == cfg["imbalance"]["cost_fn"] * row["fn"] + cfg["imbalance"]["cost_fp"] * row["fp"]
