"""Tests for src/calibration.py (STEP 7). Synthetic probabilities, no data download."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import brier_score_loss

from src.calibration import (apply_map, bayes_threshold, calibration_metrics, choose_method,
                             cross_validated_calibration, expected_calibration_error, fit_map,
                             fit_platt, platt_parameters, reliability_table)
from src.data import load_config


def _calibrated(n: int = 20_000, seed: int = 0):
    """y drawn with exactly probability p: a perfectly calibrated model (up to noise)."""
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.01, 0.99, n)
    y = (rng.random(n) < p).astype(int)
    return p, y


def test_ece_is_small_for_a_calibrated_model_and_large_for_an_overconfident_one():
    p, y = _calibrated()
    assert expected_calibration_error(y, p) < 0.02
    overconfident = np.clip(0.5 + 1.8 * (p - 0.5), 0.001, 0.999)    # too extreme
    assert expected_calibration_error(y, overconfident) > 0.08


def test_reliability_gap_sign():
    p = np.full(1000, 0.8)
    y = np.r_[np.ones(600), np.zeros(400)].astype(int)               # observed 60% for predicted 80%
    t = reliability_table(y, p)
    assert len(t) == 1 and t["gap"].iloc[0] == pytest.approx(-0.2)   # negative = over-confident
    assert t["n"].sum() == 1000


def test_brier_matches_sklearn():
    p, y = _calibrated(500)
    assert calibration_metrics(y, p)["brier"] == pytest.approx(brier_score_loss(y, p))


def test_platt_on_calibrated_probabilities_is_close_to_identity():
    p, y = _calibrated()
    a, b = platt_parameters(fit_platt(p, y))
    assert a == pytest.approx(1.0, abs=0.1) and b == pytest.approx(0.0, abs=0.1)


def test_platt_corrects_an_overconfident_model():
    p, y = _calibrated()
    s = np.log(p / (1 - p))
    over = 1 / (1 + np.exp(-2.0 * s))                                # scores twice too extreme
    fitted = fit_map("platt", over, y)
    assert platt_parameters(fitted)[0] == pytest.approx(0.5, abs=0.05)   # A < 1: over-confident
    assert expected_calibration_error(y, apply_map("platt", fitted, over)) < expected_calibration_error(y, over)


def test_isotonic_output_is_monotone_and_in_0_1():
    p, y = _calibrated(2000)
    q = apply_map("isotonic", fit_map("isotonic", p, y), np.sort(p))
    assert np.all(np.diff(q) >= 0) and q.min() >= 0 and q.max() <= 1


def test_choose_method_keeps_raw_without_a_regular_gain():
    rows = [{"fold": k, "method": m, "brier": b}
            for k in range(1, 6)
            for m, b in [("raw", 0.10), ("platt", 0.10 + (0.001 if k % 2 else -0.001)), ("isotonic", 0.101)]]
    assert choose_method(pd.DataFrame(rows)) == "raw"
    rows = [{"fold": k, "method": m, "brier": b}
            for k in range(1, 6) for m, b in [("raw", 0.10), ("platt", 0.09), ("isotonic", 0.095)]]
    assert choose_method(pd.DataFrame(rows)) == "platt"


def test_cross_validated_calibration_has_one_row_per_fold_and_method():
    p, y = _calibrated(1000)
    idx = np.arange(1000)
    folds = [(np.setdiff1d(idx, idx[k::5]), idx[k::5]) for k in range(5)]
    t = cross_validated_calibration(pd.Series(y), p, folds)
    assert len(t) == 15 and set(t["method"]) == {"raw", "platt", "isotonic"}


def test_bayes_threshold_is_one_sixth_for_x_equal_5():
    assert bayes_threshold(load_config()) == pytest.approx(1 / 6)
