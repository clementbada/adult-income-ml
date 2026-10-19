"""Tests for src/fairness.py (STEP 9). Synthetic data, no download, no test-set access."""

import numpy as np
import pandas as pd
import pytest

from src.data import load_config
from src.fairness import (bootstrap_tpr_gap, business_cost, choose_lambda, fit_weighted, gap_summary,
                          group_label_weights, group_metrics, recall_within_strata)
from tests.test_models import _raw_frame


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def _session9_example():
    """Worked example of the Session 9 slides: TPR 0.70 and FPR 0.10 in both groups,
    base rates 18% (12,000 people) and 25% (18,000 people)."""
    rows = []
    for group, n, base in [("A", 12_000, 0.18), ("B", 18_000, 0.25)]:
        pos, neg = int(n * base), n - int(n * base)
        tp, fp = int(0.70 * pos), int(0.10 * neg)
        rows += [(group, 1, 1)] * tp + [(group, 1, 0)] * (pos - tp) + [(group, 0, 1)] * fp + [(group, 0, 0)] * (neg - fp)
    df = pd.DataFrame(rows, columns=["group", "y", "decision"])
    return df["y"].values, df["decision"].values, df["group"].values


def test_metrics_match_the_session9_worked_example():
    y, d, g = _session9_example()
    by = group_metrics(y, d, g).by_group
    assert by.loc["A", "selection_rate"] == pytest.approx(0.208, abs=1e-3)
    assert by.loc["B", "selection_rate"] == pytest.approx(0.250, abs=1e-3)
    assert by.loc["A", "ppv"] == pytest.approx(0.606, abs=1e-3)
    assert by.loc["B", "ppv"] == pytest.approx(0.700, abs=1e-3)
    gaps = gap_summary(group_metrics(y, d, g))
    assert gaps["equalized_odds_gap"] == pytest.approx(0.0, abs=1e-9)        # equalized odds holds...
    assert gaps["selection_rate_ratio"] == pytest.approx(0.832, abs=1e-3)    # ...four-fifths rule passes...
    assert gaps["ppv_gap"] == pytest.approx(0.094, abs=1e-3)                 # ...predictive parity fails


def test_weights_only_change_the_chosen_label_group_cell():
    y = np.array([1, 1, 0, 0])
    sex = np.array(["Female", "Male", "Female", "Male"])
    assert (group_label_weights(y, sex, 1.0, "Female") == 1).all()           # lam = 1: STEP 6 model
    assert list(group_label_weights(y, sex, 6.0, "Female")) == [6.0, 1.0, 1.0, 1.0]


def test_choose_lambda_takes_the_smallest_weight_reaching_the_target():
    curve = pd.DataFrame({"tpr_gap": [0.10, 0.06, 0.025, 0.02]}, index=pd.Index([1, 2, 4, 6], name="lam"))
    assert choose_lambda(curve, 0.03) == 4
    assert choose_lambda(curve, 0.001) == 6                                   # unreachable -> smallest gap


def test_recall_within_strata_separates_composition_from_treatment():
    """Same recall for both sexes inside each marital group, different overall recall because
    the groups are mixed differently: the strata table must show no gap inside the groups."""
    marital = pd.Series(["Married-civ-spouse"] * 10 + ["Never-married"] * 10)
    sex = np.array(["Male"] * 8 + ["Female"] * 2 + ["Male"] * 2 + ["Female"] * 8)
    y = np.ones(20, dtype=int)
    d = np.array([1] * 10 + [1, 0] * 5)                                       # married: all found; not married: half
    t = recall_within_strata(y, d, sex, marital)
    assert t.loc[("married", "Male"), "tpr"] == t.loc[("married", "Female"), "tpr"] == 1.0
    overall = group_metrics(y, d, sex).by_group["tpr"]
    assert overall["Male"] > overall["Female"]


def test_bootstrap_interval_contains_the_observed_gap():
    rng = np.random.default_rng(0)
    sex = rng.choice(["Male", "Female"], 4000)
    y = rng.integers(0, 2, 4000)
    d = np.where(sex == "Male", rng.random(4000) < 0.9, rng.random(4000) < 0.8).astype(int)
    low, high = bootstrap_tpr_gap(y, d, sex, n_boot=300, seed=42)
    observed = d[(sex == "Male") & (y == 1)].mean() - d[(sex == "Female") & (y == 1)].mean()
    assert low < observed < high


def test_business_cost_matches_step4(cfg):
    y = np.array([1, 1, 0, 0])
    d = np.array([0, 1, 1, 0])                                                # one miss, one useless call
    assert business_cost(y, d, cfg) == cfg["imbalance"]["cost_fn"] + cfg["imbalance"]["cost_fp"]


def test_weights_never_make_sex_a_model_input(cfg):
    X, y = _raw_frame()
    w = group_label_weights(y.values, X["sex"].astype(str).values, 6.0, "Female")
    pipe = fit_weighted(cfg, {"n_estimators": 20, "num_leaves": 8, "min_child_samples": 5}, X, y, w)
    Z = X
    for _, step in pipe.steps[:-1]:
        Z = step.transform(Z)
    assert "sex" not in Z.columns and "race" not in Z.columns
    X_swapped = X.assign(sex=np.where(X["sex"].astype(str) == "Male", "Female", "Male"))
    assert np.allclose(pipe.predict_proba(X)[:, 1], pipe.predict_proba(X_swapped)[:, 1])  # same answer
