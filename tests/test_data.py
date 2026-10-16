"""Tests for src/data.py. They use a small synthetic table, so they run without internet."""

import numpy as np
import pandas as pd

from src.data import (
    get_cv,
    load_config,
    remove_exact_duplicates,
    split_dev_test,
    split_features_target,
)


def _toy_frame(n: int = 1000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": rng.integers(17, 91, n),
        "fnlwgt": rng.integers(10_000, 1_000_000, n),
        "sex": pd.Categorical(rng.choice(["Male", "Female"], n)),
        "class": pd.Categorical(rng.choice(["<=50K", ">50K"], n, p=[0.76, 0.24])),
    })
    return pd.concat([df, df.iloc[:5]], ignore_index=True)  # add 5 exact duplicates


def test_config_has_required_keys():
    cfg = load_config()
    assert cfg["seed"] == 42
    assert cfg["split"]["test_size"] == 0.20
    assert cfg["split"]["n_folds"] == 5


def test_exact_duplicates_are_removed():
    df = _toy_frame()
    out = remove_exact_duplicates(df)
    assert len(out) == len(df) - 5
    assert not out.duplicated().any()


def test_target_is_binary_and_removed_from_features():
    X, y = split_features_target(_toy_frame(), "class", ">50K")
    assert set(y.unique()) <= {0, 1}
    assert "class" not in X.columns


def test_split_is_stratified_and_disjoint():
    X, y = split_features_target(remove_exact_duplicates(_toy_frame(5000)), "class", ">50K")
    X_dev, X_test, y_dev, y_test = split_dev_test(X, y, test_size=0.2, seed=42)
    assert len(X_test) == round(0.2 * len(X))
    assert abs(y_dev.mean() - y_test.mean()) < 0.01            # same prevalence
    assert set(X_dev.index).isdisjoint(set(X_test.index))      # no row in both sets


def test_split_is_reproducible():
    X, y = split_features_target(_toy_frame(), "class", ">50K")
    a = split_dev_test(X, y, test_size=0.2, seed=42)[1].index
    b = split_dev_test(X, y, test_size=0.2, seed=42)[1].index
    assert list(a) == list(b)


def test_cv_folds_keep_prevalence():
    cfg = load_config()
    X, y = split_features_target(_toy_frame(5000), "class", ">50K")
    for _, va in get_cv(cfg).split(X, y):
        assert abs(y.iloc[va].mean() - y.mean()) < 0.01
