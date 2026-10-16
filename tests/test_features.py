"""Tests for src/features.py. Small synthetic table with the raw Adult columns: no internet."""

import numpy as np
import pandas as pd
import pytest

from src.data import load_config
from src.features import AdultFeatureBuilder, build_preprocessor, categorical_mask


def _raw_frame(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cat = lambda values, p=None: pd.Categorical(rng.choice(values, n, p=p))  # noqa: E731
    df = pd.DataFrame({
        "age": rng.integers(17, 91, n),
        "workclass": cat(["Private", "Self-emp-inc", "State-gov"]),
        "fnlwgt": rng.integers(10_000, 1_000_000, n),
        "education": cat(["Bachelors", "HS-grad"]),
        "education-num": rng.integers(1, 17, n),
        "marital-status": cat(["Married-civ-spouse", "Never-married"]),
        "occupation": cat(["Sales", "Tech-support", "Exec-managerial"]),
        "relationship": cat(["Husband", "Wife", "Own-child", "Unmarried"]),
        "race": cat(["White", "Black"]),
        "sex": cat(["Male", "Female"]),
        "capital-gain": rng.choice([0, 0, 0, 5000, 99999], n),
        "capital-loss": rng.choice([0, 0, 0, 1887], n),
        "hours-per-week": rng.integers(1, 100, n),
        "native-country": cat(["United-States", "Mexico", "Laos"], p=[0.9, 0.098, 0.002]),
    })
    df.loc[:9, ["workclass", "occupation"]] = np.nan   # informative missing values
    return df


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def test_builder_drops_sensitive_and_redundant_columns():
    out = AdultFeatureBuilder().fit(_raw_frame()).transform(_raw_frame())
    for col in ["sex", "race", "fnlwgt", "education"]:
        assert col not in out.columns


def test_relationship_recode():
    out = AdultFeatureBuilder().fit(_raw_frame()).transform(_raw_frame())
    assert not out["relationship"].isin(["Husband", "Wife"]).any()
    assert "Spouse" in set(out["relationship"])


@pytest.mark.parametrize("capital, present, absent", [
    ("indicators", ["has_capital_gain", "has_capital_loss"], ["capital-gain", "log_capital_gain"]),
    ("amounts", ["has_capital_gain", "capital-gain", "log_capital_gain"], []),
    ("none", [], ["has_capital_gain", "capital-gain"]),
])
def test_capital_modes(capital, present, absent):
    out = AdultFeatureBuilder(capital=capital).fit(_raw_frame()).transform(_raw_frame())
    assert set(present) <= set(out.columns)
    assert not set(absent) & set(out.columns)


def test_capital_flags_values():
    df = _raw_frame()
    out = AdultFeatureBuilder().fit(df).transform(df)
    assert (out["has_capital_gain"] == (df["capital-gain"] > 0)).all()
    assert out.loc[df["capital-gain"] == 99999, "has_capital_gain"].eq(1).all()


def test_bands_cover_every_row():
    out = AdultFeatureBuilder().fit(_raw_frame()).transform(_raw_frame())
    assert out["age_band"].notna().all() and out["hours_band"].notna().all()


@pytest.mark.parametrize("family", ["linear", "tree"])
def test_preprocessor_output_has_no_nan_and_tolerates_unknown_category(cfg, family):
    df = _raw_frame()
    pre = build_preprocessor(cfg, family=family).fit(df)
    assert not np.isnan(pre.transform(df)).any()
    new = df.iloc[[0]].copy()
    new["native-country"] = new["native-country"].astype(object)
    new.loc[:, "native-country"] = "Atlantis"
    assert pre.transform(new).shape[1] == pre.transform(df).shape[1]


def test_missing_becomes_its_own_column(cfg):
    pre = build_preprocessor(cfg, family="linear").fit(_raw_frame())
    names = list(pre.get_feature_names_out())
    assert "workclass_Missing" in names and "occupation_Missing" in names


def test_rare_categories_are_grouped(cfg):
    pre = build_preprocessor(cfg, family="linear").fit(_raw_frame())
    names = list(pre.get_feature_names_out())
    assert "native-country_Laos" not in names          # 0.2% < 0.5% threshold


def test_tree_preprocessor_has_no_bands_and_a_categorical_mask(cfg):
    pre = build_preprocessor(cfg, family="tree").fit(_raw_frame())
    names = list(pre.get_feature_names_out())
    assert "age_band" not in names and "hours_band" not in names
    assert sum(categorical_mask(pre)) == 5


def test_preprocessor_is_fitted_on_training_rows_only(cfg):
    """The scaler must use the rows it is fitted on: different rows -> different means."""
    df = _raw_frame(800)
    a = build_preprocessor(cfg).fit(df.iloc[:400]).named_steps["encode"]
    b = build_preprocessor(cfg).fit(df.iloc[400:]).named_steps["encode"]
    mean_a = a.named_transformers_["num"].named_steps["scale"].mean_
    mean_b = b.named_transformers_["num"].named_steps["scale"].mean_
    assert not np.allclose(mean_a, mean_b)
