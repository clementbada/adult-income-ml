"""STEP 2 - Feature engineering: cleaning, missing values, encoding, engineered features.

Decisions implemented here (see DECISIONS.md, STEP 1 and STEP 2):
- `fnlwgt`, `education`, `sex` and `race` never reach the model (decisions 3, 4, 8);
- `relationship`: `Husband` and `Wife` are merged into `Spouse` (decision 9);
- capital amounts are replaced by `has_capital_gain` / `has_capital_loss` (decision 7);
  the exact amounts stay available as an option, only for the STEP 6 reference model;
- missing categorical values become an explicit "Missing" category (decision 5);
- `age_band` and `hours_band` are added for the linear model (STEP 2).

Two layers, for leakage reasons (Session 5):
1. `AdultFeatureBuilder`: stateless, row-by-row rules. It learns nothing from the data.
2. `ColumnTransformer`: imputation, encoding, scaling. These steps learn from data
   (medians, list of categories, means), so they are fitted on training folds only.

`build_preprocessor(cfg, family)` returns both layers as one Pipeline. The model of
STEP 4 is added after it.

Run from the project root:  python -m src.features   (prints the shape of each variant)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------
# Column groups
# ----------------------------------------------------------------------------------------------
DROP_COLUMNS = ["fnlwgt", "education"]
SENSITIVE_COLUMNS = ["sex", "race"]
CATEGORICAL_COLUMNS = ["workclass", "marital-status", "occupation", "relationship", "native-country"]
NUMERICAL_COLUMNS = ["age", "education-num", "hours-per-week"]
CAPITAL_COLUMNS = ["capital-gain", "capital-loss"]
CAPITAL_FLAGS = ["has_capital_gain", "has_capital_loss"]
CAPITAL_LOGS = ["log_capital_gain", "log_capital_loss"]
RELATIONSHIP_MAP = {"Husband": "Spouse", "Wife": "Spouse"}

FAMILIES = ("linear", "tree")
CAPITAL_MODES = ("indicators", "amounts", "none")
BANDS = ("age_band", "hours_band")
MISSING = "Missing"


class AdultFeatureBuilder(BaseEstimator, TransformerMixin):
    """Stateless cleaning and feature creation on the raw Adult columns.

    Parameters
    ----------
    capital : "indicators" (default, decision 7), "amounts" (STEP 6 reference) or "none".
    bands : which band features to add, among ("age_band", "hours_band").
    age_bins, hours_bins : band edges (right-inclusive), from config.yaml.
    """

    def __init__(
        self,
        capital: str = "indicators",
        bands: tuple[str, ...] = BANDS,
        age_bins: tuple[float, ...] = (0, 25, 35, 45, 55, 65, np.inf),
        age_labels: tuple[str, ...] = ("17-25", "26-35", "36-45", "46-55", "56-65", "66+"),
        hours_bins: tuple[float, ...] = (-np.inf, 34, 40, 49, np.inf),
        hours_labels: tuple[str, ...] = ("<35", "35-40", "41-49", "50+"),
    ):
        self.capital = capital
        self.bands = bands
        self.age_bins = age_bins
        self.age_labels = age_labels
        self.hours_bins = hours_bins
        self.hours_labels = hours_labels

    def fit(self, X: pd.DataFrame, y: Any = None) -> "AdultFeatureBuilder":
        if self.capital not in CAPITAL_MODES:
            raise ValueError(f"capital must be one of {CAPITAL_MODES}, got {self.capital!r}")
        if set(self.bands) - set(BANDS):
            raise ValueError(f"Unknown bands: {set(self.bands) - set(BANDS)}")
        needed = set(CATEGORICAL_COLUMNS + NUMERICAL_COLUMNS + CAPITAL_COLUMNS)
        if needed - set(X.columns):
            raise ValueError(f"Input is missing raw columns: {needed - set(X.columns)}")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.feature_names_out_ = np.asarray(self.transform(X.iloc[:0]).columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.drop(columns=DROP_COLUMNS + SENSITIVE_COLUMNS, errors="ignore").copy()

        # 'category' dtype refuses new values such as "Spouse": switch to plain strings.
        # NaN stays NaN here; the imputer of the ColumnTransformer turns it into "Missing".
        for col in CATEGORICAL_COLUMNS:
            out[col] = out[col].astype(object).where(out[col].notna(), np.nan)
        out["relationship"] = out["relationship"].replace(RELATIONSHIP_MAP)

        gain, loss = out["capital-gain"], out["capital-loss"]
        if self.capital in ("indicators", "amounts"):
            out["has_capital_gain"] = (gain > 0).astype(float).where(gain.notna())
            out["has_capital_loss"] = (loss > 0).astype(float).where(loss.notna())
        if self.capital == "amounts":
            out["log_capital_gain"] = np.log1p(gain)
            out["log_capital_loss"] = np.log1p(loss)
        if self.capital != "amounts":
            out = out.drop(columns=CAPITAL_COLUMNS)   # decision 7: amounts never reach the model

        if "age_band" in self.bands:
            out["age_band"] = pd.cut(out["age"], bins=list(self.age_bins),
                                     labels=list(self.age_labels)).astype(object)
        if "hours_band" in self.bands:
            out["hours_band"] = pd.cut(out["hours-per-week"], bins=list(self.hours_bins),
                                       labels=list(self.hours_labels)).astype(object)
        return out

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return self.feature_names_out_


def model_inputs(family: str, capital: str, bands: tuple[str, ...]) -> tuple[list[str], list[str]]:
    """Numerical and categorical columns given to each model family."""
    if family not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}, got {family!r}")
    num = list(NUMERICAL_COLUMNS)
    if capital in ("indicators", "amounts"):
        num += CAPITAL_FLAGS
    if capital == "amounts":
        # Linear: log1p (a few 99,999 would dominate the coefficient). Tree: raw values
        # (trees are insensitive to monotone transforms).
        num += CAPITAL_LOGS if family == "linear" else CAPITAL_COLUMNS
    cat = list(CATEGORICAL_COLUMNS)
    if family == "linear":
        cat += [b for b in BANDS if b in bands]
    # Trees find their own cut points: the bands would be redundant, they are not given.
    return num, cat


def build_preprocessor(
    cfg: dict[str, Any] | None = None,
    family: str = "linear",
    capital: str | None = None,
    bands: tuple[str, ...] | None = None,
) -> Pipeline:
    """Feature builder + ColumnTransformer, ready to receive a model as last step.

    family="linear": median imputer + StandardScaler on numbers; "Missing" + one-hot on
                     categories (rare categories grouped, unknown categories tolerated).
    family="tree":   median imputer on numbers; "Missing" + ordinal codes on categories.
    `capital` and `bands` override config.yaml (used for the ablation and for STEP 6).
    """
    fcfg = (cfg or {}).get("features", {})
    capital = capital or fcfg.get("capital", "indicators")
    bands = tuple(fcfg.get("bands", BANDS)) if bands is None else tuple(bands)
    min_freq = fcfg.get("rare_min_frequency", 0.005)
    builder_kwargs = {"capital": capital, "bands": bands}
    if "age_bins" in fcfg:
        builder_kwargs.update(age_bins=tuple(float(b) for b in fcfg["age_bins"]),
                              age_labels=tuple(fcfg["age_labels"]))
    if "hours_bins" in fcfg:
        builder_kwargs.update(hours_bins=tuple(float(b) for b in fcfg["hours_bins"]),
                              hours_labels=tuple(fcfg["hours_labels"]))

    num, cat = model_inputs(family, capital, bands)
    if family == "linear":
        num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                             ("scale", StandardScaler())])
        encoder = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=min_freq,
                                sparse_output=False)
    else:
        num_pipe = Pipeline([("impute", SimpleImputer(strategy="median"))])
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                 min_frequency=min_freq)
    cat_pipe = Pipeline([("impute", SimpleImputer(strategy="constant", fill_value=MISSING)),
                         ("encode", encoder)])
    columns = ColumnTransformer([("num", num_pipe, num), ("cat", cat_pipe, cat)],
                                remainder="drop", verbose_feature_names_out=False)
    return Pipeline([("features", AdultFeatureBuilder(**builder_kwargs)), ("encode", columns)])


def categorical_mask(preprocessor: Pipeline) -> list[bool]:
    """For a fitted "tree" preprocessor: True for the ordinal-coded categorical columns.
    Pass it to HistGradientBoostingClassifier(categorical_features=...)."""
    cat_cols = preprocessor.named_steps["encode"].transformers_[1][2]
    return [name in cat_cols for name in preprocessor.get_feature_names_out()]


def main() -> None:
    """Fit every preprocessor variant on the development set and log its output shape."""
    from src.data import load_config, load_splits

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    X_dev, _, _, _ = load_splits(cfg)
    for family in FAMILIES:
        for capital in CAPITAL_MODES:
            Xt = build_preprocessor(cfg, family=family, capital=capital).fit_transform(X_dev)
            logger.info("%-6s capital=%-10s -> %d columns, %d NaN left",
                        family, capital, Xt.shape[1], int(np.isnan(Xt).sum()))


if __name__ == "__main__":
    main()
