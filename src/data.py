"""STEP 1 - Data loading, exact-duplicate removal and stratified development/test split.

Decisions implemented here (see DECISIONS.md, STEP 1):
- the 52 exact duplicate rows (identical on every column, fnlwgt included) are removed BEFORE the
  split, so that the same record cannot end up in both the development and the test set;
- the target is recoded as 1 for ">50K" and 0 otherwise;
- 80% development / 20% test split, stratified on the target, with the seed from config.yaml;
- 5-fold StratifiedKFold inside the development set for every later choice.

All other transformations (dropping fnlwgt and education, setting sex and race aside, recoding
relationship, capital indicators, missing values) are done in STEP 2 inside the Pipeline, so the
saved data keep every raw column.

Run from the project root:  python -m src.data
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from sklearn.datasets import fetch_openml
from sklearn.model_selection import StratifiedKFold, train_test_split

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET_NAME = "target"


def load_config(path: str | Path = PROJECT_ROOT / "config.yaml") -> dict[str, Any]:
    """Read the YAML configuration file and return it as a dictionary."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(path: str | Path) -> Path:
    """Turn a path from config.yaml (relative to the project root) into an absolute path."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_raw(cfg: dict[str, Any]) -> pd.DataFrame:
    """Download (or read from the local cache) the Adult dataset from OpenML.

    Returns the full raw table, target column included. Missing values are already NaN
    in OpenML version 2 (the "?" of the original UCI files are converted).
    """
    data_cfg = cfg["data"]
    bunch = fetch_openml(
        name=data_cfg["openml_name"],
        version=data_cfg["openml_version"],
        as_frame=True,
        data_home=str(resolve(data_cfg["raw_cache_dir"])),
    )
    df = bunch.frame.copy()
    logger.info("Loaded raw data: %d rows, %d columns", *df.shape)
    return df


def remove_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows identical on ALL columns (fnlwgt included): the same record entered twice.

    Rows identical on everything except fnlwgt are kept: they are distinct people with the
    same profile (see notebook 01, section C.4).
    """
    n_before = len(df)
    out = df.drop_duplicates().reset_index(drop=True)
    logger.info("Removed %d exact duplicate rows -> %d rows", n_before - len(out), len(out))
    return out


def split_features_target(
    df: pd.DataFrame, target_column: str, positive_label: str
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the raw features X from the binary target y (1 = positive_label)."""
    y = (df[target_column].astype(str).str.strip() == positive_label).astype(int).rename(TARGET_NAME)
    X = df.drop(columns=[target_column])
    return X, y


def split_dev_test(
    X: pd.DataFrame, y: pd.Series, test_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified development/test split. The test set is used only once, in STEP 6."""
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    logger.info(
        "Split: development %d rows (prevalence %.4f), test %d rows (prevalence %.4f)",
        len(y_dev), y_dev.mean(), len(y_test), y_test.mean(),
    )
    return X_dev, X_test, y_dev, y_test


def get_cv(cfg: dict[str, Any]) -> StratifiedKFold:
    """The cross-validation splitter used by every later step (same folds for everyone)."""
    return StratifiedKFold(n_splits=cfg["split"]["n_folds"], shuffle=True, random_state=cfg["seed"])


def save_splits(
    X_dev: pd.DataFrame, X_test: pd.DataFrame, y_dev: pd.Series, y_test: pd.Series,
    cfg: dict[str, Any],
) -> None:
    """Save the split as parquet files (parquet keeps the 'category' column types)."""
    out_dir = resolve(cfg["data"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    X_dev.assign(**{TARGET_NAME: y_dev}).to_parquet(out_dir / "dev.parquet")
    X_test.assign(**{TARGET_NAME: y_test}).to_parquet(out_dir / "test.parquet")
    logger.info("Saved dev.parquet and test.parquet in %s", out_dir)


def load_splits(
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Read the saved split. Used by every later step instead of re-splitting."""
    out_dir = resolve(cfg["data"]["processed_dir"])
    dev = pd.read_parquet(out_dir / "dev.parquet")
    test = pd.read_parquet(out_dir / "test.parquet")
    return (dev.drop(columns=[TARGET_NAME]), test.drop(columns=[TARGET_NAME]),
            dev[TARGET_NAME], test[TARGET_NAME])


def main() -> None:
    """Full STEP 1 data preparation: load, deduplicate, split, check the folds, save."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    df = remove_exact_duplicates(load_raw(cfg))
    X, y = split_features_target(df, cfg["data"]["target_column"], cfg["data"]["positive_label"])
    X_dev, X_test, y_dev, y_test = split_dev_test(
        X, y, cfg["split"]["test_size"], cfg["seed"]
    )
    for k, (_, va) in enumerate(get_cv(cfg).split(X_dev, y_dev), start=1):
        logger.info("Fold %d: %d validation rows, prevalence %.4f", k, len(va), y_dev.iloc[va].mean())
    save_splits(X_dev, X_test, y_dev, y_test, cfg)


if __name__ == "__main__":
    main()
