"""STEP 3 - Evaluation module: the same metrics, computed the same way, for every model.

Metrics (Sessions 1, 6):
- threshold-free (ranking quality over all thresholds): PR-AUC (= average precision), with the
  prevalence as random baseline, and ROC-AUC;
- probability quality: log-loss (the loss that trains logistic regression, Session 1);
- threshold-dependent (one operating point): precision, recall, F1 and the confusion matrix.
  In STEP 3 they are reported at the default threshold 0.5 only as a reference; the operating
  threshold is chosen in STEP 4 on out-of-fold predictions (Session 6).

`cross_validate_oof` runs the 5-fold StratifiedKFold of STEP 1 by hand, so that we keep, for every
development row, the probability predicted by the model that did NOT see that row in training
(out-of-fold predictions). STEP 4 (threshold) and STEP 7 (calibration) need them.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import BaseCrossValidator

logger = logging.getLogger(__name__)

METRIC_NAMES = ["pr_auc", "roc_auc", "log_loss", "precision", "recall", "f1"]


def compute_metrics(y_true: np.ndarray | pd.Series, proba: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """All metrics for one set of predictions. `proba` = predicted P(Y=1)."""
    y_true = np.asarray(y_true)
    y_pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "pr_auc": average_precision_score(y_true, proba),
        "roc_auc": roc_auc_score(y_true, proba),
        "log_loss": log_loss(y_true, proba, labels=[0, 1]),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "prevalence": float(y_true.mean()),       # PR-AUC of a random ranking
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def cross_validate_oof(
    model: Any, X: pd.DataFrame, y: pd.Series, cv: BaseCrossValidator, threshold: float = 0.5,
) -> tuple[pd.DataFrame, np.ndarray]:
    """K-fold cross-validation on the development set.

    For each fold, a fresh copy of the full Pipeline (preprocessing + model) is fitted on the
    training folds only, then predicts the validation fold.
    Returns (one row of metrics per fold, out-of-fold probabilities for every row of X).
    """
    oof = np.full(len(y), np.nan)
    rows = []
    for k, (tr, va) in enumerate(cv.split(X, y), start=1):
        fitted = clone(model).fit(X.iloc[tr], y.iloc[tr])
        proba = fitted.predict_proba(X.iloc[va])[:, 1]
        oof[va] = proba
        rows.append({"fold": k, **compute_metrics(y.iloc[va], proba, threshold)})
        logger.info("Fold %d: PR-AUC %.4f, ROC-AUC %.4f", k, rows[-1]["pr_auc"], rows[-1]["roc_auc"])
    return pd.DataFrame(rows).set_index("fold"), oof


def summarize(per_fold: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation across folds for each metric."""
    return per_fold[METRIC_NAMES].agg(["mean", "std"]).T


def format_summary(per_fold: pd.DataFrame) -> dict[str, str]:
    """'0.745 +/- 0.009' strings, for printing and for the report."""
    s = summarize(per_fold)
    return {m: f"{s.loc[m, 'mean']:.3f} +/- {s.loc[m, 'std']:.3f}" for m in METRIC_NAMES}
