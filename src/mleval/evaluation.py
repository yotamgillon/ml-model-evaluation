"""Repeated stratified k-fold evaluation.

The positive class is malignant (y == 1). Precision, recall and F1 use
zero_division=0, so a model that never predicts malignant (the Dummy baseline)
scores 0 instead of raising a warning.
"""

from collections.abc import Callable, Mapping

import pandas as pd
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from mleval.config import N_REPEATS, N_SPLITS, SEED

METRICS = ("roc_auc", "recall", "precision", "f1")

SCORING = {
    "roc_auc": "roc_auc",
    "recall": make_scorer(recall_score, pos_label=1, zero_division=0),
    "precision": make_scorer(precision_score, pos_label=1, zero_division=0),
    "f1": make_scorer(f1_score, pos_label=1, zero_division=0),
}


def make_cv(n_splits: int = N_SPLITS, n_repeats: int = N_REPEATS, seed: int = SEED):
    return RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)


def evaluate_model(
    pipeline_factory: Callable[[], Pipeline],
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = N_SPLITS,
    n_repeats: int = N_REPEATS,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cross-validate one model.

    Returns (per_fold, summary). per_fold has one row per fold with the four
    metrics. summary has one row with mean and std of each metric over folds.
    """
    cv = make_cv(n_splits, n_repeats, seed)
    raw = cross_validate(pipeline_factory(), X, y, cv=cv, scoring=SCORING, n_jobs=1)
    per_fold = pd.DataFrame({m: raw[f"test_{m}"] for m in METRICS})
    per_fold.index.name = "fold"
    return per_fold, summarize(per_fold)


def summarize(per_fold: pd.DataFrame) -> pd.DataFrame:
    """One row: <metric>_mean and <metric>_std over the folds."""
    row = {}
    for m in METRICS:
        row[f"{m}_mean"] = per_fold[m].mean()
        row[f"{m}_std"] = per_fold[m].std()
    return pd.DataFrame([row])


def evaluate_all(
    registry: Mapping[str, Callable[[], Pipeline]],
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = N_SPLITS,
    n_repeats: int = N_REPEATS,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate every model in the registry on identical folds.

    Returns (per_fold, summary). Both are indexed or keyed by model name.
    """
    folds, summaries = [], []
    for name, factory in registry.items():
        per_fold, summary = evaluate_model(factory, X, y, n_splits, n_repeats, seed)
        folds.append(per_fold.assign(model=name).reset_index())
        summaries.append(summary.assign(model=name))
    per_fold_all = pd.concat(folds, ignore_index=True)[["model", "fold", *METRICS]]
    summary_all = pd.concat(summaries, ignore_index=True).set_index("model")
    return per_fold_all, summary_all
