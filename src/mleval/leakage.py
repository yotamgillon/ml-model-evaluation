"""Leakage experiments: preprocessing fitted OUTSIDE versus INSIDE cross-validation.

WRONG: scaler and/or feature selector are fitted on the full data before the
split, so every test fold has already influenced the preprocessing (and the
selector has seen the test labels).
RIGHT: the same steps live in one Pipeline, so they are refitted on the
training part of each fold only.

Both ways use the same classifier, folds and metrics.
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mleval.config import N_REPEATS, N_SPLITS, SEED
from mleval.data import load_data, make_synthetic_noise
from mleval.evaluation import METRICS, evaluate_model

K_FEATURES = 10
N_SYNTHETIC_SAMPLES = 569
N_SYNTHETIC_FEATURES = 500


def _classifier_only() -> Pipeline:
    return Pipeline([("clf", LogisticRegression(max_iter=1000, random_state=SEED))])


def _evaluate_wrong(preprocessors, X, y, n_splits, n_repeats, seed) -> pd.DataFrame:
    """WRONG: fit preprocessing on ALL rows (and labels), then cross-validate."""
    X_transformed = X
    for step in preprocessors:
        step.fit(X_transformed, y)
        X_transformed = step.transform(X_transformed)
    X_transformed = np.asarray(X_transformed)
    return evaluate_model(_classifier_only, X_transformed, y, n_splits, n_repeats, seed)[1]


def _evaluate_right(steps, X, y, n_splits, n_repeats, seed) -> pd.DataFrame:
    """RIGHT: the same preprocessing inside the Pipeline, refit on each training fold."""

    def factory() -> Pipeline:
        clf = LogisticRegression(max_iter=1000, random_state=SEED)
        return Pipeline([*steps(), ("clf", clf)])

    return evaluate_model(factory, X, y, n_splits, n_repeats, seed)[1]


def leakage_gap(wrong_df: pd.DataFrame, right_df: pd.DataFrame) -> float:
    """ROC AUC (mean) of the wrong way minus the right way, from one-row summaries."""
    return float(wrong_df["roc_auc_mean"].iloc[0] - right_df["roc_auc_mean"].iloc[0])


def compare(wrong_df: pd.DataFrame, right_df: pd.DataFrame) -> pd.DataFrame:
    """Rows = metrics. Columns: wrong/right mean and std, and gap = wrong mean - right mean."""
    rows = {}
    for m in METRICS:
        w_mean, r_mean = wrong_df[f"{m}_mean"].iloc[0], right_df[f"{m}_mean"].iloc[0]
        rows[m] = {
            "wrong_mean": w_mean,
            "wrong_std": wrong_df[f"{m}_std"].iloc[0],
            "right_mean": r_mean,
            "right_std": right_df[f"{m}_std"].iloc[0],
            "gap": w_mean - r_mean,
        }
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("metric")


def run_experiment_a_real(
    n_splits: int = N_SPLITS, n_repeats: int = N_REPEATS, seed: int = SEED
) -> pd.DataFrame:
    """Experiment A (REAL data): scaler + SelectKBest(f_classif, k=10), then logistic regression."""
    X, y = load_data()
    wrong = _evaluate_wrong(
        [StandardScaler(), SelectKBest(f_classif, k=K_FEATURES)], X, y, n_splits, n_repeats, seed
    )
    right = _evaluate_right(
        lambda: [("scaler", StandardScaler()), ("select", SelectKBest(f_classif, k=K_FEATURES))],
        X,
        y,
        n_splits,
        n_repeats,
        seed,
    )
    return compare(wrong, right)


def run_experiment_b_synthetic(
    n_splits: int = N_SPLITS, n_repeats: int = N_REPEATS, seed: int = SEED
) -> pd.DataFrame:
    """Experiment B (SYNTHETIC data): 569 x 500 pure noise features with the REAL labels.

    The features carry no information about y, so honest scores should be near
    chance. SelectKBest(f_classif, k=10), then logistic regression.
    """
    _, y = load_data()
    X_noise = make_synthetic_noise(N_SYNTHETIC_SAMPLES, N_SYNTHETIC_FEATURES, seed)
    wrong = _evaluate_wrong(
        [SelectKBest(f_classif, k=K_FEATURES)], X_noise, y, n_splits, n_repeats, seed
    )
    right = _evaluate_right(
        lambda: [("select", SelectKBest(f_classif, k=K_FEATURES))],
        X_noise,
        y,
        n_splits,
        n_repeats,
        seed,
    )
    return compare(wrong, right)
