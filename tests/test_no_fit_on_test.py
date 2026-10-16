"""Prove the pipeline never fits on test rows, and that the check can detect leakage."""

import numpy as np
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mleval.config import SEED
from mleval.data import load_data
from mleval.evaluation import make_cv

N_SPLITS, N_REPEATS = 2, 3


class RowRecorder(BaseEstimator, TransformerMixin):
    """Pass-through transformer that remembers a hash of every row it sees in fit."""

    def fit(self, X, y=None):
        self.seen_ = {hash(row.tobytes()) for row in np.asarray(X)}
        return self

    def transform(self, X):
        return np.asarray(X)


def row_hashes(X) -> set[int]:
    return {hash(row.tobytes()) for row in np.asarray(X)}


def fit_saw_only_train(recorder: RowRecorder, X_train, X_test) -> bool:
    """True if the recorder saw exactly the train rows and none of the test rows."""
    return recorder.seen_ == row_hashes(X_train) and recorder.seen_.isdisjoint(row_hashes(X_test))


def fit_pipeline_right(X_train, y_train) -> RowRecorder:
    """Recorder inside the Pipeline, fitted on the training rows only."""
    recorder = RowRecorder()
    pipe = Pipeline(
        [
            ("recorder", recorder),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=SEED)),
        ]
    )
    pipe.fit(X_train, y_train)
    return pipe.named_steps["recorder"]


def fit_preprocessing_wrong(X_all) -> RowRecorder:
    """Deliberately WRONG: the preprocessing step is fitted on ALL rows before the split."""
    return RowRecorder().fit(X_all)


@pytest.fixture(scope="module")
def data():
    X, y = load_data()
    X = X.to_numpy()
    assert len(row_hashes(X)) == len(X)  # rows are unique, so hashes identify rows
    return X, y.to_numpy()


def test_pipeline_fits_only_on_train_rows_in_every_fold(data):
    X, y = data
    folds = list(make_cv(N_SPLITS, N_REPEATS, SEED).split(X, y))
    assert len(folds) == N_SPLITS * N_REPEATS
    for train_idx, test_idx in folds:
        recorder = fit_pipeline_right(X[train_idx], y[train_idx])
        assert fit_saw_only_train(recorder, X[train_idx], X[test_idx])


def test_negative_control_check_detects_fit_on_all_rows(data):
    X, y = data
    for train_idx, test_idx in make_cv(N_SPLITS, N_REPEATS, SEED).split(X, y):
        recorder = fit_preprocessing_wrong(X)
        assert not fit_saw_only_train(recorder, X[train_idx], X[test_idx])
