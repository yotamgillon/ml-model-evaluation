import numpy as np
import pytest

from mleval.config import SEED
from mleval.data import load_data
from mleval.evaluation import make_cv

N_SPLITS, N_REPEATS = 2, 3


@pytest.fixture(scope="module")
def folds_and_labels():
    X, y = load_data()
    cv = make_cv(N_SPLITS, N_REPEATS, SEED)
    return list(cv.split(X, y)), y.to_numpy()


def test_number_of_splits(folds_and_labels):
    folds, _ = folds_and_labels
    assert len(folds) == N_SPLITS * N_REPEATS
    assert make_cv(N_SPLITS, N_REPEATS, SEED).get_n_splits() == N_SPLITS * N_REPEATS


def test_train_and_test_sizes_sum_to_569_and_are_disjoint(folds_and_labels):
    folds, _ = folds_and_labels
    for train_idx, test_idx in folds:
        assert len(train_idx) + len(test_idx) == 569
        assert set(train_idx).isdisjoint(test_idx)


def test_test_folds_of_each_repeat_cover_every_row_once(folds_and_labels):
    folds, _ = folds_and_labels
    for r in range(N_REPEATS):
        test_rows = np.concatenate([folds[r * N_SPLITS + k][1] for k in range(N_SPLITS)])
        assert sorted(test_rows) == list(range(569))


def test_class_ratio_in_every_test_fold_is_stratified(folds_and_labels):
    folds, y = folds_and_labels
    overall = 212 / 569
    for _, test_idx in folds:
        assert abs(y[test_idx].mean() - overall) <= 0.02
