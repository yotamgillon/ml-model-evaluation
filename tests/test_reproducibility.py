import pytest

from mleval.config import SEED
from mleval.data import load_data
from mleval.evaluation import evaluate_model
from mleval.pipelines import build_dummy, build_logistic_regression

N_SPLITS, N_REPEATS = 2, 3


@pytest.fixture(scope="module")
def data():
    return load_data()


@pytest.mark.parametrize("factory", [build_dummy, build_logistic_regression])
def test_same_seed_gives_identical_per_fold_scores(data, factory):
    X, y = data
    a, _ = evaluate_model(factory, X, y, N_SPLITS, N_REPEATS, SEED)
    b, _ = evaluate_model(factory, X, y, N_SPLITS, N_REPEATS, SEED)
    assert a.equals(b)


def test_different_seed_gives_different_per_fold_scores(data):
    X, y = data
    a, _ = evaluate_model(build_logistic_regression, X, y, N_SPLITS, N_REPEATS, SEED)
    b, _ = evaluate_model(build_logistic_regression, X, y, N_SPLITS, N_REPEATS, SEED + 1)
    assert not a.equals(b)
