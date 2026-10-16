import pytest

from mleval.config import SEED
from mleval.data import load_data
from mleval.evaluation import METRICS, evaluate_all, evaluate_model
from mleval.pipelines import MODEL_REGISTRY, build_gradient_boosting, build_logistic_regression

N_SPLITS, N_REPEATS = 2, 3
FAST = {name: MODEL_REGISTRY[name] for name in ("Dummy (most frequent)", "Logistic regression")}


@pytest.fixture(scope="module")
def data():
    return load_data()


@pytest.fixture(scope="module")
def fast_result(data):
    X, y = data
    return evaluate_all(FAST, X, y, N_SPLITS, N_REPEATS, SEED)


def test_number_of_folds_and_metrics(fast_result):
    per_fold, summary = fast_result
    assert set(per_fold["model"]) == set(FAST)
    for name in FAST:
        assert (per_fold["model"] == name).sum() == N_SPLITS * N_REPEATS == 6
    for m in METRICS:
        assert m in per_fold.columns
        assert f"{m}_mean" in summary.columns
        assert f"{m}_std" in summary.columns
    assert list(summary.index) == list(FAST)


def test_dummy_scores(fast_result):
    _, summary = fast_result
    dummy = summary.loc["Dummy (most frequent)"]
    assert dummy["recall_mean"] == 0
    assert dummy["precision_mean"] == 0
    assert dummy["f1_mean"] == 0
    assert dummy["roc_auc_mean"] == pytest.approx(0.5)


def test_logistic_regression_clearly_beats_dummy(fast_result):
    _, summary = fast_result
    lr = summary.loc["Logistic regression"]["roc_auc_mean"]
    dummy = summary.loc["Dummy (most frequent)"]["roc_auc_mean"]
    assert lr > dummy + 0.4


def test_every_model_uses_fixed_seed():
    for factory in MODEL_REGISTRY.values():
        assert factory().named_steps["clf"].random_state == SEED


@pytest.mark.parametrize("factory", [build_logistic_regression, build_gradient_boosting])
def test_same_seed_gives_identical_scores(data, factory):
    X, y = data
    first, _ = evaluate_model(factory, X, y, N_SPLITS, N_REPEATS, SEED)
    second, _ = evaluate_model(factory, X, y, N_SPLITS, N_REPEATS, SEED)
    assert first.equals(second)
