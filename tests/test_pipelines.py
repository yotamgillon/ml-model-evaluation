import numpy as np
import pytest
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mleval.config import SEED
from mleval.data import load_data
from mleval.pipelines import MODEL_REGISTRY, build_dummy


@pytest.fixture(scope="module")
def tiny_split():
    X, y = load_data()
    X_small, _, y_small, _ = train_test_split(
        X, y, train_size=120, stratify=y, random_state=SEED
    )
    return train_test_split(X_small, y_small, test_size=0.25, stratify=y_small, random_state=SEED)


def test_registry_has_expected_models():
    assert list(MODEL_REGISTRY) == [
        "Dummy (most frequent)",
        "Logistic regression",
        "Random forest",
        "Gradient boosting",
    ]


def test_factory_returns_fresh_pipeline_each_call():
    for factory in MODEL_REGISTRY.values():
        assert factory() is not factory()


@pytest.mark.parametrize("name", list(MODEL_REGISTRY))
def test_first_step_is_scaler(name):
    pipe = MODEL_REGISTRY[name]()
    assert isinstance(pipe, Pipeline)
    first_name, first_step = pipe.steps[0]
    assert first_name == "scaler"
    assert isinstance(first_step, StandardScaler)


@pytest.mark.parametrize("name", list(MODEL_REGISTRY))
def test_fits_and_predicts_on_tiny_subset(name, tiny_split):
    X_train, X_test, y_train, _ = tiny_split
    pipe = MODEL_REGISTRY[name]()
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    assert pred.shape == (len(X_test),)
    assert set(np.unique(pred)) <= {0, 1}
    proba = pipe.predict_proba(X_test)
    assert proba.shape == (len(X_test), 2)


def test_dummy_predicts_only_majority_class_benign(tiny_split):
    X_train, X_test, y_train, _ = tiny_split
    assert y_train.mean() < 0.5  # benign (0) is the majority
    pipe = build_dummy().fit(X_train, y_train)
    assert set(np.unique(pipe.predict(X_test))) == {0}
