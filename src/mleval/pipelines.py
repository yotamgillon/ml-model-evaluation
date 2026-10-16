"""Model pipelines.

Every model is a scikit-learn Pipeline whose first step is a StandardScaler.
Scaling therefore happens inside each cross-validation fold and is fitted on
the training part only. Nothing is fitted on test data.
"""

from collections.abc import Callable

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mleval.config import SEED


def _with_scaler(classifier) -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("clf", classifier)])


def build_dummy() -> Pipeline:
    """Baseline: always predicts the most frequent class (benign)."""
    return _with_scaler(DummyClassifier(strategy="most_frequent", random_state=SEED))


def build_logistic_regression() -> Pipeline:
    """Simple baseline: logistic regression with default regularisation."""
    return _with_scaler(LogisticRegression(max_iter=1000, random_state=SEED))


def build_random_forest() -> Pipeline:
    return _with_scaler(RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=1))


def build_gradient_boosting() -> Pipeline:
    return _with_scaler(GradientBoostingClassifier(random_state=SEED))


# name -> factory returning a fresh, unfitted pipeline
MODEL_REGISTRY: dict[str, Callable[[], Pipeline]] = {
    "Dummy (most frequent)": build_dummy,
    "Logistic regression": build_logistic_regression,
    "Random forest": build_random_forest,
    "Gradient boosting": build_gradient_boosting,
}
