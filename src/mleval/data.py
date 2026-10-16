"""Data loading.

Real data: scikit-learn's built-in Breast Cancer Wisconsin (Diagnostic) dataset.
It is public, ships with scikit-learn and needs no download. Educational use only.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

from mleval.config import N_NOISE_FEATURES, SEED


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) with MALIGNANT as the positive class (y == 1).

    scikit-learn encodes malignant as 0 and benign as 1. We flip the labels so
    that "positive" means "malignant". Then recall answers the question
    "what share of malignant cases did the model catch?".
    """
    bunch = load_breast_cancer(as_frame=True)
    X = bunch.data.copy()
    y = (1 - bunch.target).astype(int).rename("malignant")
    return X, y


def make_synthetic_noise(
    n_samples: int,
    n_features: int = N_NOISE_FEATURES,
    seed: int = SEED,
) -> pd.DataFrame:
    """SYNTHETIC data: a matrix of pure standard-normal noise.

    This is NOT real data and has no relation to any label. It is used only in
    the synthetic leakage experiment, paired with the REAL labels y. Because the
    features carry no signal, any score clearly above chance is leakage.
    """
    rng = np.random.default_rng(seed)
    values = rng.standard_normal((n_samples, n_features))
    columns = [f"synthetic_noise_{i}" for i in range(n_features)]
    return pd.DataFrame(values, columns=columns)
