import numpy as np

from mleval.data import load_data, make_synthetic_noise


def test_shape_is_569_by_30():
    X, y = load_data()
    assert X.shape == (569, 30)
    assert y.shape == (569,)


def test_positive_class_is_malignant_212_of_569():
    _, y = load_data()
    assert set(y.unique()) == {0, 1}
    assert int(y.sum()) == 212
    assert int((y == 0).sum()) == 357


def test_noise_matrix_shape():
    noise = make_synthetic_noise(569, 500)
    assert noise.shape == (569, 500)
    assert noise.columns[0].startswith("synthetic_")


def test_noise_is_deterministic_for_a_seed():
    a = make_synthetic_noise(569, 500, seed=42)
    b = make_synthetic_noise(569, 500, seed=42)
    c = make_synthetic_noise(569, 500, seed=7)
    assert np.array_equal(a.to_numpy(), b.to_numpy())
    assert not np.array_equal(a.to_numpy(), c.to_numpy())
