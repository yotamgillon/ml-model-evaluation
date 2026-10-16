import pandas as pd
import pytest

from mleval.config import SEED
from mleval.leakage import leakage_gap, run_experiment_a_real, run_experiment_b_synthetic

N_SPLITS, N_REPEATS = 2, 3


def summary(roc_auc_mean: float) -> pd.DataFrame:
    return pd.DataFrame({"roc_auc_mean": [roc_auc_mean]})


def test_leakage_gap_on_hand_made_frames():
    assert leakage_gap(summary(0.90), summary(0.70)) == pytest.approx(0.20)
    assert leakage_gap(summary(0.80), summary(0.80)) == pytest.approx(0.0)
    assert leakage_gap(summary(0.60), summary(0.75)) == pytest.approx(-0.15)


def gap_from_table(table: pd.DataFrame) -> float:
    return leakage_gap(
        summary(table.loc["roc_auc", "wrong_mean"]), summary(table.loc["roc_auc", "right_mean"])
    )


def test_synthetic_noise_leakage_is_large():
    """SYNTHETIC: pure noise features with real labels."""
    table = run_experiment_b_synthetic(N_SPLITS, N_REPEATS, SEED)
    assert gap_from_table(table) > 0.10
    assert table.loc["roc_auc", "right_mean"] < 0.60


def test_real_data_leakage_gap_is_small():
    table = run_experiment_a_real(N_SPLITS, N_REPEATS, SEED)
    assert abs(gap_from_table(table)) < 0.01
