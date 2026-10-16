"""Central constants. Every random process in the project uses SEED."""

from pathlib import Path

SEED = 42

# Cross-validation defaults for the main experiments.
N_SPLITS = 5
N_REPEATS = 10

# Synthetic noise experiment (pure noise, see data.make_synthetic_noise).
N_NOISE_FEATURES = 500

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
