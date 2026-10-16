"""Run every experiment and write tables, plots and a log to results/.

Usage: python scripts/run_experiments.py   (or: make run)
"""

import time

import pandas as pd

from mleval.config import N_REPEATS, N_SPLITS, RESULTS_DIR, SEED
from mleval.data import load_data
from mleval.evaluation import METRICS, evaluate_all
from mleval.leakage import run_experiment_a_real, run_experiment_b_synthetic
from mleval.pipelines import MODEL_REGISTRY
from mleval.plots import plot_confusion_matrix, plot_roc_curves

METRIC_LABELS = {"roc_auc": "ROC AUC", "recall": "Recall", "precision": "Precision", "f1": "F1"}
LOG_LINES: list[str] = []


def log(text: str = "") -> None:
    print(text)
    LOG_LINES.append(text)


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def results_markdown(summary: pd.DataFrame) -> str:
    headers = ["Model", *(METRIC_LABELS[m] for m in METRICS)]
    rows = [
        [name, *(f"{r[f'{m}_mean']:.4f} +/- {r[f'{m}_std']:.4f}" for m in METRICS)]
        for name, r in summary.iterrows()
    ]
    return markdown_table(headers, rows)


def leakage_markdown(table: pd.DataFrame) -> str:
    headers = [
        "Metric",
        "Wrong mean",
        "Wrong std",
        "Right mean",
        "Right std",
        "Gap (wrong - right)",
    ]
    rows = [
        [
            m,
            f"{r.wrong_mean:.4f}",
            f"{r.wrong_std:.4f}",
            f"{r.right_mean:.4f}",
            f"{r.right_std:.4f}",
            f"{r.gap:+.4f}",
        ]
        for m, r in table.iterrows()
    ]
    return markdown_table(headers, rows)


def timed(label: str, start: float) -> None:
    log(f"[{label}] elapsed: {time.perf_counter() - start:.1f} s")


def main() -> None:
    total_start = time.perf_counter()
    RESULTS_DIR.mkdir(exist_ok=True)
    X, y = load_data()
    log("ml-model-evaluation: public sklearn Breast Cancer Wisconsin data, educational use only")
    log(f"Data: {X.shape[0]} rows, {X.shape[1]} features, malignant (positive) = {int(y.sum())}")
    log(f"Seed: {SEED}. Cross-validation: repeated stratified {N_SPLITS}-fold x {N_REPEATS}")
    log()

    # 1. Main model comparison
    start = time.perf_counter()
    _, summary = evaluate_all(MODEL_REGISTRY, X, y)
    summary.round(4).to_csv(RESULTS_DIR / "results_table.csv")
    results_md = results_markdown(summary)
    (RESULTS_DIR / "results_table.md").write_text(
        f"## Model comparison (mean +/- std over {N_SPLITS * N_REPEATS} folds)\n\n{results_md}\n"
    )
    log(f"Model comparison (mean +/- std over {N_SPLITS * N_REPEATS} folds)")
    log(results_md)
    timed("model comparison", start)
    log()

    # 2. Leakage experiments
    start = time.perf_counter()
    leak_a = run_experiment_a_real()
    title_a = "Leakage experiment A (REAL data): scaler + SelectKBest k=10 + logistic regression"
    leak_a.round(4).assign(data="real").to_csv(RESULTS_DIR / "leakage_real.csv")
    log(title_a)
    log(leakage_markdown(leak_a))
    timed("leakage A", start)
    log()

    start = time.perf_counter()
    leak_b = run_experiment_b_synthetic()
    title_b = (
        "Leakage experiment B (SYNTHETIC data): 569 x 500 pure noise features with the real "
        "labels, SelectKBest k=10 + logistic regression"
    )
    leak_b.round(4).assign(data="synthetic").to_csv(RESULTS_DIR / "leakage_synthetic.csv")
    log(title_b)
    log(leakage_markdown(leak_b))
    timed("leakage B (SYNTHETIC)", start)
    log()
    (RESULTS_DIR / "leakage.md").write_text(
        f"## {title_a}\n\n{leakage_markdown(leak_a)}\n\n"
        f"## {title_b}\n\n{leakage_markdown(leak_b)}\n"
    )

    # 3. Plots (out-of-fold predictions)
    start = time.perf_counter()
    aucs = plot_roc_curves(MODEL_REGISTRY, X, y, RESULTS_DIR / "roc_curves.png")
    cm = plot_confusion_matrix(
        MODEL_REGISTRY["Logistic regression"], X, y, RESULTS_DIR / "confusion_matrix.png"
    )
    log("Out-of-fold plots (single 5-fold stratified split, pooled predictions)")
    for name, auc in aucs.items():
        log(f"  pooled out-of-fold ROC AUC, {name}: {auc:.4f}")
    log("  Confusion matrix, logistic regression (rows = true, columns = predicted)")
    log(f"    true benign:    predicted benign {cm[0, 0]}, predicted malignant {cm[0, 1]}")
    log(f"    true malignant: predicted benign {cm[1, 0]}, predicted malignant {cm[1, 1]}")
    timed("plots", start)
    log()

    timed("total", total_start)
    (RESULTS_DIR / "run_log.txt").write_text("\n".join(LOG_LINES) + "\n")


if __name__ == "__main__":
    main()
