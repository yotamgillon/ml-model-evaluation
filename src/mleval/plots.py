"""Plots built from OUT-OF-FOLD predictions (each row is predicted by a model that never saw it)."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_predict  # noqa: E402

from mleval.config import SEED  # noqa: E402

# Colorblind-safe hues, assigned in a fixed order per model.
MODEL_COLORS = {
    "Logistic regression": "#0072B2",
    "Random forest": "#E69F00",
    "Gradient boosting": "#009E73",
}
DUMMY_COLOR = "#6B7280"
DUMMY_NAME = "Dummy (most frequent)"


def _folds() -> StratifiedKFold:
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def plot_roc_curves(registry, X, y, path) -> dict[str, float]:
    """Out-of-fold ROC curves for every model. Returns the pooled out-of-fold AUC per model."""
    fig, ax = plt.subplots(figsize=(6.4, 5.4), dpi=150)
    aucs = {}
    for name, factory in registry.items():
        proba = cross_val_predict(factory(), X, y, cv=_folds(), method="predict_proba")[:, 1]
        fpr, tpr, _ = roc_curve(y, proba)
        aucs[name] = roc_auc_score(y, proba)
        label = f"{name} (AUC {aucs[name]:.3f})"
        if name == DUMMY_NAME:
            ax.plot(fpr, tpr, color=DUMMY_COLOR, lw=1.5, ls="--", label=label)
        else:
            ax.plot(fpr, tpr, color=MODEL_COLORS[name], lw=2, label=label)
    ax.set_xlabel("False positive rate (benign flagged as malignant)")
    ax.set_ylabel("True positive rate (recall for malignant)")
    ax.set_title(
        "ROC curves from out-of-fold predictions\n(5-fold stratified CV, pooled predictions)",
        fontsize=11,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return aucs


def plot_confusion_matrix(factory, X, y, path, model_name="Logistic regression") -> np.ndarray:
    """Out-of-fold confusion matrix. Rows = true class, columns = predicted class."""
    pred = cross_val_predict(factory(), X, y, cv=_folds(), method="predict")
    cm = confusion_matrix(y, pred, labels=[0, 1])
    labels = ["benign", "malignant"]
    fig, ax = plt.subplots(figsize=(5.2, 4.6), dpi=150)
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(
        f"{model_name}: confusion matrix\nout-of-fold predictions (5-fold stratified CV)",
        fontsize=11,
    )
    threshold = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=14,
                color="white" if cm[i, j] > threshold else "black",
            )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return cm
