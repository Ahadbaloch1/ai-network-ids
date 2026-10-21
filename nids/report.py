"""Plots for the evaluation report."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def save_reports(metrics, importance, outdir):
    """Write confusion_matrix.png and feature_importance.png into outdir."""
    os.makedirs(outdir, exist_ok=True)
    paths = []

    labels, matrix = metrics["labels"], metrics["confusion"]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Confusion matrix (held-out test set)")
    threshold = matrix.max() / 2.0
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, int(matrix[i, j]), ha="center", va="center",
                    color="white" if matrix[i, j] > threshold else "black")
    fig.tight_layout()
    path = os.path.join(outdir, "confusion_matrix.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    paths.append(path)

    top = importance.head(12)[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(np.arange(len(top)), top.values, color="#3b6fb6")
    ax.set_yticks(np.arange(len(top)))
    ax.set_yticklabels(top.index)
    ax.set_xlabel("Importance")
    ax.set_title("Top feature importances (Random Forest)")
    fig.tight_layout()
    path = os.path.join(outdir, "feature_importance.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    paths.append(path)
    return paths
