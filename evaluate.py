"""
Evaluation Module for Sentiment Analysis Models.

Author: Surag M S
Project: Syntecxhub AI Internship - Project 2
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def calculate_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
    """Calculate key classification metrics.

    Args:
        y_true: Ground truth target values.
        y_pred: Predicted target values.

    Returns:
        Dict[str, float]: Dictionary containing accuracy, precision, recall, and f1_score.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    return {
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
    }


def generate_classification_report_str(
    y_true: List[str], y_pred: List[str], target_names: Optional[List[str]] = None
) -> str:
    """Generate a detailed classification report string.

    Args:
        y_true: Ground truth target values.
        y_pred: Predicted target values.
        target_names: Optional list of class names.

    Returns:
        str: Formatted classification report.
    """
    return classification_report(
        y_true, y_pred, target_names=target_names, zero_division=0, digits=4
    )


def generate_confusion_matrix(
    y_true: List[str],
    y_pred: List[str],
    labels: Optional[List[str]] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> np.ndarray:
    """Compute and optionally plot/save the confusion matrix.

    Args:
        y_true: Ground truth target values.
        y_pred: Predicted target values.
        labels: List of unique class labels.
        output_path: File path to save the confusion matrix image.

    Returns:
        np.ndarray: Confusion matrix array.
    """
    if labels is None:
        labels = sorted(list(set(y_true) | set(y_pred)))

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(6, 5))
        cax = ax.matshow(cm, cmap=plt.cm.Blues)
        fig.colorbar(cax)

        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="left")
        ax.set_yticklabels(labels)

        # Annotate matrix entries with values
        for i in range(len(labels)):
            for j in range(len(labels)):
                color = "white" if cm[i, j] > cm.max() / 2 else "black"
                ax.text(
                    j,
                    i,
                    str(cm[i, j]),
                    va="center",
                    ha="center",
                    color=color,
                    fontsize=12,
                    fontweight="bold",
                )

        ax.set_xlabel("Predicted Label", fontweight="bold")
        ax.set_ylabel("True Label", fontweight="bold")
        ax.set_title("Confusion Matrix - Sentiment Analysis", fontweight="bold", pad=20)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()

    return cm


def save_metrics(
    metrics_dict: Dict[str, float],
    report_str: str,
    confusion_mat: np.ndarray,
    labels: List[str],
    comparison_str: str,
    output_path: Union[str, Path],
) -> None:
    """Save comprehensive evaluation summary to a text file.

    Args:
        metrics_dict: Dictionary of primary metrics.
        report_str: Detailed classification report string.
        confusion_mat: Confusion matrix array.
        labels: List of label names.
        comparison_str: Formatted string comparing tested models.
        output_path: File path for destination text file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("==================================================\n")
        f.write("       SYNTECXHUB AI INTERNSHIP PROJECT 2\n")
        f.write("        SENTIMENT ANALYSIS MODEL METRICS\n")
        f.write("==================================================\n")
        f.write("Author: Surag M S\n\n")

        f.write("========================================\n")
        f.write("MODEL COMPARISON\n")
        f.write("========================================\n")
        f.write(comparison_str + "\n\n")

        f.write("========================================\n")
        f.write("SELECTED BEST MODEL METRICS\n")
        f.write("========================================\n")
        f.write(f"Accuracy : {metrics_dict['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics_dict['precision']:.4f}\n")
        f.write(f"Recall   : {metrics_dict['recall']:.4f}\n")
        f.write(f"F1 Score : {metrics_dict['f1_score']:.4f}\n\n")

        f.write("========================================\n")
        f.write("CLASSIFICATION REPORT\n")
        f.write("========================================\n")
        f.write(report_str + "\n\n")

        f.write("========================================\n")
        f.write("CONFUSION MATRIX\n")
        f.write("========================================\n")
        f.write(f"Labels: {labels}\n")
        f.write(str(confusion_mat) + "\n")
