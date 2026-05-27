from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    auc,
    brier_score_loss,
)
from .evaluation import predict_scores


def save_accuracy_comparison(results_df: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ax = results_df.sort_values("test_acc").plot(
        x="model", y=["train_acc", "test_acc"], kind="barh", figsize=(10, 6)
    )
    ax.set_title("Classification Model Accuracy Comparison")
    ax.set_xlabel("Accuracy")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_roc_curves(models: dict, X_test, y_test, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 7))
    for name, model in models.items():
        y_score = predict_scores(model, X_test)
        if y_score is None:
            continue
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random (AUC=0.50)")
    plt.title("ROC Curves")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_precision_recall_curves(models: dict, X_test, y_test, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 7))
    for name, model in models.items():
        y_score = predict_scores(model, X_test)
        if y_score is None:
            continue
        precision, recall, _ = precision_recall_curve(y_test, y_score)
        ap = average_precision_score(y_test, y_score)
        plt.plot(recall, precision, label=f"{name} (AP={ap:.2f})")
    plt.title("Precision-Recall Curves")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_calibration_curves(models: dict, X_test, y_test, output_path: str | Path) -> dict:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    brier_scores = {}
    plt.figure(figsize=(10, 7))
    for name, model in models.items():
        y_score = predict_scores(model, X_test)
        if y_score is None:
            continue
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_test, y_score, n_bins=10
        )
        plt.plot(mean_predicted_value, fraction_of_positives, label=name)
        brier_scores[name] = brier_score_loss(y_test, y_score)
    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    plt.title("Calibration Curves")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return brier_scores
