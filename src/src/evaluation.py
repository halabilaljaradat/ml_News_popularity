import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
)


def predict_scores(model, X):
    """Return predicted probabilities or decision scores for positive class."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        # Map arbitrary decision scores to [0, 1] for metric compatibility.
        return (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)
    return None


def evaluate_classifier(model, X_train, X_test, y_train, y_test, model_name: str) -> dict:
    """Evaluate a classification model with common metrics."""
    y_pred = model.predict(X_test)
    y_score = predict_scores(model, X_test)

    result = {
        "model": model_name,
        "train_acc": model.score(X_train, y_train),
        "test_acc": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }

    if y_score is not None:
        result["roc_auc"] = roc_auc_score(y_test, y_score)
        result["brier_score"] = brier_score_loss(y_test, y_score)
    else:
        result["roc_auc"] = np.nan
        result["brier_score"] = np.nan

    return result


def print_classifier_report(model, X_test, y_test, target_names=None) -> None:
    """Print a classification report and confusion matrix."""
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=target_names))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
