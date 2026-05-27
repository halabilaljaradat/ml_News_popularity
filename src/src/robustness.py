from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def xgboost_training_size_robustness(
    X_train,
    X_test,
    y_train,
    y_test,
    fractions=(0.1, 0.3, 0.5, 0.99),
) -> pd.DataFrame:
    """Evaluate XGBoost accuracy as the training set size is reduced."""
    results = []
    for frac in fractions:
        X_frac, _, y_frac, _ = train_test_split(
            X_train,
            y_train,
            train_size=frac,
            random_state=42,
            stratify=y_train,
        )
        model = XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss",
        )
        model.fit(X_frac, y_frac)
        acc = accuracy_score(y_test, model.predict(X_test))
        results.append({"fraction_of_training_data": frac, "test_accuracy": acc})
    return pd.DataFrame(results)


def plot_robustness_results(results_df: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(
        results_df["fraction_of_training_data"],
        results_df["test_accuracy"],
        marker="o",
    )
    plt.title("XGBoost — Accuracy vs Training Size")
    plt.xlabel("Fraction of Training Data")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
