"""Run Task A classification workflow for the DS230 final project.

Run from repository root:
    python scripts/run_task_a_classification.py
"""
from pathlib import Path
import sys
import joblib
import pandas as pd

# Make src importable when running as a script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data_loading import load_online_news_data
from src.data_cleaning import basic_cleaning, data_quality_summary, duplicate_summary
from src.features import add_category_label, add_engineered_features, log_transform_skewed_features
from src.preprocessing import define_classification_xy, split_classification_data, scale_selected_features
from src.feature_selection import select_features_by_three_methods
from src.models import (
    train_logistic_regression,
    train_knn_with_k_search,
    train_decision_tree,
    train_random_forest,
    tune_random_forest_randomized,
    train_sampled_rbf_svm,
    train_linear_svc_calibrated,
    train_xgboost,
)
from src.evaluation import evaluate_classifier
from src.visualization import (
    save_accuracy_comparison,
    plot_roc_curves,
    plot_precision_recall_curves,
    plot_calibration_curves,
)
from src.robustness import xgboost_training_size_robustness, plot_robustness_results


def main():
    data_path = PROJECT_ROOT / "data" / "raw" / "OnlineNewsPopularity.csv"
    processed_dir = PROJECT_ROOT / "data" / "processed"
    figures_dir = PROJECT_ROOT / "figures"
    models_dir = PROJECT_ROOT / "models"

    processed_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = load_online_news_data(data_path)
    df = basic_cleaning(df)
    df = add_category_label(df)
    df = add_engineered_features(df)

    print("Saving data-quality summaries...")
    data_quality_summary(df).to_csv(processed_dir / "data_quality_summary.csv")
    pd.DataFrame([duplicate_summary(df)]).to_csv(processed_dir / "duplicate_summary.csv", index=False)

    print("Defining classification target...")
    X, y, threshold = define_classification_xy(df)
    print(f"Median-share threshold: {threshold}")
    print(y.value_counts())

    print("Splitting data before feature selection/scaling...")
    X_train_raw, X_test_raw, y_train, y_test = split_classification_data(X, y)

    print("Applying log transforms to skewed features after the split...")
    X_train_raw, X_test_raw = log_transform_skewed_features(X_train_raw, X_test_raw)

    print("Selecting features using training data only...")
    feature_selection = select_features_by_three_methods(X_train_raw, y_train)
    selected_features = feature_selection["final_features"]
    pd.Series(selected_features, name="selected_features").to_csv(
        processed_dir / "selected_features.csv", index=False
    )
    print(f"Selected {len(selected_features)} features.")

    print("Scaling selected features using training data only...")
    X_train_scaled, X_test_scaled, scaler = scale_selected_features(
        X_train_raw, X_test_raw, selected_features
    )
    joblib.dump(scaler, models_dir / "scaler.joblib")

    print("Training classification models...")
    fitted_models = {}
    results = []

    lr = train_logistic_regression(X_train_scaled, y_train)
    fitted_models["Logistic Regression"] = lr
    results.append(evaluate_classifier(lr, X_train_scaled, X_test_scaled, y_train, y_test, "Logistic Regression"))

    knn, best_k, knn_accuracies = train_knn_with_k_search(X_train_scaled, X_test_scaled, y_train, y_test)
    fitted_models[f"KNN (k={best_k})"] = knn
    results.append(evaluate_classifier(knn, X_train_scaled, X_test_scaled, y_train, y_test, f"KNN (k={best_k})"))

    dt = train_decision_tree(X_train_scaled, y_train)
    fitted_models["Decision Tree"] = dt
    results.append(evaluate_classifier(dt, X_train_scaled, X_test_scaled, y_train, y_test, "Decision Tree"))

    rf = train_random_forest(X_train_scaled, y_train)
    fitted_models["Random Forest"] = rf
    results.append(evaluate_classifier(rf, X_train_scaled, X_test_scaled, y_train, y_test, "Random Forest"))

    rf_search = tune_random_forest_randomized(X_train_scaled, y_train)
    best_rf = rf_search.best_estimator_
    fitted_models["Random Forest Tuned"] = best_rf
    results.append(evaluate_classifier(best_rf, X_train_scaled, X_test_scaled, y_train, y_test, "Random Forest Tuned"))
    joblib.dump(best_rf, models_dir / "best_random_forest_classifier.joblib")

    # RBF SVM can be slow; sampled version follows the notebook's approach.
    svm_rbf, X_sample, y_sample = train_sampled_rbf_svm(X_train_scaled, y_train)
    fitted_models["SVM RBF Sampled"] = svm_rbf
    results.append(evaluate_classifier(svm_rbf, X_sample, X_test_scaled, y_sample, y_test, "SVM RBF Sampled"))

    svm_linear = train_linear_svc_calibrated(X_train_scaled, y_train)
    fitted_models["SVM Linear"] = svm_linear
    results.append(evaluate_classifier(svm_linear, X_train_scaled, X_test_scaled, y_train, y_test, "SVM Linear"))

    xgb = train_xgboost(X_train_scaled, y_train)
    fitted_models["XGBoost"] = xgb
    results.append(evaluate_classifier(xgb, X_train_scaled, X_test_scaled, y_train, y_test, "XGBoost"))
    joblib.dump(xgb, models_dir / "best_xgboost_classifier.joblib")

    print("Saving results...")
    results_df = pd.DataFrame(results).sort_values("test_acc", ascending=False)
    results_df.to_csv(processed_dir / "classification_results.csv", index=False)
    print(results_df)

    save_accuracy_comparison(results_df, figures_dir / "model_accuracy_comparison.png")
    plot_roc_curves(fitted_models, X_test_scaled, y_test, figures_dir / "roc_curves.png")
    plot_precision_recall_curves(fitted_models, X_test_scaled, y_test, figures_dir / "precision_recall_curves.png")
    brier_scores = plot_calibration_curves(fitted_models, X_test_scaled, y_test, figures_dir / "calibration_curves.png")
    pd.Series(brier_scores, name="brier_score").to_csv(processed_dir / "brier_scores.csv")

    print("Running robustness test...")
    robustness_df = xgboost_training_size_robustness(
        X_train_scaled, X_test_scaled, y_train, y_test
    )
    robustness_df.to_csv(processed_dir / "robustness_results.csv", index=False)
    plot_robustness_results(robustness_df, figures_dir / "robustness_xgboost.png")

    print("Done.")


if __name__ == "__main__":
    main()
