from collections import Counter
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression


def correlation_selected_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    threshold: float = 0.03,
) -> list[str]:
    """Select features based on absolute correlation with the training target."""
    train_for_corr = X_train.copy()
    train_for_corr["target"] = y_train.values
    corr = train_for_corr.corr(numeric_only=True)["target"].abs().drop("target")
    return corr[corr >= threshold].index.tolist()


def tree_importance_selected_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    threshold: float = 0.005,
    random_state: int = 42,
) -> list[str]:
    """Select features using random-forest feature importance on training data only."""
    rf_selector = RandomForestClassifier(
        n_estimators=100,
        random_state=random_state,
        max_depth=10,
        n_jobs=-1,
    )
    rf_selector.fit(X_train, y_train)
    importance = pd.Series(rf_selector.feature_importances_, index=X_train.columns)
    return importance[importance >= threshold].sort_values(ascending=False).index.tolist()


def rfe_selected_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_features_to_select: int = 20,
    random_state: int = 42,
) -> list[str]:
    """Select features using RFE with logistic regression on training data only."""
    n_features_to_select = min(n_features_to_select, X_train.shape[1])
    rfe_model = LogisticRegression(max_iter=1000, random_state=random_state)
    rfe = RFE(estimator=rfe_model, n_features_to_select=n_features_to_select)
    rfe.fit(X_train, y_train)
    return X_train.columns[rfe.support_].tolist()


def voting_selected_features(
    feature_lists: list[list[str]],
    min_votes: int = 2,
) -> list[str]:
    """Keep features selected by at least min_votes selection methods."""
    counts = Counter(feature for features in feature_lists for feature in features)
    selected = [feature for feature, count in counts.items() if count >= min_votes]
    if not selected:
        # Conservative fallback: use all features selected by any method if voting is too strict.
        selected = list(counts.keys())
    return selected


def select_features_by_three_methods(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> dict:
    """Run correlation, tree-importance, and RFE selection, then combine by voting."""
    corr_features = correlation_selected_features(X_train, y_train)
    importance_features = tree_importance_selected_features(X_train, y_train)
    rfe_features = rfe_selected_features(X_train, y_train)
    final_features = voting_selected_features(
        [corr_features, importance_features, rfe_features], min_votes=2
    )
    return {
        "corr_features": corr_features,
        "importance_features": importance_features,
        "rfe_features": rfe_features,
        "final_features": final_features,
    }
