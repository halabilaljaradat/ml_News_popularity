from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import pandas as pd


@dataclass
class PreparedClassificationData:
    X_train_raw: pd.DataFrame
    X_test_raw: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    X_train_scaled: pd.DataFrame
    X_test_scaled: pd.DataFrame
    selected_features: list[str]
    scaler: RobustScaler
    threshold: float


def define_classification_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, float]:
    """Define X and binary popularity target using median shares."""
    threshold = float(df["shares"].median())
    y = (df["shares"] > threshold).astype(int)
    X = df.drop(columns=["shares", "log_shares", "category"], errors="ignore")
    return X, y, threshold


def split_classification_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Split before any scaling or feature selection to prevent leakage."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def scale_selected_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    selected_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, RobustScaler]:
    """Scale selected features using RobustScaler fitted only on training data."""
    scaler = RobustScaler()
    X_train_selected = X_train[selected_features].copy()
    X_test_selected = X_test[selected_features].copy()

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_selected),
        columns=selected_features,
        index=X_train_selected.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test_selected),
        columns=selected_features,
        index=X_test_selected.index,
    )
    return X_train_scaled, X_test_scaled, scaler
