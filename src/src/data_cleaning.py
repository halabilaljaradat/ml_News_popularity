import numpy as np
import pandas as pd


def drop_non_predictive_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that are not useful or available for prediction."""
    return df.drop(columns=["url", "timedelta"], errors="ignore")


def add_log_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add log-transformed shares for EDA and future regression work."""
    df = df.copy()
    df["log_shares"] = np.log1p(df["shares"])
    return df


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Apply basic project cleaning steps."""
    df = df.copy()
    df.columns = df.columns.str.strip()
    df = drop_non_predictive_columns(df)
    df = add_log_target(df)
    return df


def data_quality_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a compact data-quality summary for the report."""
    return pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing_count": df.isna().sum(),
        "missing_percent": df.isna().mean() * 100,
        "n_unique": df.nunique(dropna=True),
    })


def duplicate_summary(df: pd.DataFrame) -> dict:
    """Return duplicate-row count and percentage."""
    duplicate_count = int(df.duplicated().sum())
    return {
        "duplicate_count": duplicate_count,
        "duplicate_percent": duplicate_count / len(df) * 100,
    }
