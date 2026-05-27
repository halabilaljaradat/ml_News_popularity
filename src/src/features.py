import numpy as np
import pandas as pd

ENGINEERED_FEATURES = [
    "title_word_ratio",
    "img_wrd_ratio",
    "keyword_richness",
    "content_complexity",
    "sentiment_score",
    "self_refer_ratio",
    "log_title_length",
    "total_media",
]

SKEWED_COLUMNS = [
    "img_wrd_ratio",
    "n_unique_tokens",
    "n_non_stop_unique_tokens",
    "content_complexity",
    "title_word_ratio",
    "kw_max_min",
    "keyword_richness",
    "self_reference_min_shares",
    "self_reference_avg_sharess",
    "kw_avg_min",
]


def add_category_label(df: pd.DataFrame) -> pd.DataFrame:
    """Create a readable category label for EDA only."""
    df = df.copy()
    df["category"] = "other"
    mapping = {
        "data_channel_is_lifestyle": "lifestyle",
        "data_channel_is_entertainment": "entertainment",
        "data_channel_is_bus": "business",
        "data_channel_is_socmed": "social media",
        "data_channel_is_tech": "tech",
        "data_channel_is_world": "world",
    }
    for col, label in mapping.items():
        if col in df.columns:
            df.loc[df[col] == 1, "category"] = label
    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features used in the classification notebook."""
    df = df.copy()

    if {"n_tokens_title", "n_tokens_content"}.issubset(df.columns):
        df["title_word_ratio"] = df["n_tokens_title"] / (df["n_tokens_content"] + 1)

    if {"num_imgs", "n_tokens_content"}.issubset(df.columns):
        df["img_wrd_ratio"] = df["num_imgs"] / (df["n_tokens_content"] + 1)

    if {"num_keywords", "n_tokens_content"}.issubset(df.columns):
        df["keyword_richness"] = df["num_keywords"] / (df["n_tokens_content"] + 1)

    if {"n_unique_tokens", "average_token_length"}.issubset(df.columns):
        df["content_complexity"] = df["n_unique_tokens"] * df["average_token_length"]

    if {"global_sentiment_polarity", "global_subjectivity"}.issubset(df.columns):
        df["sentiment_score"] = df["global_sentiment_polarity"] * df["global_subjectivity"]

    if {"num_self_hrefs", "num_hrefs"}.issubset(df.columns):
        df["self_refer_ratio"] = df["num_self_hrefs"] / (df["num_hrefs"] + 1)

    if "n_tokens_title" in df.columns:
        df["log_title_length"] = np.log1p(df["n_tokens_title"].clip(lower=0))

    if {"num_imgs", "num_videos"}.issubset(df.columns):
        df["total_media"] = df["num_imgs"] + df["num_videos"]

    return df


def log_transform_skewed_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    skewed_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply log1p transformation to selected non-negative skewed columns after split."""
    if skewed_cols is None:
        skewed_cols = SKEWED_COLUMNS

    X_train = X_train.copy()
    X_test = X_test.copy()

    for col in skewed_cols:
        if col in X_train.columns:
            X_train[col] = np.log1p(X_train[col].clip(lower=0))
            X_test[col] = np.log1p(X_test[col].clip(lower=0))

    return X_train, X_test
