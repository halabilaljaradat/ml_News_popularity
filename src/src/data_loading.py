from pathlib import Path
import pandas as pd


def load_online_news_data(csv_path: str | Path) -> pd.DataFrame:
    """Load the Online News Popularity CSV and strip whitespace from column names."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Could not find dataset at {csv_path}. "
            "Place OnlineNewsPopularity.csv in data/raw/."
        )
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    return df
