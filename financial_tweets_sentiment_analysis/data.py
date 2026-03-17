from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from financial_tweets_sentiment_analysis import features
from financial_tweets_sentiment_analysis.config import LABELS, LABEL_COLUMN, RANDOM_STATE, TEXT_COLUMN, logger

REQUIRED_COLUMNS = ["id", "tweet", "tweet_raw", "tweet_clean", "sentiment", "source", "created_at", "ticker_mentions", "has_url", "split"]
SUPPORTED_FILE_TYPES = {"csv", "parquet"}


def infer_file_type(path: str, file_type: Optional[str] = None) -> str:
    """Infer the dataset file type from an explicit hint or file suffix."""
    if file_type:
        normalized = file_type.lower()
    else:
        suffix = Path(path).suffix.lower().lstrip(".")
        normalized = suffix
    if normalized not in SUPPORTED_FILE_TYPES:
        raise ValueError(f"Unsupported file type: {normalized}")
    return normalized


def read_dataset(path: str, file_type: Optional[str] = None) -> pd.DataFrame:
    """Read a dataset from CSV or parquet into a pandas DataFrame."""
    file_type = infer_file_type(path, file_type=file_type)
    if file_type == "csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def standardize_labels(series: pd.Series) -> pd.Series:
    """Map raw sentiment labels into the canonical project label set."""
    label_map = {
        0: "bullish",
        1: "neutral",
        2: "bearish",
        "0": "bullish",
        "1": "neutral",
        "2": "bearish",
        "positive": "bullish",
        "negative": "bearish",
    }
    standardized = series.map(lambda value: label_map.get(value, str(value).strip().lower()))
    if not set(standardized.unique()).issubset(set(LABELS)):
        invalid = sorted(set(standardized.unique()) - set(LABELS))
        raise ValueError(f"Unexpected labels found: {invalid}")
    return standardized


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common alternative column names into the project schema."""
    rename_map = {}
    if "text" in df.columns and "tweet" not in df.columns:
        rename_map["text"] = "tweet"
    if "label" in df.columns and "sentiment" not in df.columns:
        rename_map["label"] = "sentiment"
    if "url" in df.columns and "source" not in df.columns:
        rename_map["url"] = "source"
    df = df.rename(columns=rename_map)
    if TEXT_COLUMN not in df.columns or LABEL_COLUMN not in df.columns:
        raise ValueError("Dataset must include 'tweet' and 'sentiment' columns.")
    return df


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply schema normalization, text cleaning, and derived feature creation."""
    df = _normalize_columns(df.copy())
    df = df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN]).reset_index(drop=True)
    df["tweet_raw"] = df[TEXT_COLUMN].astype(str)
    df["tweet"] = df["tweet_raw"]
    df["tweet_clean"] = df["tweet_raw"].map(features.clean_tweet_text)
    df["sentiment"] = standardize_labels(df["sentiment"])
    df["source"] = df["source"].fillna("").astype(str) if "source" in df.columns else ""
    df["created_at"] = df["created_at"].fillna("").astype(str) if "created_at" in df.columns else ""
    df["ticker_mentions"] = df["tweet_raw"].map(features.extract_ticker_mentions)
    df["has_url"] = df["tweet_raw"].map(features.has_url)
    df = df.drop_duplicates(subset=["tweet_clean", "sentiment"]).reset_index(drop=True)
    df["id"] = [f"tweet-{idx:06d}" for idx in range(len(df))]
    df["split"] = df["split"].fillna("").astype(str) if "split" in df.columns else ""
    return df[REQUIRED_COLUMNS]


def validate_dataframe(df: pd.DataFrame) -> None:
    """Validate required columns, label values, and unique identifiers."""
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    labels = set(df["sentiment"].unique())
    if not labels.issubset(set(LABELS)):
        raise ValueError(f"Unexpected sentiment labels: {sorted(labels)}")
    if df["id"].duplicated().any():
        raise ValueError("IDs must be unique.")


def load_data(path: str, file_type: Optional[str] = None, num_samples: Optional[int] = None) -> pd.DataFrame:
    """Load, clean, validate, and optionally sample a dataset."""
    df = read_dataset(path, file_type=file_type)
    df = prepare_dataframe(df)
    validate_dataframe(df)
    if num_samples:
        df = df.sample(min(num_samples, len(df)), random_state=RANDOM_STATE).reset_index(drop=True)
    logger.info(f"Loaded {len(df)} rows from {path}")
    return df


def split_dataframe(
    df: pd.DataFrame,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create stratified train, validation, and test splits."""
    if round(train_size + val_size + test_size, 5) != 1.0:
        raise ValueError("train_size + val_size + test_size must equal 1.0")
    train_df, temp_df = train_test_split(
        df,
        test_size=(1 - train_size),
        stratify=df["sentiment"],
        random_state=random_state,
    )
    relative_test_size = test_size / (test_size + val_size)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_size,
        stratify=temp_df["sentiment"],
        random_state=random_state,
    )
    split_frames = {"train": train_df.copy(), "val": val_df.copy(), "test": test_df.copy()}
    for split_name, split_df in split_frames.items():
        split_df["split"] = split_name
    return split_frames["train"].reset_index(drop=True), split_frames["val"].reset_index(drop=True), split_frames["test"].reset_index(drop=True)


def create_dataset_splits(path: str, file_type: Optional[str] = None, num_samples: Optional[int] = None) -> Dict[str, pd.DataFrame]:
    """Load a dataset and return named train/validation/test splits."""
    df = load_data(path=path, file_type=file_type, num_samples=num_samples)
    train_df, val_df, test_df = split_dataframe(df)
    return {"train": train_df, "val": val_df, "test": test_df}


def save_split_datasets(splits: Dict[str, pd.DataFrame], output_dir: Path) -> Dict[str, Path]:
    """Persist split datasets to disk and return their output paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = {}
    for split_name, split_df in splits.items():
        path = output_dir / f"{split_name}.csv"
        split_df.to_csv(path, index=False)
        saved_paths[split_name] = path
    return saved_paths
