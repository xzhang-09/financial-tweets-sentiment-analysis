import pandas as pd

from financial_tweets_sentiment_analysis import data


def test_load_data(dataset_loc):
    df = data.load_data(dataset_loc, num_samples=10)
    assert len(df) == 10
    assert {"tweet", "tweet_clean", "sentiment", "ticker_mentions", "has_url"}.issubset(df.columns)


def test_split_dataframe_stratified():
    df = pd.DataFrame(
        {
            "tweet": [f"tweet {index}" for index in range(30)],
            "sentiment": ["bullish"] * 10 + ["neutral"] * 10 + ["bearish"] * 10,
        }
    )
    prepared = data.prepare_dataframe(df)
    train_df, val_df, test_df = data.split_dataframe(prepared)
    assert len(train_df) + len(val_df) + len(test_df) == len(prepared)
    assert set(train_df["sentiment"].unique()) == {"bullish", "neutral", "bearish"}


def test_prepare_dataframe_standardizes_schema():
    raw = pd.DataFrame({"text": ["$AAPL looks strong"], "label": ["positive"], "url": ["kaggle"]})
    prepared = data.prepare_dataframe(raw)
    assert prepared.loc[0, "sentiment"] == "bullish"
    assert prepared.loc[0, "ticker_mentions"] == ["$AAPL"]
