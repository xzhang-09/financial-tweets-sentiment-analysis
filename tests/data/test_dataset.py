from financial_tweets_sentiment_analysis import data


def test_dataset_schema(dataset_loc):
    df = data.load_data(dataset_loc, num_samples=100)
    data.validate_dataframe(df)
    assert set(df["sentiment"].unique()).issubset({"bullish", "neutral", "bearish"})
    assert not df["id"].duplicated().any()
