from financial_tweets_sentiment_analysis import predict


def test_model_predicts_single_tweet(predictor_run_id):
    results = predict.predict_texts(predictor_run_id, ["$AAPL looks strong into earnings"])
    assert len(results) == 1
    assert results[0]["prediction"] in {"bullish", "neutral", "bearish"}
    assert set(results[0]["probabilities"]) == {"bearish", "bullish", "neutral"}
