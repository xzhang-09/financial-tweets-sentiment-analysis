from financial_tweets_sentiment_analysis import predict


def get_label(text, run_id):
    results = predict.predict_texts(run_id, [text])
    return results[0]["prediction"]
