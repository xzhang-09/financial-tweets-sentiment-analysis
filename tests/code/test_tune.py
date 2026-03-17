from financial_tweets_sentiment_analysis import tune


def test_tune_models(dataset_loc):
    results = tune.tune_models(
        model_type="baseline",
        dataset_loc=dataset_loc,
        search_space={"learning_rate": [2e-5], "batch_size": [8]},
        num_samples=120,
        max_runs=1,
    )
    assert len(results["results"]) == 1
