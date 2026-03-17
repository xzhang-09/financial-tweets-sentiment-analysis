import pytest

from financial_tweets_sentiment_analysis import data


@pytest.fixture(scope="module")
def dataset_loc():
    return "datasets/processed/train/train_sample_5000.csv"


@pytest.fixture(scope="module")
def df(dataset_loc):
    return data.load_data(dataset_loc, num_samples=100)
