import json
from pathlib import Path

import pytest

from financial_tweets_sentiment_analysis import predict, train


@pytest.fixture(scope="module")
def run_id():
    return "test-model-behavior"


@pytest.fixture(scope="module")
def dataset_loc():
    return "datasets/processed/train/train_sample_5000.csv"


@pytest.fixture(scope="module")
def predictor_run_id(dataset_loc, run_id):
    artifact_dir = Path("artifacts/model_registry") / run_id
    if not artifact_dir.exists():
        train.train_model(model_type="baseline", dataset_loc=dataset_loc, num_samples=120, run_name=run_id)
    return run_id
