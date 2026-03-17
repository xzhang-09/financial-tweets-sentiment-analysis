import tempfile
from pathlib import Path

import numpy as np

from financial_tweets_sentiment_analysis import utils


def test_set_seed():
    utils.set_seeds()
    a = np.random.randn(2, 3)
    utils.set_seeds()
    b = np.random.randn(2, 3)
    assert np.array_equal(a, b)


def test_save_and_load_dict():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "data.json"
        utils.save_dict({"hello": "world"}, path)
        assert utils.load_dict(path)["hello"] == "world"


def test_dict_to_list():
    data = {"a": [1, 2], "b": [3, 4]}
    assert utils.dict_to_list(data, keys=["a", "b"]) == [{"a": 1, "b": 3}, {"a": 2, "b": 4}]
