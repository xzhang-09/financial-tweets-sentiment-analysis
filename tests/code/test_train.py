from pathlib import Path

from financial_tweets_sentiment_analysis import train


def test_train_baseline_model(dataset_loc):
    summary = train.train_model(
        model_type="baseline",
        dataset_loc=dataset_loc,
        num_samples=120,
        run_name="test-baseline-run",
    )
    assert summary["model_type"] == "baseline"
    assert "macro_f1" in summary["metrics"]
    assert Path("artifacts/model_registry/test-baseline-run/run_summary.json").exists()
