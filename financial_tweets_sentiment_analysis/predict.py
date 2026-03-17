from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
import typer

from financial_tweets_sentiment_analysis import data, utils
from financial_tweets_sentiment_analysis.config import MODEL_REGISTRY, logger
from financial_tweets_sentiment_analysis.models import load_model

app = typer.Typer()


def decode(indices: Iterable[int], index_to_class: Dict[int, str]) -> List[str]:
    """Decode integer predictions into label strings."""
    return [index_to_class[index] for index in indices]


def format_prob(prob: Iterable[float], index_to_class: Dict[int, str]) -> Dict[str, float]:
    """Format a probability vector as a label-to-probability mapping."""
    return {index_to_class[index]: float(value) for index, value in enumerate(prob)}


def get_artifact_dir(run_id: str) -> Path:
    """Resolve the artifact directory for a saved run."""
    artifact_dir = MODEL_REGISTRY / run_id
    if not artifact_dir.exists():
        raise FileNotFoundError(f"Run artifact not found: {artifact_dir}")
    return artifact_dir


def get_best_run_id(metric: str = "macro_f1", mode: str = "max") -> str:
    """Return the best run ID in the local registry for a metric and mode."""
    summaries = []
    for summary_path in MODEL_REGISTRY.glob("*/run_summary.json"):
        summary = utils.load_dict(summary_path)
        summaries.append(summary)
    if not summaries:
        raise FileNotFoundError("No trained runs found in the model registry.")
    reverse = mode.lower() == "max"
    summaries.sort(key=lambda item: item["metrics"][metric], reverse=reverse)
    return summaries[0]["run_id"]


def predict_texts(run_id: str, texts: List[str]) -> List[Dict]:
    """Run inference for one or more raw tweet texts."""
    model = load_model(get_artifact_dir(run_id))
    cleaned_texts = [data.prepare_dataframe(pd.DataFrame({"tweet": [text], "sentiment": ["neutral"]}))["tweet_clean"].iloc[0] for text in texts]
    outputs = model.predict(cleaned_texts)
    results = []
    for text, prediction, probabilities in zip(texts, outputs.predictions, outputs.probabilities):
        results.append({"tweet": text, "prediction": prediction, "probabilities": probabilities})
    return results


@app.command("predict")
def predict_command(
    run_id: str = typer.Option(..., help="Saved run identifier to load."),
    tweet: str = typer.Option(..., help="Raw tweet text to score."),
) -> List[Dict]:
    results = predict_texts(run_id=run_id, texts=[tweet])
    logger.info(json.dumps(results, indent=2))
    return results


@app.command("predict-batch")
def predict_batch_command(
    run_id: str = typer.Option(..., help="Saved run identifier to load."),
    input_file: str = typer.Option(..., help="CSV or parquet file containing a `tweet` column."),
    output_file: Optional[str] = typer.Option(None, help="Optional JSON file for predictions."),
    file_type: Optional[str] = typer.Option(None, help="Input format override: csv|parquet."),
) -> List[Dict]:
    batch_df = data.read_dataset(input_file, file_type=file_type)
    if "tweet" not in batch_df.columns:
        raise ValueError("Batch input file must contain a 'tweet' column.")
    results = predict_texts(run_id=run_id, texts=batch_df["tweet"].astype(str).tolist())
    if output_file:
        utils.save_dict({"results": results}, output_file)
    return results


if __name__ == "__main__":
    app()
