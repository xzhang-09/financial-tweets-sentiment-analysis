from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import typer
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from financial_tweets_sentiment_analysis import data, features, predict, utils
from financial_tweets_sentiment_analysis.config import ERROR_ANALYSIS_DIR, LABELS, METRICS_DIR, logger

app = typer.Typer()


def get_overall_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
    """Compute macro and weighted classification metrics."""
    macro = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    return {
        "macro_precision": float(macro[0]),
        "macro_recall": float(macro[1]),
        "macro_f1": float(macro[2]),
        "weighted_precision": float(weighted[0]),
        "weighted_recall": float(weighted[1]),
        "weighted_f1": float(weighted[2]),
        "num_samples": float(len(y_true)),
    }


def get_per_class_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, Dict[str, float]]:
    """Compute precision, recall, F1, and support for each class."""
    metrics = precision_recall_fscore_support(y_true, y_pred, labels=list(LABELS), average=None, zero_division=0)
    return {
        label: {
            "precision": float(metrics[0][index]),
            "recall": float(metrics[1][index]),
            "f1": float(metrics[2][index]),
            "support": float(metrics[3][index]),
        }
        for index, label in enumerate(LABELS)
    }


def get_slice_metrics(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Compute metrics for predefined business-relevant data slices."""
    slices = {
        "short_text": df["tweet_clean"].map(features.is_short_text),
        "has_ticker": df["ticker_mentions"].map(bool),
        "has_url": df["has_url"],
        "news_headline": df["tweet_raw"].map(features.is_news_headline),
    }
    metrics = {}
    for slice_name, mask in slices.items():
        if mask.sum() == 0:
            continue
        metrics[slice_name] = get_overall_metrics(df.loc[mask, "sentiment"], df.loc[mask, "prediction"])
    for source, source_df in df.groupby("source"):
        if source:
            metrics[f"source::{source}"] = get_overall_metrics(source_df["sentiment"], source_df["prediction"])
    return metrics


def get_error_analysis(df: pd.DataFrame, top_k: int = 20) -> Dict[str, List[Dict]]:
    """Collect high-value misclassification slices for qualitative review."""
    mistakes = df[df["sentiment"] != df["prediction"]].copy()
    mistakes["confidence"] = mistakes["probabilities"].map(lambda item: max(item.values()))
    mistakes = mistakes.sort_values("confidence", ascending=False)
    high_confidence = mistakes.head(top_k)
    bullish_bearish = mistakes[
        mistakes["sentiment"].isin(["bullish", "bearish"]) & mistakes["prediction"].isin(["bullish", "bearish"])
    ].head(top_k)
    sarcasm_like = mistakes[mistakes["tweet_raw"].str.contains(r"[!?]|yeah right|sure", case=False, regex=True)].head(top_k)
    return {
        "high_confidence_errors": high_confidence[["tweet_raw", "sentiment", "prediction", "confidence"]].to_dict(orient="records"),
        "bullish_bearish_confusions": bullish_bearish[["tweet_raw", "sentiment", "prediction", "confidence"]].to_dict(orient="records"),
        "sarcasm_or_rumor_candidates": sarcasm_like[["tweet_raw", "sentiment", "prediction", "confidence"]].to_dict(orient="records"),
    }


def evaluate_run(run_id: str, dataset_loc: str, file_type: Optional[str] = None, results_fp: Optional[str] = None) -> Dict:
    """Evaluate a saved run against a labeled dataset and persist the report."""
    df = data.load_data(dataset_loc, file_type=file_type)
    predictions = predict.predict_texts(run_id, df["tweet_raw"].tolist())
    prediction_df = pd.DataFrame(predictions)
    df = df.merge(prediction_df, left_on="tweet_raw", right_on="tweet", how="left")
    df["tweet"] = df["tweet_raw"]

    report = {
        "run_id": run_id,
        "overall": get_overall_metrics(df["sentiment"].tolist(), df["prediction"].tolist()),
        "per_class": get_per_class_metrics(df["sentiment"].tolist(), df["prediction"].tolist()),
        "confusion_matrix": confusion_matrix(df["sentiment"], df["prediction"], labels=list(LABELS)).tolist(),
        "slices": get_slice_metrics(df),
        "error_analysis": get_error_analysis(df),
    }

    metrics_path = METRICS_DIR / f"{run_id}_evaluation.json"
    error_path = ERROR_ANALYSIS_DIR / f"{run_id}_errors.json"
    utils.save_dict(report, metrics_path)
    utils.save_dict(report["error_analysis"], error_path)
    if results_fp:
        utils.save_dict(report, results_fp)
    logger.info(json.dumps(report["overall"], indent=2))
    return report


@app.command("evaluate")
def evaluate_command(
    run_id: str = typer.Option(..., help="Saved run identifier to evaluate."),
    dataset_loc: str = typer.Option(..., help="Labeled evaluation dataset."),
    file_type: Optional[str] = typer.Option(None, help="Input format override: csv|parquet."),
    results_fp: Optional[str] = typer.Option(None, help="Optional JSON file for the evaluation report."),
) -> Dict:
    return evaluate_run(run_id=run_id, dataset_loc=dataset_loc, file_type=file_type, results_fp=results_fp)


if __name__ == "__main__":
    app()
