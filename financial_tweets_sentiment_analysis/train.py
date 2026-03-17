from __future__ import annotations

import datetime
import json
import uuid
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import typer
from sklearn.metrics import f1_score
try:
    import torch
    from torch.optim import AdamW
    from torch.utils.data import DataLoader, Dataset
except ModuleNotFoundError:  # pragma: no cover - optional for baseline-only environments
    torch = None
    AdamW = None

    class Dataset:  # type: ignore[override]
        pass

    DataLoader = None

from financial_tweets_sentiment_analysis import data, utils
from financial_tweets_sentiment_analysis.config import DEFAULT_TRANSFORMER_MODEL, MODEL_REGISTRY, RANDOM_STATE, logger, mlflow
from financial_tweets_sentiment_analysis.models import BaselineSentimentModel, TransformerSentimentModel

app = typer.Typer()


class TextDataset(Dataset):
    """Tokenized dataset wrapper used for transformer fine-tuning."""

    def __init__(self, texts, labels, tokenizer, label_to_index):
        self.texts = list(texts)
        self.labels = [label_to_index[label] for label in labels]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        encoded = self.tokenizer(self.texts[index], truncation=True, padding="max_length", max_length=128, return_tensors="pt")
        item = {key: value.squeeze(0) for key, value in encoded.items()}
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
        return item


def _classification_metrics(y_true, y_pred) -> Dict[str, float]:
    """Compute the core validation metrics used across model types."""
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
    }


def _artifact_dir(run_id: str) -> Path:
    """Create or resolve the artifact directory for a training run."""
    artifact_dir = MODEL_REGISTRY / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir


def train_baseline(train_df, val_df) -> Dict:
    """Train the linear baseline on cleaned tweet text."""
    model = BaselineSentimentModel()
    model.fit(train_df["tweet_clean"], train_df["sentiment"])
    val_output = model.predict(val_df["tweet_clean"])
    metrics = _classification_metrics(val_df["sentiment"], val_output.predictions)
    return {"model": model, "metrics": metrics}


def train_transformer(train_df, val_df, model_name: str, num_epochs: int, batch_size: int, learning_rate: float) -> Dict:
    """Fine-tune a transformer classifier on the tweet sentiment task."""
    if torch is None or DataLoader is None or AdamW is None:
        raise ModuleNotFoundError("torch is required for transformer training.")
    model = TransformerSentimentModel(model_name=model_name)
    train_dataset = TextDataset(train_df["tweet_clean"], train_df["sentiment"], model.tokenizer, model.label_to_index)
    val_dataset = TextDataset(val_df["tweet_clean"], val_df["sentiment"], model.tokenizer, model.label_to_index)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.model.to(device)
    optimizer = AdamW(model.model.parameters(), lr=learning_rate)
    y_train = np.array([model.label_to_index[label] for label in train_df["sentiment"]])
    class_counts = np.bincount(y_train, minlength=len(model.label_to_index))
    class_weights = len(y_train) / (len(class_counts) * np.maximum(class_counts, 1))
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))

    for _ in range(num_epochs):
        model.model.train()
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad()
            outputs = model.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            loss = loss_fn(outputs.logits, batch["labels"])
            loss.backward()
            optimizer.step()

    model.model.eval()
    predictions = []
    truth = []
    with torch.inference_mode():
        for batch in val_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            predictions.extend(outputs.logits.argmax(dim=1).cpu().tolist())
            truth.extend(batch["labels"].cpu().tolist())
    metrics = _classification_metrics(truth, predictions)
    return {"model": model, "metrics": metrics}


def train_model(
    model_type: str,
    dataset_loc: str,
    file_type: Optional[str] = None,
    num_samples: Optional[int] = None,
    num_epochs: int = 1,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    transformer_model_name: str = DEFAULT_TRANSFORMER_MODEL,
    experiment_name: str = "financial-tweets-sentiment",
    run_name: Optional[str] = None,
    results_fp: Optional[str] = None,
) -> Dict:
    """Train either the baseline or transformer pipeline and save run artifacts."""
    utils.set_seeds(RANDOM_STATE)
    mlflow.set_experiment(experiment_name)
    splits = data.create_dataset_splits(path=dataset_loc, file_type=file_type, num_samples=num_samples)
    train_df, val_df, test_df = splits["train"], splits["val"], splits["test"]

    if model_type == "baseline":
        result = train_baseline(train_df=train_df, val_df=val_df)
    elif model_type == "transformer":
        result = train_transformer(
            train_df=train_df,
            val_df=val_df,
            model_name=transformer_model_name,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )
    else:
        raise ValueError("model_type must be one of: baseline, transformer")

    run_id = run_name or f"{model_type}-{uuid.uuid4().hex[:8]}"
    artifact_dir = _artifact_dir(run_id)
    result["model"].save(artifact_dir)
    data.save_split_datasets(splits, artifact_dir / "splits")

    metadata = {
        "run_id": run_id,
        "experiment_name": experiment_name,
        "model_type": model_type,
        "dataset_loc": dataset_loc,
        "created_at": datetime.datetime.now().isoformat(),
        "metrics": result["metrics"],
        "num_samples": num_samples,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "transformer_model_name": transformer_model_name if model_type == "transformer" else None,
        "test_size": len(test_df),
    }
    utils.save_dict(metadata, artifact_dir / "run_summary.json")

    with mlflow.start_run(run_name=run_id, nested=False):
        mlflow.set_tag("model_type", model_type)
        mlflow.set_tag("dataset_loc", dataset_loc)
        mlflow.log_params(
            {
                "num_samples": num_samples or len(train_df) + len(val_df) + len(test_df),
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
            }
        )
        mlflow.log_metrics(result["metrics"])
        mlflow.log_artifact(str(artifact_dir / "run_summary.json"))

    if results_fp:
        utils.save_dict(metadata, results_fp)
    logger.info(json.dumps(metadata, indent=2))
    return metadata


@app.command("train")
def train_command(
    model_type: str = typer.Option("baseline", help="Training pipeline to run: baseline|transformer."),
    dataset_loc: str = typer.Option(..., help="Input dataset location."),
    file_type: Optional[str] = typer.Option(None, help="Input format override: csv|parquet."),
    num_samples: Optional[int] = typer.Option(None, help="Optional row limit for fast experiments."),
    num_epochs: int = typer.Option(1, help="Number of epochs for transformer training."),
    batch_size: int = typer.Option(8, help="Mini-batch size for transformer training."),
    learning_rate: float = typer.Option(2e-5, help="Learning rate for transformer training."),
    transformer_model_name: str = typer.Option(DEFAULT_TRANSFORMER_MODEL, help="Hugging Face model name."),
    experiment_name: str = typer.Option("financial-tweets-sentiment", help="Experiment name for tracking."),
    run_name: Optional[str] = typer.Option(None, help="Optional custom run identifier."),
    results_fp: Optional[str] = typer.Option(None, help="Optional JSON file for the training summary."),
):
    train_model(
        model_type=model_type,
        dataset_loc=dataset_loc,
        file_type=file_type,
        num_samples=num_samples,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        transformer_model_name=transformer_model_name,
        experiment_name=experiment_name,
        run_name=run_name,
        results_fp=results_fp,
    )


if __name__ == "__main__":
    app()
