from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional until transformer training is used
    torch = None
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except ModuleNotFoundError:  # pragma: no cover - optional until transformer training is used
    AutoModelForSequenceClassification = None
    AutoTokenizer = None

from financial_tweets_sentiment_analysis.config import LABELS


def get_label_mapping(labels: Sequence[str] = LABELS) -> Dict[str, int]:
    return {label: index for index, label in enumerate(labels)}


@dataclass
class PredictionOutput:
    predictions: List[str]
    probabilities: List[Dict[str, float]]


class BaselineSentimentModel:
    def __init__(self, max_features: int = 20000, ngram_range: tuple = (1, 2)):
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
        self.classifier = SGDClassifier(loss="log_loss", class_weight="balanced", random_state=42)
        self.label_to_index = get_label_mapping()
        self.index_to_label = {value: key for key, value in self.label_to_index.items()}

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> None:
        x = self.vectorizer.fit_transform(texts)
        y = np.array([self.label_to_index[label] for label in labels])
        self.classifier.fit(x, y)

    def predict(self, texts: Sequence[str]) -> PredictionOutput:
        x = self.vectorizer.transform(texts)
        pred_ids = self.classifier.predict(x)
        pred_probs = self.classifier.predict_proba(x)
        predictions = [self.index_to_label[pred_id] for pred_id in pred_ids]
        probabilities = [
            {self.index_to_label[index]: float(probability) for index, probability in enumerate(sample_probs)}
            for sample_probs in pred_probs
        ]
        return PredictionOutput(predictions=predictions, probabilities=probabilities)

    def save(self, artifact_dir: Path) -> None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        with open(artifact_dir / "baseline.pkl", "wb") as file_pointer:
            pickle.dump(self, file_pointer)
        with open(artifact_dir / "metadata.json", "w") as file_pointer:
            json.dump({"model_type": "baseline", "labels": list(self.label_to_index)}, file_pointer, indent=2)

    @classmethod
    def load(cls, artifact_dir: Path) -> "BaselineSentimentModel":
        with open(artifact_dir / "baseline.pkl", "rb") as file_pointer:
            return pickle.load(file_pointer)


class TransformerSentimentModel:
    def __init__(self, model_name: str, num_labels: int = 3):
        if torch is None or AutoTokenizer is None or AutoModelForSequenceClassification is None:
            raise ModuleNotFoundError("torch and transformers are required for transformer training and inference.")
        self.model_name = model_name
        self.label_to_index = get_label_mapping()
        self.index_to_label = {value: key for key, value in self.label_to_index.items()}
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label=self.index_to_label,
            label2id=self.label_to_index,
        )

    def predict(self, texts: Sequence[str], batch_size: int = 16) -> PredictionOutput:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()
        predictions: List[str] = []
        probabilities: List[Dict[str, float]] = []
        for start in range(0, len(texts), batch_size):
            batch_texts = list(texts[start : start + batch_size])
            encoded = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.inference_mode():
                outputs = self.model(**encoded)
                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
            pred_ids = probs.argmax(axis=1)
            predictions.extend(self.index_to_label[pred_id] for pred_id in pred_ids)
            probabilities.extend(
                [{self.index_to_label[index]: float(probability) for index, probability in enumerate(sample_probs)} for sample_probs in probs]
            )
        return PredictionOutput(predictions=predictions, probabilities=probabilities)

    def save(self, artifact_dir: Path) -> None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(artifact_dir / "transformer")
        self.tokenizer.save_pretrained(artifact_dir / "transformer")
        with open(artifact_dir / "metadata.json", "w") as file_pointer:
            json.dump(
                {"model_type": "transformer", "labels": list(self.label_to_index), "model_name": self.model_name},
                file_pointer,
                indent=2,
            )

    @classmethod
    def load(cls, artifact_dir: Path) -> "TransformerSentimentModel":
        if torch is None or AutoTokenizer is None or AutoModelForSequenceClassification is None:
            raise ModuleNotFoundError("torch and transformers are required for transformer inference.")
        metadata = json.loads((artifact_dir / "metadata.json").read_text())
        instance = cls.__new__(cls)
        instance.model_name = metadata["model_name"]
        instance.label_to_index = get_label_mapping(metadata["labels"])
        instance.index_to_label = {value: key for key, value in instance.label_to_index.items()}
        instance.tokenizer = AutoTokenizer.from_pretrained(artifact_dir / "transformer")
        instance.model = AutoModelForSequenceClassification.from_pretrained(artifact_dir / "transformer")
        return instance


def load_model(artifact_dir: Path):
    metadata = json.loads((artifact_dir / "metadata.json").read_text())
    if metadata["model_type"] == "baseline":
        return BaselineSentimentModel.load(artifact_dir)
    return TransformerSentimentModel.load(artifact_dir)
