# Financial Tweets Sentiment Analysis

An end-to-end NLP and MLOps portfolio project for classifying financial tweets into `bullish`, `neutral`, and `bearish` sentiment.

This repository is designed to show more than model training accuracy. It demonstrates how to structure a real machine learning project around:
- dataset standardization and split discipline
- a strong baseline plus a transformer track
- reproducible training, evaluation, and inference interfaces
- slice-based analysis and qualitative error review
- lightweight serving for production-style usage

## Why This Project

Financial tweet sentiment is a useful but noisy classification problem:
- language is short, informal, and often ambiguous
- tweets mix opinion, headlines, rumors, and market jargon
- cashtags, URLs, and source variation introduce domain-specific signal
- bullish and bearish language can be subtle and easily confused

That makes the task a good showcase for both modeling and machine learning system design.

## What This Repository Demonstrates

This project is built to highlight skills that are useful in applied ML and MLOps roles:
- designing a consistent data schema from mixed sources
- separating raw, interim, and processed datasets
- building a baseline pipeline before fine-tuning larger models
- evaluating beyond a single score with per-class metrics, slices, and error analysis
- exposing trained models through clean prediction and API interfaces

## Modeling Strategy

The repository uses a two-track modeling strategy:

1. Baseline
   `tweet_clean -> TF-IDF -> linear classifier`

2. Transformer
   `tweet_clean -> FinBERT / transformer fine-tuning`

This setup makes it easier to answer practical questions such as:
- How much does a transformer improve over a strong classical baseline?
- Which kinds of tweets remain difficult even after fine-tuning?
- Are gains consistent across different sources and slices?

## Project Structure

```text
financial_tweets_sentiment_analysis/
  config.py
  data.py
  evaluate.py
  features.py
  models.py
  predict.py
  serve.py
  train.py
  tune.py
datasets/
  raw/
  interim/
  processed/
    train/
      train_full.csv
      train_sample_5000.csv
    test/
      holdout_full.csv
      holdout_sample_1000.csv
reports/
  figures/
  metrics/
  error_analysis/
artifacts/
  model_registry/
```

## Data Design

The project uses a unified schema so all stages of the pipeline can share the same assumptions:
- `id`
- `tweet`
- `tweet_raw`
- `tweet_clean`
- `sentiment`
- `source`
- `created_at`
- `ticker_mentions`
- `has_url`
- `split`

Dataset layout:

```text
datasets/
  raw/
  interim/
  processed/
    train/
      train_full.csv
      train_sample_5000.csv
    test/
      holdout_full.csv
      holdout_sample_1000.csv
```

Use cases for the current files:
- `train_full.csv`: larger training set for regular experiments
- `train_sample_5000.csv`: faster iteration for notebooks, debugging, and tests
- `holdout_full.csv`: main holdout set for final evaluation
- `holdout_sample_1000.csv`: smaller holdout set for quick validation

More detail is available in [datasets/README.md]

## Evaluation Philosophy

The project is intentionally not framed around a single headline metric.

Evaluation includes:
- macro and weighted F1
- per-class precision, recall, and F1
- confusion matrix output
- slice metrics for short text, ticker presence, URLs, headlines, and source groups
- error analysis artifacts for high-confidence mistakes and bullish/bearish confusions

This is the part of the project that is usually most valuable in an interview conversation because it shows how model behavior is understood, not just measured.

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -r requirements.txt
export PYTHONPATH=$PWD
```

## Training

Train the baseline model:

```bash
python -m financial_tweets_sentiment_analysis.train \
  --model-type baseline \
  --dataset-loc datasets/processed/train/train_sample_5000.csv \
  --num-samples 2000 \
  --run-name baseline-v1
```

Train the transformer model:

```bash
python -m financial_tweets_sentiment_analysis.train \
  --model-type transformer \
  --dataset-loc datasets/processed/train/train_sample_5000.csv \
  --num-samples 1000 \
  --num-epochs 1 \
  --batch-size 8 \
  --transformer-model-name ProsusAI/finbert \
  --run-name finbert-v1
```

## Prediction

Single-text prediction:

```bash
python -m financial_tweets_sentiment_analysis.predict \
  --run-id baseline-v1 \
  --tweet '$AAPL looks strong into earnings'
```

Batch prediction:

```bash
python -m financial_tweets_sentiment_analysis.predict predict-batch \
  --run-id baseline-v1 \
  --input-file datasets/processed/test/holdout_sample_1000.csv \
  --output-file reports/metrics/batch_predictions.json
```

## Evaluation

```bash
python -m financial_tweets_sentiment_analysis.evaluate \
  --run-id baseline-v1 \
  --dataset-loc datasets/processed/test/holdout_sample_1000.csv \
  --results-fp reports/metrics/baseline_eval.json
```

Evaluation outputs include:
- overall metrics
- per-class metrics
- confusion matrix
- slice metrics
- error analysis

## Serving

Start the API locally:

```bash
python -m financial_tweets_sentiment_analysis.serve --run-id baseline-v1
```

Available endpoints:
- `GET /`
- `POST /predict`
- `POST /predict-batch`
- `POST /evaluate`

## Testing

Run the core checks:

```bash
pytest tests/code/test_data.py tests/code/test_train.py tests/code/test_predict.py -q
```
