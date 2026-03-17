from __future__ import annotations

from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from financial_tweets_sentiment_analysis import evaluate, predict

app = FastAPI(title="Financial Tweets Sentiment Analysis", version="0.1.0")


class PredictRequest(BaseModel):
    tweet: str


class BatchPredictRequest(BaseModel):
    tweets: List[str]


class EvaluateRequest(BaseModel):
    dataset_loc: str
    file_type: Optional[str] = None


_CURRENT_RUN_ID: Optional[str] = None


def configure(run_id: str) -> None:
    """Bind the API server to a saved run identifier."""
    global _CURRENT_RUN_ID
    _CURRENT_RUN_ID = run_id


@app.get("/")
def healthcheck() -> Dict[str, str]:
    return {"status": "ok", "run_id": _CURRENT_RUN_ID or ""}


@app.post("/predict")
def predict_endpoint(request: PredictRequest) -> Dict:
    if not _CURRENT_RUN_ID:
        raise RuntimeError("Server has not been configured with a run_id.")
    return {"results": predict.predict_texts(run_id=_CURRENT_RUN_ID, texts=[request.tweet])}


@app.post("/predict-batch")
def predict_batch_endpoint(request: BatchPredictRequest) -> Dict:
    if not _CURRENT_RUN_ID:
        raise RuntimeError("Server has not been configured with a run_id.")
    return {"results": predict.predict_texts(run_id=_CURRENT_RUN_ID, texts=request.tweets)}


@app.post("/evaluate")
def evaluate_endpoint(request: EvaluateRequest) -> Dict:
    if not _CURRENT_RUN_ID:
        raise RuntimeError("Server has not been configured with a run_id.")
    return evaluate.evaluate_run(run_id=_CURRENT_RUN_ID, dataset_loc=request.dataset_loc, file_type=request.file_type)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True, help="Saved run identifier to serve.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    args = parser.parse_args()
    configure(args.run_id)
    uvicorn.run(app, host=args.host, port=args.port)
