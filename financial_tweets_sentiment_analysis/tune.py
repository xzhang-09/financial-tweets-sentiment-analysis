from __future__ import annotations

import itertools
import json
from typing import Dict, List, Optional

import typer

from financial_tweets_sentiment_analysis import train, utils
from financial_tweets_sentiment_analysis.config import logger

app = typer.Typer()


def tune_models(
    model_type: str,
    dataset_loc: str,
    search_space: Optional[Dict[str, List]] = None,
    num_samples: Optional[int] = None,
    max_runs: int = 4,
    transformer_model_name: Optional[str] = None,
) -> Dict:
    """Run a lightweight local parameter sweep over the chosen training pipeline."""
    search_space = search_space or {"learning_rate": [2e-5, 5e-5], "batch_size": [8]}
    keys = list(search_space)
    values = [search_space[key] for key in keys]
    results = []
    for index, combo in enumerate(itertools.product(*values)):
        if index >= max_runs:
            break
        params = dict(zip(keys, combo))
        run_summary = train.train_model(
            model_type=model_type,
            dataset_loc=dataset_loc,
            num_samples=num_samples,
            batch_size=params.get("batch_size", 8),
            learning_rate=params.get("learning_rate", 2e-5),
            num_epochs=params.get("num_epochs", 1),
            transformer_model_name=transformer_model_name or train.DEFAULT_TRANSFORMER_MODEL,
            run_name=f"tune-{model_type}-{index}",
        )
        results.append({"params": params, "metrics": run_summary["metrics"], "run_id": run_summary["run_id"]})
    best_result = max(results, key=lambda item: item["metrics"]["macro_f1"])
    payload = {"results": results, "best_result": best_result}
    logger.info(json.dumps(payload, indent=2))
    return payload


@app.command("tune")
def tune_command(
    model_type: str = typer.Option("baseline", help="Training pipeline to sweep: baseline|transformer."),
    dataset_loc: str = typer.Option(..., help="Input dataset location."),
    search_space: Optional[str] = typer.Option(None, help="JSON dictionary of candidate parameter lists."),
    num_samples: Optional[int] = typer.Option(None, help="Optional row limit for faster sweeps."),
    max_runs: int = typer.Option(4, help="Maximum number of runs to execute."),
    results_fp: Optional[str] = typer.Option(None, help="Optional JSON file for tuning results."),
):
    payload = tune_models(
        model_type=model_type,
        dataset_loc=dataset_loc,
        search_space=json.loads(search_space) if search_space else None,
        num_samples=num_samples,
        max_runs=max_runs,
    )
    if results_fp:
        utils.save_dict(payload, results_fp)


if __name__ == "__main__":
    app()
