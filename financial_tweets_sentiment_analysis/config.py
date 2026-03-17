import logging
import logging.config
import os
from pathlib import Path

try:
    import mlflow
except ModuleNotFoundError:  # pragma: no cover - dependency fallback for lightweight environments
    class _DummyRun:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _DummyMlflow:
        def set_tracking_uri(self, uri):
            self.uri = uri

        def set_experiment(self, name):
            self.name = name

        def start_run(self, **kwargs):
            return _DummyRun()

        def set_tag(self, *args, **kwargs):
            return None

        def log_params(self, *args, **kwargs):
            return None

        def log_metrics(self, *args, **kwargs):
            return None

        def log_artifact(self, *args, **kwargs):
            return None

    mlflow = _DummyMlflow()

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = ROOT_DIR / "datasets"
RAW_DATA_DIR = DATASETS_DIR / "raw"
INTERIM_DATA_DIR = DATASETS_DIR / "interim"
PROCESSED_DATA_DIR = DATASETS_DIR / "processed"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"
ERROR_ANALYSIS_DIR = REPORTS_DIR / "error_analysis"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
LOGS_DIR = ROOT_DIR / "logs"

for directory in [
    RAW_DATA_DIR,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    FIGURES_DIR,
    METRICS_DIR,
    ERROR_ANALYSIS_DIR,
    ARTIFACTS_DIR,
    LOGS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

EFS_DIR = ARTIFACTS_DIR
MODEL_REGISTRY = ARTIFACTS_DIR / "model_registry"
MODEL_REGISTRY.mkdir(parents=True, exist_ok=True)
MLFLOW_TRACKING_URI = "file://" + str((ARTIFACTS_DIR / "mlruns").resolve())
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

LABELS = ("bearish", "bullish", "neutral")
TEXT_COLUMN = "tweet"
LABEL_COLUMN = "sentiment"
RANDOM_STATE = 42
DEFAULT_TRANSFORMER_MODEL = os.environ.get("TRANSFORMER_MODEL_NAME", "ProsusAI/finbert")
DEFAULT_BASELINE_MODEL = "logistic_regression"

logging_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "minimal": {"format": "%(message)s"},
        "detailed": {
            "format": "%(levelname)s %(asctime)s [%(name)s:%(filename)s:%(funcName)s:%(lineno)d]\n%(message)s\n"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "minimal",
            "level": "INFO",
        },
        "info": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": str(LOGS_DIR / "info.log"),
            "maxBytes": 10 * 1024 * 1024,
            "backupCount": 5,
            "formatter": "detailed",
            "level": "INFO",
        },
    },
    "root": {"handlers": ["console", "info"], "level": "INFO"},
}

logging.config.dictConfig(logging_config)
logger = logging.getLogger("financial_tweets_sentiment_analysis")
