# Datasets

The project uses a layered dataset layout to separate raw inputs, intermediate cleaned assets, and model-ready files.

## Layout

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

## File Roles

- `train_full.csv`
  Full training set for standard experiments.

- `train_sample_5000.csv`
  Smaller training sample for quick iteration, tests, and notebooks.

- `holdout_full.csv`
  Full holdout set for final evaluation.

- `holdout_sample_1000.csv`
  Smaller holdout sample for fast validation and demos.

## Workflow Guidance

1. Put source files in `datasets/raw/`.
2. Save cleaned or merged working data in `datasets/interim/`.
3. Save final train/test assets in `datasets/processed/`.

This structure helps keep experiments reproducible and reduces the chance of mixing raw data with model-ready assets.
