# Dataset Layout

This project uses a layered dataset structure so it is easier to tell which files are raw inputs, which files are cleaned working assets, and which files are ready for modeling.

## Directory Structure

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

## Folder Roles

- `raw/`
  Store untouched source data here. Files in this folder should preserve original columns and content from the upstream source whenever possible.

- `interim/`
  Store cleaned, standardized, merged, or deduplicated working datasets here before the final training/test split is created.

- `processed/train/`
  Store modeling-ready training datasets here.

- `processed/test/`
  Store modeling-ready holdout or evaluation datasets here.

## Current Files

- `processed/train/train_full.csv`
  The larger training dataset intended for regular experiments and model development.

- `processed/train/train_sample_5000.csv`
  A smaller training sample for fast experiments, debugging, tests, and notebook demos.

- `processed/test/holdout_full.csv`
  The main holdout dataset intended for final evaluation.

- `processed/test/holdout_sample_1000.csv`
  A smaller holdout sample for quick evaluation runs and demos.

## Recommended Usage

- Use `train_sample_5000.csv` when iterating quickly in notebooks or tests.
- Use `train_full.csv` for more realistic training runs.
- Use `holdout_sample_1000.csv` for quick evaluation checks.
- Use `holdout_full.csv` for final reporting.

## Notes

- Keep the holdout files separate from training workflows to avoid accidental leakage.
- Prefer adding new source files to `raw/` first, then creating cleaned assets in `interim/`, and only then writing finalized model-ready files into `processed/`.
