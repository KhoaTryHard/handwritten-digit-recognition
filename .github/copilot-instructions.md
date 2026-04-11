# Project Guidelines

## Code Style
- Prefer `project_paths.project_file()` and `project_paths.project_path()` for repository-relative paths; do not hardcode absolute paths.
- Keep root-level scripts thin. Put reusable logic in `digit_pipeline/` and compose it from the entrypoints.
- Use module-level constants at the top of scripts for configurable values such as batch size, epochs, paths, and thresholds.
- In scripts that import TensorFlow, set `os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"` before importing TensorFlow.
- Follow the existing 28x28 grayscale image flow and normalized float32 preprocessing in `digit_pipeline/data.py` and `digit_pipeline/preprocessing.py`.

## Architecture
- `digit_pipeline/` contains shared data loading, preprocessing, model, training, and evaluation helpers.
- Root scripts implement the stage workflow and analysis entrypoints described in [README.md](../README.md).
- `models/` stores saved `.keras` artifacts, and `reports/` stores exported CSVs and confusion-pair images.

## Build and Test
- There is no separate build step.
- Run the relevant script from the repository root to validate a change, following the workflow in [README.md](../README.md).
- For finetuning changes, use the stage scripts in order: `train_mnist_base.py`, `finetune_emnist.py`, `split_personal_data.py`, `convert_personal_data_to_28.py`, `finetune_personal_data.py`, `finetune_legacy_data.py`.

## Conventions
- Dataset directories must be organized by class name: `0/` through `9/`.
- Validation data may be loaded from a separate `val/` directory; if it is empty, the code falls back to an automatic train/validation split with a fixed seed.
- Model output classes must match the dataset class count.
- If a path, artifact name, or workflow step changes, update the README rather than duplicating the full explanation here.
