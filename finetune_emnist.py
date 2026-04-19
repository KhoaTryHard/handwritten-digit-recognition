# Module nay chay stage 2 de fine-tune model MNIST tren EMNIST.
"""Fine-tune the base classifier on EMNIST digits."""

from __future__ import annotations

from digit_pipeline.config import project_file
from digit_pipeline.training import EmnistTrainingConfig, train_emnist_stage


CONFIG = EmnistTrainingConfig(
    input_model_path=project_file("models", "stage_01_mnist_base.keras"),
    output_model_path=project_file("models", "stage_02_emnist_finetuned.keras"),
    batch_size=128,
    epochs=5,
    learning_rate=1e-4,
    rotation=0.10,
    translation=0.15,
    zoom=0.12,
)


def validate_dependencies() -> None:
    """Ensure optional TFDS resources are available before training."""
    try:
        import importlib.resources  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    try:
        import importlib_resources  # noqa: F401
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing importlib resource support. "
            "Run 'pip install -r requirements.txt' and try again."
        ) from exc


def main() -> None:
    """Run stage 2 fine-tuning on EMNIST."""
    validate_dependencies()
    result = train_emnist_stage(CONFIG)
    if result.validation_accuracy is not None:
        print(f"Validation accuracy: {result.validation_accuracy * 100:.2f}%")
    print(f"Classes: {list(result.class_names)}")
    print(f"Saved: {result.output_model_path}")


if __name__ == "__main__":
    main()
