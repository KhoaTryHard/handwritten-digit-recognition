# Module nay chay stage 1 de train CNN nen tren MNIST.
"""Train the base MNIST digit classifier."""

from __future__ import annotations

from digit_pipeline.config import project_file
from digit_pipeline.training import MnistTrainingConfig, train_mnist_stage


CONFIG = MnistTrainingConfig(
    output_model_path=project_file("models", "stage_01_mnist_base.keras"),
    batch_size=64,
    epochs=5,
    learning_rate=1e-3,
)


def main() -> None:
    """Run stage 1 training on MNIST."""
    result = train_mnist_stage(CONFIG)
    if result.validation_accuracy is not None:
        print(f"Validation accuracy: {result.validation_accuracy * 100:.2f}%")
    print(f"Classes: {list(result.class_names)}")
    print(f"Saved: {result.output_model_path}")


if __name__ == "__main__":
    main()
