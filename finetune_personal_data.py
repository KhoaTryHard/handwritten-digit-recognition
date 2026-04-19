# Module nay chay stage 3 de fine-tune tren du lieu viet tay ca nhan 28x28.
"""Fine-tune the classifier on the personal 28x28 dataset."""

from __future__ import annotations

from digit_pipeline.config import project_file
from digit_pipeline.training import DirectoryFineTuneConfig, fine_tune_directory_stage


CONFIG = DirectoryFineTuneConfig(
    train_dir=project_file("my_digits_28", "train"),
    val_dir=project_file("my_digits_28", "val"),
    input_model_path=project_file("models", "stage_02_emnist_finetuned.keras"),
    output_model_path=project_file("models", "stage_03_final.keras"),
    batch_size=64,
    epochs=40,
    learning_rate=1e-4,
)


def main() -> None:
    """Run stage 3 fine-tuning on the personal dataset."""
    result = fine_tune_directory_stage(CONFIG)
    if result.validation_accuracy is not None:
        print(f"Validation accuracy: {result.validation_accuracy * 100:.2f}%")
    print(f"Classes: {list(result.class_names)}")
    print("Stage 3/3: final training on the unified 28x28 personal dataset")
    print(f"Saved: {result.output_model_path}")


if __name__ == "__main__":
    main()
