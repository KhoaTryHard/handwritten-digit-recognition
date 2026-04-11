import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from digit_pipeline.training import FineTuneConfig, run_finetune
from project_paths import project_path


CONFIG = FineTuneConfig(
    train_dir=project_path("my_digits_28", "train"),
    val_dir=project_path("my_digits_28", "val"),
    input_model=project_path("models", "stage_02_emnist_finetuned.keras"),
    output_model=project_path("models", "stage_03_final.keras"),
    batch_size=64,
    epochs=40,
    learning_rate=1e-4,
)


def main() -> None:
    class_names = run_finetune(CONFIG)
    print(f"Classes: {class_names}")
    print("Stage 3/3: final training on the unified 28x28 personal dataset")
    print(f"Saved: {CONFIG.output_model}")


if __name__ == "__main__":
    main()
