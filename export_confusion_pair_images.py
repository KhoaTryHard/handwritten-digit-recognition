import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from pathlib import Path

import numpy as np
import tensorflow as tf

from digit_pipeline.evaluation import collect_directory_predictions
from project_paths import project_path


VAL_DIR = project_path("my_digits_28", "val")
MODEL_PATH = project_path("models", "stage_03_final.keras")
OUTPUT_DIR = project_path("reports", "confusion_pairs")

BATCH_SIZE = 64
MAX_SAVE = 50
TARGET_TRUE = 9
TARGET_PRED = 3


def main() -> None:
    results = collect_directory_predictions(
        VAL_DIR,
        MODEL_PATH,
        BATCH_SIZE,
        keep_images=True,
    )
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    if results.images_u8 is None:
        raise RuntimeError("Raw images were not collected for export.")

    matching_indices = np.where(
        (results.y_true == TARGET_TRUE) & (results.y_pred == TARGET_PRED)
    )[0][:MAX_SAVE]

    for index in matching_indices:
        tf.keras.utils.save_img(
            str(output_dir / f"{TARGET_TRUE}_to_{TARGET_PRED}_idx{index}.png"),
            results.images_u8[index],
        )

    print(f"Count saved {TARGET_TRUE}->{TARGET_PRED}: {len(matching_indices)}")
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()
