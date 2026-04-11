import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf

from digit_pipeline.evaluation import collect_directory_predictions
from project_paths import project_path


VAL_DIR = project_path("my_digits_28", "val")
MODEL_PATH = project_path("models", "stage_03_final.keras")
BATCH_SIZE = 64


def main() -> None:
    results = collect_directory_predictions(
        VAL_DIR,
        MODEL_PATH,
        BATCH_SIZE,
    )

    print(f"Classes: {results.class_names}")

    confusion = tf.math.confusion_matrix(
        results.y_true,
        results.y_pred,
        num_classes=len(results.class_names),
    ).numpy()

    print("\nConfusion matrix (rows=true, cols=pred):\n", confusion)

    row_sums = confusion.sum(axis=1)
    diagonal = np.diag(confusion)
    per_class_accuracy = np.divide(
        diagonal,
        row_sums,
        out=np.zeros_like(diagonal, dtype=float),
        where=row_sums != 0,
    )

    print("\nPer-class accuracy:")
    for index, accuracy in enumerate(per_class_accuracy):
        print(
            f"  class {results.class_names[index]}: "
            f"{accuracy:.4f} ({diagonal[index]}/{row_sums[index]})"
        )

    wrong_indices = np.where(results.y_true != results.y_pred)[0]
    print(f"\nNum wrong: {len(wrong_indices)}/{len(results.y_true)}")
    print(
        "Some wrong (true -> pred):",
        list(zip(results.y_true[wrong_indices[:20]], results.y_pred[wrong_indices[:20]])),
    )

    confusion_no_diag = confusion.copy()
    np.fill_diagonal(confusion_no_diag, 0)
    pairs = []

    for true_idx in range(confusion_no_diag.shape[0]):
        for pred_idx in range(confusion_no_diag.shape[1]):
            count = confusion_no_diag[true_idx, pred_idx]
            if count > 0:
                pairs.append((count, true_idx, pred_idx))

    pairs.sort(reverse=True)

    print("Top confusions:")
    for rank, (count, true_idx, pred_idx) in enumerate(pairs[:15], start=1):
        print(f"{rank:02d}. true {true_idx} -> pred {pred_idx}: {count}")


if __name__ == "__main__":
    main()
