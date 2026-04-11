import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from pathlib import Path

import numpy as np
import pandas as pd

from digit_pipeline.evaluation import collect_directory_predictions
from project_paths import project_path


VAL_DIR = project_path("my_digits_28", "val")
MODEL_PATH = project_path("models", "stage_03_final.keras")
OUTPUT_CSV = project_path("reports", "misclassified_validation.csv")
BATCH_SIZE = 64


def main() -> None:
    results = collect_directory_predictions(
        VAL_DIR,
        MODEL_PATH,
        BATCH_SIZE,
    )
    wrong_indices = np.where(results.y_true != results.y_pred)[0]
    rows = []

    for index in wrong_indices:
        true_idx = int(results.y_true[index])
        pred_idx = int(results.y_pred[index])
        probabilities = results.probabilities[index]
        top3 = np.argsort(-probabilities)[:3]

        rows.append(
            {
                "path": results.file_paths[index],
                "true": results.class_names[true_idx],
                "pred": results.class_names[pred_idx],
                "p_pred": float(probabilities[pred_idx]),
                "p_true": float(probabilities[true_idx]),
                "top3": ",".join(
                    f"{results.class_names[top_idx]}({probabilities[top_idx]:.3f})"
                    for top_idx in top3
                ),
            }
        )

    output_path = Path(OUTPUT_CSV)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataframe = pd.DataFrame(rows)
    print(f"Total wrong: {len(rows)}/{len(results.file_paths)}")
    print(dataframe.head(1000).to_string(index=False))

    dataframe.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Saved CSV: {output_path}")


if __name__ == "__main__":
    main()
