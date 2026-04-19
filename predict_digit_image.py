# Module nay du doan mot anh don le bang pipeline suy luan dung chung.
"""Predict a digit for a single image path."""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt

from digit_pipeline.config import PredictionDefaults, project_file
from digit_pipeline.evaluation import load_digit_model, predict_digit_from_image


MODEL_PATH = project_file("models", "stage_03_final.keras")
IMAGE_PATH: str | None = None
PREDICTION_DEFAULTS = PredictionDefaults()


def resolve_image_path() -> str:
    """Resolve the image path from the script constant or CLI argument."""
    if IMAGE_PATH:
        return IMAGE_PATH

    if len(sys.argv) >= 2:
        return sys.argv[1]

    raise SystemExit(
        "Set IMAGE_PATH in predict_digit_image.py or run: "
        "python predict_digit_image.py <image_path>"
    )


def main() -> None:
    """Run prediction for one image and display the preview."""
    image_path = resolve_image_path()
    model = load_digit_model(MODEL_PATH)
    result = predict_digit_from_image(
        image_path,
        model,
        preprocess_threshold=PREDICTION_DEFAULTS.preprocess_threshold,
        tta_samples=PREDICTION_DEFAULTS.tta_samples,
        top_k=PREDICTION_DEFAULTS.top_k,
    )

    print("Top predictions:")
    for digit_index in result.top_indices:
        print(int(digit_index), float(result.probabilities[digit_index]))

    print(f"Pred = {result.prediction} | confidence = {result.confidence:.4f}")

    # Hien thi anh preview sau tien xu ly de doi chieu voi ket qua du doan.
    plt.imshow(result.preview, cmap="gray")
    plt.title(f"Pred={result.prediction}")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
