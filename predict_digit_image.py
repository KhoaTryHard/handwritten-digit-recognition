import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import matplotlib.pyplot as plt

from digit_pipeline.inference import load_digit_model, predict_digit_from_image
from project_paths import project_path


MODEL_PATH = project_path("models", "stage_03_final.keras")
IMAGE_PATH = r"C:\Users\LENOVO\Pictures\Screenshots\Screenshot 2026-04-09 083700.png"
PREPROCESS_THRESHOLD = 0.18
TTA_SAMPLES = 30


def resolve_image_path() -> str:
    if IMAGE_PATH:
        return IMAGE_PATH

    if len(sys.argv) >= 2:
        return sys.argv[1]

    raise SystemExit(
        "Set IMAGE_PATH in predict_digit_image.py or run: "
        "python predict_digit_image.py <image_path>"
    )


def main() -> None:
    image_path = resolve_image_path()
    model = load_digit_model(MODEL_PATH)
    result = predict_digit_from_image(
        image_path,
        model,
        preprocess_threshold=PREPROCESS_THRESHOLD,
        tta_samples=TTA_SAMPLES,
    )

    print("Top-5 predictions:")
    for digit in result.top_indices:
        print(int(digit), float(result.probabilities[digit]))

    print(f"Pred = {result.prediction} | confidence = {result.confidence:.4f}")

    plt.imshow(result.preview, cmap="gray")
    plt.title(f"Pred={result.prediction}")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
