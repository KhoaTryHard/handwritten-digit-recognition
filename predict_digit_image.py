import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from digit_pipeline.preprocessing import (
    predict_tta,
    preprocess_handwritten_mnist_like,
)
from project_paths import project_path


MODEL_PATH = project_path("models", "stage_03_final.keras")
PREPROCESS_THRESHOLD = 0.18
TTA_SAMPLES = 30


def resolve_image_path() -> str:
    if len(sys.argv) >= 2:
        return sys.argv[1]

    raise SystemExit("Usage: python predict_digit_image.py <image_path>")


def main() -> None:
    image_path = resolve_image_path()
    model = tf.keras.models.load_model(MODEL_PATH)
    x, preview = preprocess_handwritten_mnist_like(
        image_path,
        threshold=PREPROCESS_THRESHOLD,
    )
    probabilities = predict_tta(model, x, num_samples=TTA_SAMPLES)

    top5 = probabilities.argsort()[-5:][::-1]
    print("Top-5 predictions:")
    for digit in top5:
        print(digit, float(probabilities[digit]))

    prediction = int(top5[0])
    confidence = float(probabilities[prediction])
    print(f"Pred = {prediction} | confidence = {confidence:.4f}")

    plt.imshow(preview, cmap="gray")
    plt.title(f"Pred={prediction}")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
