from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from digit_pipeline.preprocessing import predict_tta, preprocess_handwritten_mnist_like


@dataclass(frozen=True)
class SingleImagePrediction:
    image_path: str
    probabilities: np.ndarray
    preview: Image.Image
    prediction: int
    confidence: float
    top_indices: np.ndarray


def load_digit_model(model_path: str | Path) -> tf.keras.Model:
    return tf.keras.models.load_model(model_path)


def predict_digit_from_image(
    image_path: str | Path,
    model: tf.keras.Model,
    *,
    preprocess_threshold: float = 0.18,
    tta_samples: int = 30,
    top_k: int = 5,
) -> SingleImagePrediction:
    x, preview = preprocess_handwritten_mnist_like(
        image_path,
        threshold=preprocess_threshold,
    )
    probabilities = predict_tta(model, x, num_samples=tta_samples)
    top_indices = probabilities.argsort()[-top_k:][::-1]
    prediction = int(top_indices[0])
    confidence = float(probabilities[prediction])

    return SingleImagePrediction(
        image_path=str(image_path),
        probabilities=probabilities,
        preview=preview,
        prediction=prediction,
        confidence=confidence,
        top_indices=top_indices,
    )
