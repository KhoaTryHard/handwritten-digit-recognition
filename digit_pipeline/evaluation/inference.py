# Module nay thuc hien tien xu ly va suy luan mot anh chu so don le.
"""Single-image inference helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from digit_pipeline.preprocessing import build_digit_augmenter, preprocess_handwritten_image


@dataclass(frozen=True)
class SingleImagePrediction:
    """Prediction details for one image."""

    image_path: str
    probabilities: np.ndarray
    preview: Image.Image
    prediction: int
    confidence: float
    top_indices: np.ndarray


def load_digit_model(model_path: str | Path) -> tf.keras.Model:
    """Load a saved Keras digit classifier."""
    return tf.keras.models.load_model(model_path)


@tf.function(reduce_retracing=True)
def _predict_with_tta(
    model: tf.keras.Model,
    augmenter: tf.keras.Sequential,
    image_batch: tf.Tensor,
    num_samples: int,
) -> tf.Tensor:
    """Predict one batch with test-time augmentation and averaging."""
    repeated_images = tf.repeat(image_batch, repeats=num_samples, axis=0)
    augmented_images = augmenter(repeated_images, training=True)
    probability_batch = model(augmented_images, training=False)
    probability_batch = tf.reshape(
        probability_batch,
        (num_samples, tf.shape(image_batch)[0], tf.shape(probability_batch)[-1]),
    )
    return tf.reduce_mean(probability_batch, axis=0)


def predict_digit_from_image(
    image_path: str | Path,
    model: tf.keras.Model,
    *,
    preprocess_threshold: float = 0.18,
    tta_samples: int = 30,
    top_k: int = 5,
) -> SingleImagePrediction:
    """Predict a handwritten digit from an image path."""
    processed_image = preprocess_handwritten_image(
        image_path,
        threshold=preprocess_threshold,
    )
    augmenter = build_digit_augmenter(
        rotation=0.04,
        translation=0.04,
        zoom=0.04,
        name="tta_augmenter",
    )

    # Lap lai du doan voi augmentation nhe de trung binh hoa xac suat.
    effective_samples = max(1, tta_samples)
    probability_tensor = _predict_with_tta(
        model,
        augmenter,
        tf.convert_to_tensor(processed_image.tensor),
        effective_samples,
    )
    probabilities = np.asarray(probability_tensor.numpy()[0], dtype=np.float32)
    top_count = min(top_k, probabilities.shape[0])
    top_indices = probabilities.argsort()[-top_count:][::-1]
    prediction = int(top_indices[0])
    confidence = float(probabilities[prediction])

    return SingleImagePrediction(
        image_path=str(image_path),
        probabilities=probabilities,
        preview=processed_image.preview,
        prediction=prediction,
        confidence=confidence,
        top_indices=top_indices,
    )
