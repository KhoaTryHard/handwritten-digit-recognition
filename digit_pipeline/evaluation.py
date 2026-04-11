from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from digit_pipeline.data import load_evaluation_dataset


@dataclass(frozen=True)
class DirectoryPredictions:
    class_names: list[str]
    file_paths: list[str]
    y_true: np.ndarray
    y_pred: np.ndarray
    probabilities: np.ndarray
    images_u8: np.ndarray | None = None


def collect_directory_predictions(
    val_dir: str,
    model_path: str,
    batch_size: int,
    *,
    keep_images: bool = False,
) -> DirectoryPredictions:
    datasets = load_evaluation_dataset(val_dir, batch_size)
    model = tf.keras.models.load_model(model_path)

    if model.output_shape[-1] != len(datasets.class_names):
        raise ValueError(
            f"Mismatch classes: dataset={len(datasets.class_names)}, "
            f"model={model.output_shape[-1]}"
        )

    y_true_batches: list[np.ndarray] = []
    probability_batches: list[np.ndarray] = []
    image_batches: list[np.ndarray] = []

    for x_u8, y in datasets.raw_ds:
        x = tf.cast(x_u8, tf.float32) / 255.0
        probs = model.predict(x, verbose=0)
        y_true_batches.append(y.numpy())
        probability_batches.append(probs)

        if keep_images:
            image_batches.append(x_u8.numpy())

    probabilities = (
        np.concatenate(probability_batches, axis=0)
        if probability_batches
        else np.empty((0, len(datasets.class_names)), dtype=float)
    )
    y_true = (
        np.concatenate(y_true_batches, axis=0)
        if y_true_batches
        else np.empty((0,), dtype=int)
    )
    y_pred = (
        np.argmax(probabilities, axis=1)
        if len(probabilities) > 0
        else np.empty((0,), dtype=int)
    )
    images_u8 = (
        np.concatenate(image_batches, axis=0)
        if image_batches
        else None
    )

    return DirectoryPredictions(
        class_names=datasets.class_names,
        file_paths=datasets.file_paths,
        y_true=y_true,
        y_pred=y_pred,
        probabilities=probabilities,
        images_u8=images_u8,
    )
