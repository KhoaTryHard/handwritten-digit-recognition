from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import tensorflow as tf


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


@dataclass(frozen=True)
class TrainValDatasets:
    train_ds: tf.data.Dataset
    val_ds: tf.data.Dataset
    class_names: list[str]


@dataclass(frozen=True)
class EvaluationDatasets:
    raw_ds: tf.data.Dataset
    prepared_ds: tf.data.Dataset
    class_names: list[str]
    file_paths: list[str]


def normalize_batch(x: tf.Tensor, y: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    return tf.cast(x, tf.float32) / 255.0, y


def build_digit_augmenter(
    rotation: float = 0.08,
    translation: float = 0.10,
    zoom: float = 0.08,
) -> tf.keras.Sequential:
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(rotation),
            tf.keras.layers.RandomTranslation(translation, translation),
            tf.keras.layers.RandomZoom(zoom),
        ],
        name="digit_augmenter",
    )


def apply_augmentation(
    dataset: tf.data.Dataset,
    augmenter: tf.keras.Sequential,
) -> tf.data.Dataset:
    return dataset.map(
        lambda x, y: (augmenter(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    ).prefetch(tf.data.AUTOTUNE)


def _prepare_dataset(dataset: tf.data.Dataset) -> tf.data.Dataset:
    return dataset.map(
        normalize_batch,
        num_parallel_calls=tf.data.AUTOTUNE,
    ).prefetch(tf.data.AUTOTUNE)


def directory_has_images(directory: str | Path) -> bool:
    path = Path(directory)
    if not path.is_dir():
        return False

    return any(
        image_path.suffix.lower() in IMAGE_EXTENSIONS
        for image_path in path.rglob("*")
        if image_path.is_file()
    )


def load_train_val_image_datasets(
    train_dir: str,
    val_dir: str,
    batch_size: int,
    *,
    image_size: tuple[int, int] = (28, 28),
    validation_split: float = 0.2,
    seed: int = 42,
) -> TrainValDatasets:
    if directory_has_images(val_dir):
        train_raw = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            color_mode="grayscale",
            image_size=image_size,
            batch_size=batch_size,
            label_mode="int",
        )
        val_raw = tf.keras.utils.image_dataset_from_directory(
            val_dir,
            color_mode="grayscale",
            image_size=image_size,
            batch_size=batch_size,
            label_mode="int",
        )
    else:
        train_raw = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            color_mode="grayscale",
            image_size=image_size,
            batch_size=batch_size,
            label_mode="int",
            validation_split=validation_split,
            subset="training",
            seed=seed,
        )
        val_raw = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            color_mode="grayscale",
            image_size=image_size,
            batch_size=batch_size,
            label_mode="int",
            validation_split=validation_split,
            subset="validation",
            seed=seed,
        )

    return TrainValDatasets(
        train_ds=_prepare_dataset(train_raw),
        val_ds=_prepare_dataset(val_raw),
        class_names=list(train_raw.class_names),
    )


def load_evaluation_dataset(
    directory: str,
    batch_size: int,
    *,
    image_size: tuple[int, int] = (28, 28),
) -> EvaluationDatasets:
    raw_ds = tf.keras.utils.image_dataset_from_directory(
        directory,
        color_mode="grayscale",
        image_size=image_size,
        batch_size=batch_size,
        label_mode="int",
        shuffle=False,
    )

    return EvaluationDatasets(
        raw_ds=raw_ds,
        prepared_ds=_prepare_dataset(raw_ds),
        class_names=list(raw_ds.class_names),
        file_paths=list(getattr(raw_ds, "file_paths", [])),
    )
