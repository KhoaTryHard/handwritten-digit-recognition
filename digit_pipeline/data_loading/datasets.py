# Module nay chua cac pipeline tf.data de nap va chuan hoa du lieu anh.
"""Dataset builders for MNIST, EMNIST, and directory-based images."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from digit_pipeline.config.settings import (
    DEFAULT_SEED,
    DIGIT_IMAGE_SIZE,
    DIGIT_LABELS,
)
from digit_pipeline.utils import list_image_files


AUTOTUNE = tf.data.AUTOTUNE


@dataclass(frozen=True)
class DatasetBundle:
    """Training and validation datasets with class names."""

    train_ds: tf.data.Dataset
    val_ds: tf.data.Dataset
    class_names: tuple[str, ...]


@dataclass(frozen=True)
class EvaluationDatasetBundle:
    """Evaluation datasets with file path ordering."""

    raw_ds: tf.data.Dataset
    prepared_ds: tf.data.Dataset
    class_names: tuple[str, ...]
    file_paths: tuple[str, ...]


@tf.function(reduce_retracing=True)
def normalize_supervised_example(
    image: tf.Tensor,
    label: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Normalize a supervised example to float32 grayscale."""
    normalized_image = tf.cast(image, tf.float32) / 255.0
    if normalized_image.shape.rank == 2:
        normalized_image = tf.expand_dims(normalized_image, axis=-1)
    return normalized_image, tf.cast(label, tf.int32)


def _cache_if_needed(
    dataset: tf.data.Dataset,
    cache_in_memory: bool,
) -> tf.data.Dataset:
    """Cache a dataset when requested."""
    return dataset.cache() if cache_in_memory else dataset


def _prepare_unbatched_dataset(
    dataset: tf.data.Dataset,
    *,
    batch_size: int,
    seed: int,
    shuffle_buffer: int | None = None,
    cache_in_memory: bool = True,
) -> tf.data.Dataset:
    """Normalize, optionally shuffle, batch, and prefetch a sample dataset."""
    normalized_dataset = dataset.map(
        normalize_supervised_example,
        num_parallel_calls=AUTOTUNE,
    )
    normalized_dataset = _cache_if_needed(normalized_dataset, cache_in_memory)

    # Shuffle chi ap dung cho tap train de giu thu tu tap validation.
    if shuffle_buffer is not None:
        normalized_dataset = normalized_dataset.shuffle(
            shuffle_buffer,
            seed=seed,
            reshuffle_each_iteration=True,
        )

    return normalized_dataset.batch(batch_size).prefetch(AUTOTUNE)


def _prepare_batched_dataset(
    dataset: tf.data.Dataset,
    *,
    cache_in_memory: bool = False,
) -> tf.data.Dataset:
    """Normalize, optionally cache, and prefetch an already-batched dataset."""
    normalized_dataset = dataset.map(
        normalize_supervised_example,
        num_parallel_calls=AUTOTUNE,
    )
    normalized_dataset = _cache_if_needed(normalized_dataset, cache_in_memory)
    return normalized_dataset.prefetch(AUTOTUNE)


def load_mnist_arrays() -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Return raw MNIST arrays for visualization or custom processing."""
    return tf.keras.datasets.mnist.load_data()


def build_mnist_datasets(
    batch_size: int,
    *,
    seed: int = DEFAULT_SEED,
    cache_in_memory: bool = True,
) -> DatasetBundle:
    """Build tf.data datasets for MNIST."""
    (train_images, train_labels), (test_images, test_labels) = load_mnist_arrays()
    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    val_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))

    return DatasetBundle(
        train_ds=_prepare_unbatched_dataset(
            train_ds,
            batch_size=batch_size,
            seed=seed,
            shuffle_buffer=len(train_images),
            cache_in_memory=cache_in_memory,
        ),
        val_ds=_prepare_unbatched_dataset(
            val_ds,
            batch_size=batch_size,
            seed=seed,
            cache_in_memory=cache_in_memory,
        ),
        class_names=DIGIT_LABELS,
    )


def load_emnist_datasets(
    batch_size: int,
    *,
    seed: int = DEFAULT_SEED,
    cache_in_memory: bool = True,
) -> DatasetBundle:
    """Build tf.data datasets for EMNIST digits."""
    train_ds, test_ds = tfds.load(
        "emnist/digits",
        split=["train", "test"],
        as_supervised=True,
    )

    return DatasetBundle(
        train_ds=_prepare_unbatched_dataset(
            train_ds,
            batch_size=batch_size,
            seed=seed,
            shuffle_buffer=20_000,
            cache_in_memory=cache_in_memory,
        ),
        val_ds=_prepare_unbatched_dataset(
            test_ds,
            batch_size=batch_size,
            seed=seed,
            cache_in_memory=cache_in_memory,
        ),
        class_names=DIGIT_LABELS,
    )


def directory_has_images(directory: str | Path) -> bool:
    """Return whether a directory contains at least one supported image."""
    resolved_directory = Path(directory)
    if not resolved_directory.is_dir():
        return False
    return bool(list_image_files(resolved_directory, recursive=True))


def load_directory_datasets(
    train_dir: str | Path,
    val_dir: str | Path,
    batch_size: int,
    *,
    image_size: tuple[int, int] = DIGIT_IMAGE_SIZE,
    validation_split: float = 0.2,
    seed: int = DEFAULT_SEED,
    cache_in_memory: bool = False,
) -> DatasetBundle:
    """Load train and validation datasets from image directories."""
    if directory_has_images(val_dir):
        train_raw = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            color_mode="grayscale",
            image_size=image_size,
            batch_size=batch_size,
            label_mode="int",
            shuffle=True,
            seed=seed,
        )
        val_raw = tf.keras.utils.image_dataset_from_directory(
            val_dir,
            color_mode="grayscale",
            image_size=image_size,
            batch_size=batch_size,
            label_mode="int",
            shuffle=False,
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
            shuffle=False,
        )

    return DatasetBundle(
        train_ds=_prepare_batched_dataset(
            train_raw,
            cache_in_memory=cache_in_memory,
        ),
        val_ds=_prepare_batched_dataset(
            val_raw,
            cache_in_memory=cache_in_memory,
        ),
        class_names=tuple(train_raw.class_names),
    )


def load_evaluation_directory_dataset(
    directory: str | Path,
    batch_size: int,
    *,
    image_size: tuple[int, int] = DIGIT_IMAGE_SIZE,
) -> EvaluationDatasetBundle:
    """Load an evaluation dataset and preserve file ordering."""
    raw_dataset = tf.keras.utils.image_dataset_from_directory(
        directory,
        color_mode="grayscale",
        image_size=image_size,
        batch_size=batch_size,
        label_mode="int",
        shuffle=False,
    )

    return EvaluationDatasetBundle(
        raw_ds=raw_dataset,
        prepared_ds=_prepare_batched_dataset(raw_dataset),
        class_names=tuple(raw_dataset.class_names),
        file_paths=tuple(getattr(raw_dataset, "file_paths", [])),
    )
