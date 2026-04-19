# Module nay chua cac ham augmentation batch cho train va test-time augmentation.
"""TensorFlow augmentation helpers for training and inference."""

from __future__ import annotations

from typing import Callable

import tensorflow as tf


AUTOTUNE = tf.data.AUTOTUNE
BatchAugmentFn = Callable[
    [tf.Tensor, tf.Tensor, tf.keras.Sequential],
    tuple[tf.Tensor, tf.Tensor],
]


def build_digit_augmenter(
    rotation: float = 0.08,
    translation: float = 0.10,
    zoom: float = 0.08,
    *,
    name: str = "digit_augmenter",
) -> tf.keras.Sequential:
    """Build the augmentation stack used for digit images."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(rotation),
            tf.keras.layers.RandomTranslation(translation, translation),
            tf.keras.layers.RandomZoom(zoom),
        ],
        name=name,
    )


@tf.function(reduce_retracing=True)
def apply_training_augmentation(
    images: tf.Tensor,
    labels: tf.Tensor,
    augmenter: tf.keras.Sequential,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Apply generic image augmentation to a batch."""
    return augmenter(images, training=True), labels


@tf.function(reduce_retracing=True)
def add_speckle_noise(
    images: tf.Tensor,
    probability: float = 0.002,
) -> tf.Tensor:
    """Add light speckle noise to a batch of normalized images."""
    noise_mask = tf.cast(tf.random.uniform(tf.shape(images)) < probability, tf.float32)
    return tf.clip_by_value(images + 0.8 * noise_mask, 0.0, 1.0)


@tf.function(reduce_retracing=True)
def apply_emnist_augmentation(
    images: tf.Tensor,
    labels: tf.Tensor,
    augmenter: tf.keras.Sequential,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Apply stronger augmentation tailored to EMNIST digits."""
    augmented_images = augmenter(images, training=True)
    invert_mask = tf.random.uniform([tf.shape(augmented_images)[0], 1, 1, 1]) < 0.5
    augmented_images = tf.where(invert_mask, augmented_images, 1.0 - augmented_images)
    augmented_images = add_speckle_noise(augmented_images)
    return augmented_images, labels


def attach_augmentation(
    dataset: tf.data.Dataset,
    augmenter: tf.keras.Sequential,
    *,
    augment_fn: BatchAugmentFn = apply_training_augmentation,
) -> tf.data.Dataset:
    """Attach a batched augmentation function to a dataset."""

    def map_batch(images: tf.Tensor, labels: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """Apply the configured augmentation inside tf.data."""
        return augment_fn(images, labels, augmenter)

    return dataset.map(map_batch, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
