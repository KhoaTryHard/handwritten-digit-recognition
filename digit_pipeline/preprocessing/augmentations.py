# Module nay chua cac ham augmentation batch cho train va test-time augmentation.
"""Helper augmentation TensorFlow cho training va inference."""

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
    """Tao augmentation stack dung cho anh chu so."""
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
    """Ap dung augmentation anh tong quat cho mot batch."""
    return augmenter(images, training=True), labels


@tf.function(reduce_retracing=True)
def add_speckle_noise(
    images: tf.Tensor,
    probability: float = 0.002,
) -> tf.Tensor:
    """Them speckle noise nhe vao batch anh da chuan hoa."""
    noise_mask = tf.cast(tf.random.uniform(tf.shape(images)) < probability, tf.float32)
    return tf.clip_by_value(images + 0.8 * noise_mask, 0.0, 1.0)


@tf.function(reduce_retracing=True)
def apply_emnist_augmentation(
    images: tf.Tensor,
    labels: tf.Tensor,
    augmenter: tf.keras.Sequential,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Ap dung augmentation manh hon phu hop voi EMNIST digits."""
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
    """Gan ham augmentation theo batch vao dataset."""

    def map_batch(images: tf.Tensor, labels: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """Ap dung augmentation da cau hinh ben trong tf.data."""
        return augment_fn(images, labels, augmenter)

    return dataset.map(map_batch, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
