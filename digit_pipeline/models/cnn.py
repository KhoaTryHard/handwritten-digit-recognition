# Module nay dinh nghia kien truc CNN co the tai su dung cho ca 3 stage.
"""CNN architectures for handwritten digit classification."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers

from digit_pipeline.config.settings import DIGIT_IMAGE_SIZE, NUM_DIGIT_CLASSES


def build_base_digit_cnn(
    input_shape: tuple[int, int, int] = (*DIGIT_IMAGE_SIZE, 1),
    num_classes: int = NUM_DIGIT_CLASSES,
    dropout_rate: float = 0.25,
) -> tf.keras.Model:
    """Build the baseline CNN used throughout the project."""
    inputs = layers.Input(shape=input_shape, name="image")

    # Khoi tich chap dau tien hoc cac net co ban cua chu so.
    x = layers.Conv2D(32, 3, padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(32, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Dropout(dropout_rate)(x)

    # Khoi thu hai mo rong so kenh de hoc dac trung phuc tap hon.
    x = layers.Conv2D(64, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(64, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Dropout(dropout_rate + 0.05)(x)

    # Dau ra gom pooling toan cuc va softmax cho 10 lop chu so.
    x = layers.Conv2D(96, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout_rate + 0.10)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="probabilities")(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="digit_cnn")
