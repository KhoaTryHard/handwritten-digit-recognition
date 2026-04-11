from __future__ import annotations

from tensorflow.keras import layers, models


def build_base_digit_cnn(
    input_shape: tuple[int, int, int] = (28, 28, 1),
    num_classes: int = 10,
) -> models.Sequential:
    return models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="digit_cnn",
    )
