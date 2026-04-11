import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from pathlib import Path

import tensorflow as tf
import tensorflow_datasets as tfds

from digit_pipeline.data import build_digit_augmenter
from project_paths import project_file


BATCH_SIZE = 128
EPOCHS = 5
INPUT_MODEL = project_file("models", "stage_01_mnist_base.keras")
OUTPUT_MODEL = project_file("models", "stage_02_emnist_finetuned.keras")

EMNIST_AUGMENTER = build_digit_augmenter(
    rotation=0.10,
    translation=0.15,
    zoom=0.12,
)


def normalize_sample(x: tf.Tensor, y: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    x = tf.cast(x, tf.float32) / 255.0
    if x.shape.rank == 2:
        x = tf.expand_dims(x, -1)
    return x, y


def add_speckle_noise(x: tf.Tensor, probability: float = 0.002) -> tf.Tensor:
    dots = tf.cast(tf.random.uniform(tf.shape(x)) < probability, tf.float32)
    return tf.clip_by_value(x + 0.8 * dots, 0.0, 1.0)


def train_map(x: tf.Tensor, y: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    x = EMNIST_AUGMENTER(x, training=True)
    flip_mask = tf.random.uniform([tf.shape(x)[0], 1, 1, 1]) < 0.5
    x = tf.where(flip_mask, x, 1.0 - x)
    x = add_speckle_noise(x)
    return x, y


def main() -> None:
    Path(OUTPUT_MODEL).parent.mkdir(parents=True, exist_ok=True)

    train_ds, test_ds = tfds.load(
        "emnist/digits",
        split=["train", "test"],
        as_supervised=True,
    )

    train_ds = (
        train_ds.map(normalize_sample, num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(10_000)
        .batch(BATCH_SIZE)
        .map(train_map, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )
    test_ds = (
        test_ds.map(normalize_sample, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    model = tf.keras.models.load_model(INPUT_MODEL)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    model.fit(train_ds, validation_data=test_ds, epochs=EPOCHS)
    model.save(OUTPUT_MODEL)
    print(f"Saved: {OUTPUT_MODEL}")


if __name__ == "__main__":
    main()
