from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import tensorflow as tf

from digit_pipeline.data import (
    apply_augmentation,
    build_digit_augmenter,
    load_train_val_image_datasets,
)


@dataclass(frozen=True)
class FineTuneConfig:
    train_dir: str
    val_dir: str
    input_model: str
    output_model: str
    batch_size: int = 64
    epochs: int = 20
    learning_rate: float = 1e-4
    early_stopping_patience: int = 5
    plateau_factor: float = 0.5
    plateau_patience: int = 2
    min_lr: float = 1e-6


def ensure_model_matches_classes(
    model: tf.keras.Model,
    class_names: list[str],
) -> None:
    output_classes = model.output_shape[-1]

    if output_classes != len(class_names):
        raise ValueError(
            f"Mismatch classes: dataset={len(class_names)}, model={output_classes}"
        )


def build_training_callbacks(config: FineTuneConfig) -> list[tf.keras.callbacks.Callback]:
    Path(config.output_model).parent.mkdir(parents=True, exist_ok=True)

    return [
        tf.keras.callbacks.EarlyStopping(
            patience=config.early_stopping_patience,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=config.plateau_factor,
            patience=config.plateau_patience,
            min_lr=config.min_lr,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            config.output_model,
            save_best_only=True,
        ),
    ]


def run_finetune(config: FineTuneConfig) -> list[str]:
    datasets = load_train_val_image_datasets(
        config.train_dir,
        config.val_dir,
        config.batch_size,
    )
    train_ds = apply_augmentation(datasets.train_ds, build_digit_augmenter())

    model = tf.keras.models.load_model(config.input_model)
    ensure_model_matches_classes(model, datasets.class_names)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(config.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        train_ds,
        validation_data=datasets.val_ds,
        epochs=config.epochs,
        callbacks=build_training_callbacks(config),
    )
    model.save(config.output_model)

    return datasets.class_names
