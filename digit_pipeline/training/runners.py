# Module nay chua logic train va fine-tune de cac script top-level goi lai.
"""Training runners for MNIST and transfer-learning stages."""

from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf

from digit_pipeline.config import RuntimeConfig, ensure_parent_directory
from digit_pipeline.data_loading import (
    DatasetBundle,
    build_mnist_datasets,
    load_directory_datasets,
    load_emnist_datasets,
)
from digit_pipeline.models import build_base_digit_cnn
from digit_pipeline.preprocessing import (
    apply_emnist_augmentation,
    attach_augmentation,
    build_digit_augmenter,
)
from digit_pipeline.training.configs import (
    CallbackConfig,
    DirectoryFineTuneConfig,
    EmnistTrainingConfig,
    MnistTrainingConfig,
    TransferLearningConfig,
)
from digit_pipeline.utils import configure_runtime


@dataclass(frozen=True)
class TrainingResult:
    """Summary information returned after a training stage."""

    output_model_path: str
    class_names: tuple[str, ...]
    validation_accuracy: float | None


def ensure_model_matches_classes(
    model: tf.keras.Model,
    class_names: tuple[str, ...],
) -> None:
    """Validate that model outputs match dataset classes."""
    output_classes = int(model.output_shape[-1])
    if output_classes != len(class_names):
        raise ValueError(
            f"Mismatch classes: dataset={len(class_names)}, model={output_classes}"
        )


def build_training_callbacks(
    output_model_path: str,
    callback_config: CallbackConfig,
) -> list[tf.keras.callbacks.Callback]:
    """Build the standard fine-tuning callback stack."""
    ensure_parent_directory(output_model_path)
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=callback_config.early_stopping_patience,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=callback_config.plateau_factor,
            patience=callback_config.plateau_patience,
            min_lr=callback_config.min_learning_rate,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=output_model_path,
            monitor="val_loss",
            save_best_only=True,
        ),
    ]


def _compile_classifier(model: tf.keras.Model, learning_rate: float) -> None:
    """Compile a model for sparse multi-class classification."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )


def _evaluate_and_save_model(
    model: tf.keras.Model,
    val_ds: tf.data.Dataset,
    output_model_path: str,
    class_names: tuple[str, ...],
) -> TrainingResult:
    """Evaluate a model, save it, and return a compact summary."""
    evaluation = model.evaluate(val_ds, verbose=0, return_dict=True)
    model.save(output_model_path)
    return TrainingResult(
        output_model_path=output_model_path,
        class_names=class_names,
        validation_accuracy=float(evaluation.get("accuracy", 0.0)),
    )


def _run_transfer_stage(
    datasets: DatasetBundle,
    train_ds: tf.data.Dataset,
    config: TransferLearningConfig,
) -> TrainingResult:
    """Fine-tune an existing classifier on a prepared dataset."""
    model = tf.keras.models.load_model(config.input_model_path)
    ensure_model_matches_classes(model, datasets.class_names)

    # Nap model cu, compile lai va train tren dataset moi.
    _compile_classifier(model, config.learning_rate)
    model.fit(
        train_ds,
        validation_data=datasets.val_ds,
        epochs=config.epochs,
        callbacks=build_training_callbacks(
            str(config.output_model_path),
            config.callback_config,
        ),
        verbose=2,
    )

    return _evaluate_and_save_model(
        model,
        datasets.val_ds,
        str(config.output_model_path),
        datasets.class_names,
    )


def train_mnist_stage(config: MnistTrainingConfig) -> TrainingResult:
    """Train the base CNN on MNIST."""
    configure_runtime(RuntimeConfig(seed=config.seed))
    datasets = build_mnist_datasets(batch_size=config.batch_size, seed=config.seed)
    model = build_base_digit_cnn(num_classes=len(datasets.class_names))

    # Train model base tren MNIST bang pipeline tf.data da cache va prefetch.
    _compile_classifier(model, config.learning_rate)
    model.fit(
        datasets.train_ds,
        validation_data=datasets.val_ds,
        epochs=config.epochs,
        verbose=2,
    )

    ensure_parent_directory(config.output_model_path)
    return _evaluate_and_save_model(
        model,
        datasets.val_ds,
        str(config.output_model_path),
        datasets.class_names,
    )


def train_emnist_stage(config: EmnistTrainingConfig) -> TrainingResult:
    """Fine-tune the MNIST model on EMNIST digits."""
    configure_runtime(RuntimeConfig(seed=config.seed))
    datasets = load_emnist_datasets(batch_size=config.batch_size, seed=config.seed)
    augmenter = build_digit_augmenter(
        rotation=config.rotation,
        translation=config.translation,
        zoom=config.zoom,
        name="emnist_augmenter",
    )
    augmented_train_ds = attach_augmentation(
        datasets.train_ds,
        augmenter,
        augment_fn=apply_emnist_augmentation,
    )

    return _run_transfer_stage(datasets, augmented_train_ds, config)


def fine_tune_directory_stage(config: DirectoryFineTuneConfig) -> TrainingResult:
    """Fine-tune a classifier on a directory-based handwritten dataset."""
    configure_runtime(RuntimeConfig(seed=config.seed))
    datasets = load_directory_datasets(
        train_dir=config.train_dir,
        val_dir=config.val_dir,
        batch_size=config.batch_size,
        image_size=config.image_size,
        seed=config.seed,
        cache_in_memory=config.cache_in_memory,
    )
    augmented_train_ds = attach_augmentation(
        datasets.train_ds,
        build_digit_augmenter(),
    )

    return _run_transfer_stage(datasets, augmented_train_ds, config)
