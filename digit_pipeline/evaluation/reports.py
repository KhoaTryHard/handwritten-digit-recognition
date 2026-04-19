# Module nay thu thap du doan tren tap val va xuat bao cao confusion.
"""Batch evaluation and reporting helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from digit_pipeline.config import ensure_directory, ensure_parent_directory
from digit_pipeline.data_loading import load_evaluation_directory_dataset
from digit_pipeline.evaluation.inference import load_digit_model


@dataclass(frozen=True)
class DirectoryPredictions:
    """Predictions collected for a class-organized directory."""

    class_names: tuple[str, ...]
    file_paths: tuple[str, ...]
    y_true: np.ndarray
    y_pred: np.ndarray
    probabilities: np.ndarray
    images_u8: np.ndarray | None = None


@dataclass(frozen=True)
class ClassAccuracy:
    """Accuracy details for a single class."""

    label: str
    accuracy: float
    correct: int
    total: int


@dataclass(frozen=True)
class ConfusionPair:
    """One non-diagonal confusion entry."""

    true_label: str
    predicted_label: str
    count: int


@dataclass(frozen=True)
class ConfusionSummary:
    """Derived confusion-matrix statistics."""

    confusion_matrix: np.ndarray
    per_class_accuracy: tuple[ClassAccuracy, ...]
    wrong_indices: np.ndarray
    top_confusions: tuple[ConfusionPair, ...]


def collect_directory_predictions(
    directory: str | Path,
    model_path: str | Path,
    batch_size: int,
    *,
    keep_images: bool = False,
) -> DirectoryPredictions:
    """Collect model predictions for all images in a directory dataset."""
    datasets = load_evaluation_directory_dataset(directory, batch_size)
    model = load_digit_model(model_path)

    output_classes = int(model.output_shape[-1])
    if output_classes != len(datasets.class_names):
        raise ValueError(
            f"Mismatch classes: dataset={len(datasets.class_names)}, model={output_classes}"
        )

    # Predict tren pipeline da normalize de tan dung batch va prefetch.
    probabilities = np.asarray(
        model.predict(datasets.prepared_ds, verbose=0),
        dtype=np.float32,
    )
    label_batches: list[np.ndarray] = []
    image_batches: list[np.ndarray] = []

    for raw_images, labels in datasets.raw_ds:
        label_batches.append(labels.numpy())
        if keep_images:
            image_batches.append(raw_images.numpy())

    y_true = (
        np.concatenate(label_batches, axis=0)
        if label_batches
        else np.empty((0,), dtype=np.int32)
    )
    y_pred = (
        probabilities.argmax(axis=1).astype(np.int32)
        if probabilities.size
        else np.empty((0,), dtype=np.int32)
    )
    images_u8 = (
        np.concatenate(image_batches, axis=0)
        if keep_images and image_batches
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


def build_confusion_summary(
    predictions: DirectoryPredictions,
    *,
    max_pairs: int = 15,
) -> ConfusionSummary:
    """Build confusion matrix statistics from directory predictions."""
    confusion_matrix = tf.math.confusion_matrix(
        predictions.y_true,
        predictions.y_pred,
        num_classes=len(predictions.class_names),
    ).numpy()

    row_sums = confusion_matrix.sum(axis=1)
    diagonal = np.diag(confusion_matrix)
    class_accuracy_rows: list[ClassAccuracy] = []

    for class_index, class_name in enumerate(predictions.class_names):
        total = int(row_sums[class_index])
        correct = int(diagonal[class_index])
        accuracy = float(correct / total) if total else 0.0
        class_accuracy_rows.append(
            ClassAccuracy(
                label=class_name,
                accuracy=accuracy,
                correct=correct,
                total=total,
            )
        )

    wrong_indices = np.flatnonzero(predictions.y_true != predictions.y_pred)
    confusion_without_diagonal = confusion_matrix.copy()
    np.fill_diagonal(confusion_without_diagonal, 0)
    confusion_pairs: list[ConfusionPair] = []

    # Bo qua duong cheo chinh va chi giu cac cap nham lan xuat hien.
    for true_index in range(confusion_without_diagonal.shape[0]):
        for pred_index in range(confusion_without_diagonal.shape[1]):
            count = int(confusion_without_diagonal[true_index, pred_index])
            if count <= 0:
                continue
            confusion_pairs.append(
                ConfusionPair(
                    true_label=predictions.class_names[true_index],
                    predicted_label=predictions.class_names[pred_index],
                    count=count,
                )
            )

    confusion_pairs.sort(key=lambda pair: pair.count, reverse=True)
    return ConfusionSummary(
        confusion_matrix=confusion_matrix,
        per_class_accuracy=tuple(class_accuracy_rows),
        wrong_indices=wrong_indices,
        top_confusions=tuple(confusion_pairs[:max_pairs]),
    )


def export_misclassified_predictions(
    predictions: DirectoryPredictions,
    output_csv: str | Path,
) -> pd.DataFrame:
    """Export misclassified samples to a CSV file."""
    wrong_indices = np.flatnonzero(predictions.y_true != predictions.y_pred)
    rows: list[dict[str, object]] = []

    # Ghi lai nhan dung, nhan sai va top-3 xac suat de phan tich loi.
    for index in wrong_indices:
        true_index = int(predictions.y_true[index])
        pred_index = int(predictions.y_pred[index])
        probability_vector = predictions.probabilities[index]
        top_indices = np.argsort(-probability_vector)[:3]

        rows.append(
            {
                "path": predictions.file_paths[index],
                "true": predictions.class_names[true_index],
                "pred": predictions.class_names[pred_index],
                "p_pred": float(probability_vector[pred_index]),
                "p_true": float(probability_vector[true_index]),
                "top3": ",".join(
                    f"{predictions.class_names[top_index]}({probability_vector[top_index]:.3f})"
                    for top_index in top_indices
                ),
            }
        )

    dataframe = pd.DataFrame(rows)
    output_path = ensure_parent_directory(output_csv)
    dataframe.to_csv(output_path, index=False, encoding="utf-8-sig")
    return dataframe


def _resolve_class_index(class_names: tuple[str, ...], label: int | str) -> int:
    """Resolve a class label or index into a numeric index."""
    if isinstance(label, int):
        return label
    return class_names.index(str(label))


def export_confusion_pair_images(
    predictions: DirectoryPredictions,
    output_dir: str | Path,
    *,
    true_label: int | str,
    predicted_label: int | str,
    max_save: int = 50,
) -> list[Path]:
    """Export images for one chosen true/predicted confusion pair."""
    if predictions.images_u8 is None:
        raise RuntimeError("Raw images were not collected for export.")

    true_index = _resolve_class_index(predictions.class_names, true_label)
    predicted_index = _resolve_class_index(predictions.class_names, predicted_label)
    resolved_output_dir = ensure_directory(output_dir)

    matching_indices = np.where(
        (predictions.y_true == true_index) & (predictions.y_pred == predicted_index)
    )[0][:max_save]
    saved_paths: list[Path] = []

    for index in matching_indices:
        output_path = resolved_output_dir / (
            f"{predictions.class_names[true_index]}_to_"
            f"{predictions.class_names[predicted_index]}_idx{index}.png"
        )
        tf.keras.utils.save_img(str(output_path), predictions.images_u8[index])
        saved_paths.append(output_path)

    return saved_paths
