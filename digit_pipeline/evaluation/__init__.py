# Module nay gom logic suy luan, confusion matrix va xuat bao cao loi.
"""Inference and evaluation helpers for the digit pipeline."""

from digit_pipeline.evaluation.inference import (
    SingleImagePrediction,
    load_digit_model,
    predict_digit_from_image,
)
from digit_pipeline.evaluation.reports import (
    ClassAccuracy,
    ConfusionPair,
    ConfusionSummary,
    DirectoryPredictions,
    build_confusion_summary,
    collect_directory_predictions,
    export_confusion_pair_images,
    export_misclassified_predictions,
)

__all__ = [
    "ClassAccuracy",
    "ConfusionPair",
    "ConfusionSummary",
    "DirectoryPredictions",
    "SingleImagePrediction",
    "build_confusion_summary",
    "collect_directory_predictions",
    "export_confusion_pair_images",
    "export_misclassified_predictions",
    "load_digit_model",
    "predict_digit_from_image",
]
