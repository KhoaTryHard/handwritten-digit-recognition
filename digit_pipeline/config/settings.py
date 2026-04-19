# Module nay luu cac hang so va cau hinh mac dinh dung chung cho pipeline.
"""Shared constants and lightweight configuration objects."""

from __future__ import annotations

from dataclasses import dataclass


DIGIT_LABELS: tuple[str, ...] = tuple(str(index) for index in range(10))
DIGIT_IMAGE_SIZE: tuple[int, int] = (28, 28)
NUM_DIGIT_CLASSES = len(DIGIT_LABELS)
DEFAULT_SEED = 42
IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
)


@dataclass(frozen=True, kw_only=True)
class RuntimeConfig:
    """Runtime settings for TensorFlow execution."""

    seed: int = DEFAULT_SEED
    enable_gpu_growth: bool = True


@dataclass(frozen=True, kw_only=True)
class DatasetConfig:
    """Shared image dataset settings."""

    image_size: tuple[int, int] = DIGIT_IMAGE_SIZE
    num_classes: int = NUM_DIGIT_CLASSES


@dataclass(frozen=True, kw_only=True)
class PredictionDefaults:
    """Default preprocessing and inference parameters."""

    preprocess_threshold: float = 0.18
    tta_samples: int = 30
    top_k: int = 5
