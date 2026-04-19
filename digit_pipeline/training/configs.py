# Module nay chua dataclass cau hinh cho train base model va cac stage fine-tune.
"""Dataclass-based training configuration objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from digit_pipeline.config.settings import DEFAULT_SEED, DIGIT_IMAGE_SIZE


@dataclass(frozen=True, kw_only=True)
class CallbackConfig:
    """Shared callback settings for fine-tuning stages."""

    early_stopping_patience: int = 5
    plateau_factor: float = 0.5
    plateau_patience: int = 2
    min_learning_rate: float = 1e-6


@dataclass(frozen=True, kw_only=True)
class MnistTrainingConfig:
    """Configuration for stage 1 MNIST training."""

    output_model_path: Path
    batch_size: int = 64
    epochs: int = 5
    learning_rate: float = 1e-3
    seed: int = DEFAULT_SEED


@dataclass(frozen=True, kw_only=True)
class TransferLearningConfig:
    """Base configuration for loading and fine-tuning an existing model."""

    input_model_path: Path
    output_model_path: Path
    batch_size: int = 64
    epochs: int = 20
    learning_rate: float = 1e-4
    seed: int = DEFAULT_SEED
    callback_config: CallbackConfig = field(default_factory=CallbackConfig)


@dataclass(frozen=True, kw_only=True)
class EmnistTrainingConfig(TransferLearningConfig):
    """Configuration for stage 2 EMNIST fine-tuning."""

    rotation: float = 0.10
    translation: float = 0.15
    zoom: float = 0.12


@dataclass(frozen=True, kw_only=True)
class DirectoryFineTuneConfig(TransferLearningConfig):
    """Configuration for stage 3 directory-based fine-tuning."""

    train_dir: Path
    val_dir: Path
    image_size: tuple[int, int] = DIGIT_IMAGE_SIZE
    cache_in_memory: bool = False
