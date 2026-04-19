# Module nay cau hinh seed va GPU memory growth cho TensorFlow.
"""Runtime helpers for TensorFlow execution."""

from __future__ import annotations

import tensorflow as tf

from digit_pipeline.config.settings import RuntimeConfig


def configure_runtime(config: RuntimeConfig | None = None) -> RuntimeConfig:
    """Configure TensorFlow seed handling and GPU memory growth."""
    runtime_config = config or RuntimeConfig()
    tf.keras.utils.set_random_seed(runtime_config.seed)

    # Bat memory growth de TensorFlow khong chiem het VRAM ngay tu dau.
    if runtime_config.enable_gpu_growth:
        for gpu_device in tf.config.list_physical_devices("GPU"):
            try:
                tf.config.experimental.set_memory_growth(gpu_device, True)
            except RuntimeError:
                continue

    return runtime_config
