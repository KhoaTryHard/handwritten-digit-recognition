# Module nay hien thi nhanh cac mau MNIST cua mot chu so de so sanh truc quan.
"""Preview MNIST samples for a chosen digit."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from digit_pipeline.data_loading import load_mnist_arrays


TARGET_DIGIT = 9
MAX_IMAGES = 25


def main() -> None:
    """Display a grid of MNIST samples for one digit."""
    (_, _), (test_images, test_labels) = load_mnist_arrays()
    digit_indices = np.where(test_labels == TARGET_DIGIT)[0][:MAX_IMAGES]

    # Ve luoi 5x5 de quan sat nhanh hinh dang chu so trong MNIST.
    plt.figure(figsize=(6, 6))
    for plot_index, sample_index in enumerate(digit_indices, start=1):
        plt.subplot(5, 5, plot_index)
        plt.imshow(test_images[sample_index], cmap="gray")
        plt.axis("off")

    plt.suptitle(f"MNIST examples of digit {TARGET_DIGIT}")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
