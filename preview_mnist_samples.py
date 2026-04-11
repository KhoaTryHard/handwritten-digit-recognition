import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import datasets


TARGET_DIGIT = 9
MAX_IMAGES = 25


def main() -> None:
    (_, _), (test_images, test_labels) = datasets.mnist.load_data()
    indices = np.where(test_labels == TARGET_DIGIT)[0][:MAX_IMAGES]

    plt.figure(figsize=(6, 6))
    for plot_index, sample_index in enumerate(indices, start=1):
        plt.subplot(5, 5, plot_index)
        plt.imshow(test_images[sample_index], cmap="gray")
        plt.axis("off")

    plt.suptitle(f"MNIST examples of digit {TARGET_DIGIT}")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
