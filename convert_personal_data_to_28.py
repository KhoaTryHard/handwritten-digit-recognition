# Module nay chuyen anh viet tay goc sang dang 28x28 giong MNIST.
"""Convert personal handwritten images into 28x28 MNIST-like samples."""

from __future__ import annotations

from digit_pipeline.config import project_file
from digit_pipeline.preprocessing import convert_dataset_directory


SOURCE_DIR = project_file("my_digits_new", "val")
DESTINATION_DIR = project_file("my_digits_28", "val")
THRESHOLD = 0.22


def main() -> None:
    """Run the handwritten-image conversion pipeline."""
    converted_images = convert_dataset_directory(
        SOURCE_DIR,
        DESTINATION_DIR,
        threshold=THRESHOLD,
    )
    print(f"Converted {converted_images} image(s) to MNIST-like 28x28 format.")


if __name__ == "__main__":
    main()
