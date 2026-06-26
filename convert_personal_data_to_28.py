# Module nay chuyen anh viet tay goc sang dang 28x28 giong MNIST.
"""Chuyen anh viet tay ca nhan thanh mau 28x28 giong MNIST."""

from __future__ import annotations

from digit_pipeline.config import project_file
from digit_pipeline.preprocessing import convert_dataset_directory


SOURCE_DIR = project_file("my_digits_new", "val")
DESTINATION_DIR = project_file("my_digits_28", "val")
THRESHOLD = 0.22


def main() -> None:
    """Chay pipeline chuyen doi anh viet tay."""
    converted_images = convert_dataset_directory(
        SOURCE_DIR,
        DESTINATION_DIR,
        threshold=THRESHOLD,
    )
    print(f"Converted {converted_images} image(s) to MNIST-like 28x28 format.")


if __name__ == "__main__":
    main()
