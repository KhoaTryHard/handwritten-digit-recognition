from project_paths import project_file

from digit_pipeline.preprocessing import convert_dataset_directory


SOURCE_DIR = project_file("my_digits_new", "val")
DESTINATION_DIR = project_file("my_digits_28", "val")
THRESHOLD = 0.22


def main() -> None:
    converted = convert_dataset_directory(
        SOURCE_DIR,
        DESTINATION_DIR,
        threshold=THRESHOLD,
    )
    print(f"Converted {converted} image(s) to MNIST-like 28x28 format.")


if __name__ == "__main__":
    main()
