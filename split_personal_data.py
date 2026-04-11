import random
import shutil
from pathlib import Path

from project_paths import project_file


TRAIN_DIR = project_file("my_digits_new", "train")
VAL_DIR = project_file("my_digits_new", "val")
VAL_RATIO = 0.2
SEED = 42

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def list_image_files(directory: Path) -> list[Path]:
    return [
        file_path
        for file_path in directory.iterdir()
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def build_unique_destination(destination: Path) -> Path:
    if not destination.exists():
        return destination

    suffix = 1
    while True:
        candidate = destination.with_name(
            f"{destination.stem}_val{suffix}{destination.suffix}"
        )
        if not candidate.exists():
            return candidate
        suffix += 1


def main() -> None:
    random.seed(SEED)

    if not TRAIN_DIR.is_dir():
        raise FileNotFoundError(f"TRAIN_DIR not found: {TRAIN_DIR}")

    VAL_DIR.mkdir(parents=True, exist_ok=True)
    total_moved = 0

    for digit in map(str, range(10)):
        source_dir = TRAIN_DIR / digit
        destination_dir = VAL_DIR / digit

        if not source_dir.is_dir():
            print(f"[WARN] Missing class folder: {source_dir} (skip)")
            continue

        destination_dir.mkdir(parents=True, exist_ok=True)
        files = list_image_files(source_dir)

        if not files:
            print(f"[WARN] No images in: {source_dir}")
            continue

        random.shuffle(files)
        move_count = max(1, int(round(len(files) * VAL_RATIO))) if len(files) >= 5 else 1

        for file_path in files[:move_count]:
            destination_path = build_unique_destination(destination_dir / file_path.name)
            shutil.move(str(file_path), str(destination_path))
            total_moved += 1

        print(f"[{digit}] moved {move_count}/{len(files)} -> val")

    print(f"\nDone. Total moved: {total_moved}")
    print(f"Train: {TRAIN_DIR}")
    print(f"Val  : {VAL_DIR}")


if __name__ == "__main__":
    main()
