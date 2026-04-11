from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageFilter

from digit_pipeline.data import IMAGE_EXTENSIONS, build_digit_augmenter


def predict_tta(
    model: tf.keras.Model,
    x: np.ndarray,
    *,
    num_samples: int = 20,
    augmenter: tf.keras.Sequential | None = None,
) -> np.ndarray:
    augmenter = augmenter or build_digit_augmenter()
    predictions = []

    for _ in range(num_samples):
        predictions.append(model.predict(augmenter(x, training=True), verbose=0)[0])

    return np.mean(predictions, axis=0)


def dilate(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    current = mask.copy()
    height, width = current.shape

    for _ in range(iterations):
        padded = np.pad(current, 1, mode="constant", constant_values=False)
        expanded = np.zeros_like(current, dtype=bool)
        for dy in range(3):
            for dx in range(3):
                expanded |= padded[dy : dy + height, dx : dx + width]
        current = expanded

    return current


def erode(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    current = mask.copy()
    height, width = current.shape

    for _ in range(iterations):
        padded = np.pad(current, 1, mode="constant", constant_values=True)
        reduced = np.ones_like(current, dtype=bool)
        for dy in range(3):
            for dx in range(3):
                reduced &= padded[dy : dy + height, dx : dx + width]
        current = reduced

    return current


def keep_large_components(
    mask: np.ndarray,
    *,
    min_pixels: int = 25,
    keep_ratio: float = 0.25,
) -> np.ndarray:
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for row in range(height):
        for col in range(width):
            if mask[row, col] and not visited[row, col]:
                stack = [(row, col)]
                visited[row, col] = True
                component: list[tuple[int, int]] = []

                while stack:
                    current_row, current_col = stack.pop()
                    component.append((current_row, current_col))

                    for row_offset, col_offset in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        next_row = current_row + row_offset
                        next_col = current_col + col_offset

                        if (
                            0 <= next_row < height
                            and 0 <= next_col < width
                            and mask[next_row, next_col]
                            and not visited[next_row, next_col]
                        ):
                            visited[next_row, next_col] = True
                            stack.append((next_row, next_col))

                components.append(component)

    if not components:
        return mask

    sizes = np.array([len(component) for component in components], dtype=int)
    threshold = max(min_pixels, int(sizes.max() * keep_ratio))
    output = np.zeros_like(mask, dtype=bool)

    for component, size in zip(components, sizes):
        if size >= threshold:
            rows, cols = zip(*component)
            output[np.array(rows), np.array(cols)] = True

    return output


def shift_image(img: np.ndarray, shift_x: int, shift_y: int) -> np.ndarray:
    height, width = img.shape
    shifted = np.zeros_like(img)

    src_x0 = max(0, -shift_x)
    src_x1 = min(width, width - shift_x)
    dst_x0 = max(0, shift_x)
    dst_x1 = min(width, width + shift_x)

    src_y0 = max(0, -shift_y)
    src_y1 = min(height, height - shift_y)
    dst_y0 = max(0, shift_y)
    dst_y1 = min(height, height + shift_y)

    shifted[dst_y0:dst_y1, dst_x0:dst_x1] = img[src_y0:src_y1, src_x0:src_x1]
    return shifted


def preprocess_handwritten_mnist_like(
    image_path: str | Path,
    *,
    threshold: float = 0.22,
) -> tuple[np.ndarray, Image.Image]:
    with Image.open(image_path) as raw_image:
        image = raw_image.convert("RGBA")

    background = Image.new("RGBA", image.size, (255, 255, 255, 255))
    grayscale = Image.alpha_composite(background, image).convert("L")
    pixel_array = np.array(grayscale).astype(np.uint8)

    if pixel_array.mean() > 127:
        pixel_array = 255 - pixel_array

    normalized = pixel_array.astype("float32") / 255.0
    mask = normalized > threshold
    mask = dilate(mask, iterations=1)
    mask = erode(mask, iterations=1)
    mask = keep_large_components(mask, min_pixels=25, keep_ratio=0.20)

    if mask.sum() == 0:
        canvas = Image.new("L", (28, 28), 0)
        x = (np.array(canvas).astype("float32") / 255.0).reshape(1, 28, 28, 1)
        return x, canvas

    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    cropped = (pixel_array * mask.astype(np.uint8))[y0:y1, x0:x1]

    digit = Image.fromarray(cropped)
    width, height = digit.size
    scale = 20.0 / max(width, height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    digit = digit.resize((resized_width, resized_height), Image.Resampling.LANCZOS)

    canvas = Image.new("L", (28, 28), 0)
    left = (28 - resized_width) // 2
    top = (28 - resized_height) // 2
    canvas.paste(digit, (left, top))

    shifted = np.array(canvas).astype(np.uint8)
    weights = shifted.astype("float32")
    total_weight = weights.sum()

    if total_weight > 0:
        yy, xx = np.indices(shifted.shape)
        center_y = (yy * weights).sum() / total_weight
        center_x = (xx * weights).sum() / total_weight
        shifted = shift_image(
            shifted,
            int(round(13.5 - center_x)),
            int(round(13.5 - center_y)),
        )
        canvas = Image.fromarray(shifted)

    canvas = canvas.filter(ImageFilter.GaussianBlur(radius=0.4))
    x = (np.array(canvas).astype("float32") / 255.0).reshape(1, 28, 28, 1)
    return x, canvas


def convert_dataset_directory(
    source_dir: str | Path,
    destination_dir: str | Path,
    *,
    threshold: float = 0.22,
) -> int:
    source_root = Path(source_dir)
    destination_root = Path(destination_dir)
    destination_root.mkdir(parents=True, exist_ok=True)

    converted = 0

    for digit in map(str, range(10)):
        source_digit_dir = source_root / digit
        destination_digit_dir = destination_root / digit
        destination_digit_dir.mkdir(parents=True, exist_ok=True)

        if not source_digit_dir.is_dir():
            print(f"Missing folder: {source_digit_dir}")
            continue

        for image_path in sorted(source_digit_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            _, canvas = preprocess_handwritten_mnist_like(
                image_path,
                threshold=threshold,
            )
            canvas.save(destination_digit_dir / image_path.name)
            converted += 1

    return converted
