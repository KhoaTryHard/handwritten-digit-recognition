# Module nay lam sach anh viet tay va dua ve dang 28x28 giong MNIST.
"""Image preprocessing for handwritten digit normalization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from digit_pipeline.config.settings import DIGIT_LABELS
from digit_pipeline.utils import list_image_files


@dataclass(frozen=True)
class ProcessedDigitImage:
    """A normalized tensor paired with a preview image."""

    tensor: np.ndarray
    preview: Image.Image


def dilate_mask(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Dilate a binary mask using a 3x3 neighborhood."""
    current_mask = mask.copy()
    height, width = current_mask.shape

    for _ in range(iterations):
        padded_mask = np.pad(current_mask, 1, mode="constant", constant_values=False)
        dilated_mask = np.zeros_like(current_mask, dtype=bool)
        for row_offset in range(3):
            for col_offset in range(3):
                dilated_mask |= padded_mask[
                    row_offset : row_offset + height,
                    col_offset : col_offset + width,
                ]
        current_mask = dilated_mask

    return current_mask


def erode_mask(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Erode a binary mask using a 3x3 neighborhood."""
    current_mask = mask.copy()
    height, width = current_mask.shape

    for _ in range(iterations):
        padded_mask = np.pad(current_mask, 1, mode="constant", constant_values=True)
        eroded_mask = np.ones_like(current_mask, dtype=bool)
        for row_offset in range(3):
            for col_offset in range(3):
                eroded_mask &= padded_mask[
                    row_offset : row_offset + height,
                    col_offset : col_offset + width,
                ]
        current_mask = eroded_mask

    return current_mask


def keep_large_components(
    mask: np.ndarray,
    *,
    min_pixels: int = 25,
    keep_ratio: float = 0.25,
) -> np.ndarray:
    """Keep connected components that are large enough to be a digit."""
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for row_index in range(height):
        for col_index in range(width):
            if not mask[row_index, col_index] or visited[row_index, col_index]:
                continue

            component_stack = [(row_index, col_index)]
            visited[row_index, col_index] = True
            component: list[tuple[int, int]] = []

            while component_stack:
                current_row, current_col = component_stack.pop()
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
                        component_stack.append((next_row, next_col))

            components.append(component)

    if not components:
        return mask

    component_sizes = np.array([len(component) for component in components], dtype=int)
    size_threshold = max(min_pixels, int(component_sizes.max() * keep_ratio))
    filtered_mask = np.zeros_like(mask, dtype=bool)

    for component, size in zip(components, component_sizes):
        if size < size_threshold:
            continue
        rows, cols = zip(*component)
        filtered_mask[np.array(rows), np.array(cols)] = True

    return filtered_mask


def shift_image(image: np.ndarray, shift_x: int, shift_y: int) -> np.ndarray:
    """Shift an image on a zero-padded canvas."""
    height, width = image.shape
    shifted_image = np.zeros_like(image)

    src_x0 = max(0, -shift_x)
    src_x1 = min(width, width - shift_x)
    dst_x0 = max(0, shift_x)
    dst_x1 = min(width, width + shift_x)

    src_y0 = max(0, -shift_y)
    src_y1 = min(height, height - shift_y)
    dst_y0 = max(0, shift_y)
    dst_y1 = min(height, height + shift_y)

    shifted_image[dst_y0:dst_y1, dst_x0:dst_x1] = image[
        src_y0:src_y1,
        src_x0:src_x1,
    ]
    return shifted_image


def preprocess_handwritten_image(
    image_path: str | Path,
    *,
    threshold: float = 0.22,
) -> ProcessedDigitImage:
    """Convert a handwritten image into a normalized MNIST-like tensor."""
    with Image.open(image_path) as raw_image:
        rgba_image = ImageOps.exif_transpose(raw_image).convert("RGBA")

    # Tach nen va dao cuc neu anh dang chu den tren nen sang.
    background = Image.new("RGBA", rgba_image.size, (255, 255, 255, 255))
    grayscale_image = Image.alpha_composite(background, rgba_image).convert("L")
    pixel_array = np.array(grayscale_image, dtype=np.uint8)
    if float(pixel_array.mean()) > 127.0:
        pixel_array = 255 - pixel_array

    normalized_pixels = pixel_array.astype("float32") / 255.0

    # Lam sach mask de giu lai phan net chu so lon nhat.
    digit_mask = normalized_pixels > threshold
    digit_mask = dilate_mask(digit_mask, iterations=1)
    digit_mask = erode_mask(digit_mask, iterations=1)
    digit_mask = keep_large_components(digit_mask, min_pixels=25, keep_ratio=0.20)

    if int(digit_mask.sum()) == 0:
        empty_canvas = Image.new("L", (28, 28), 0)
        empty_tensor = (
            np.array(empty_canvas, dtype=np.float32).reshape(1, 28, 28, 1) / 255.0
        )
        return ProcessedDigitImage(tensor=empty_tensor, preview=empty_canvas)

    # Cat vung chu so, resize ve 20x20 roi dat vao canvas 28x28.
    y_indices, x_indices = np.where(digit_mask)
    top, bottom = y_indices.min(), y_indices.max() + 1
    left, right = x_indices.min(), x_indices.max() + 1
    cropped_digit = (pixel_array * digit_mask.astype(np.uint8))[top:bottom, left:right]

    digit_image = Image.fromarray(cropped_digit)
    digit_width, digit_height = digit_image.size
    scale = 20.0 / max(digit_width, digit_height)
    resized_width = max(1, int(round(digit_width * scale)))
    resized_height = max(1, int(round(digit_height * scale)))
    digit_image = digit_image.resize(
        (resized_width, resized_height),
        Image.Resampling.LANCZOS,
    )

    preview_canvas = Image.new("L", (28, 28), 0)
    paste_left = (28 - resized_width) // 2
    paste_top = (28 - resized_height) // 2
    preview_canvas.paste(digit_image, (paste_left, paste_top))

    # Dich anh theo tam khoi luong de ket qua gan voi phan bo MNIST.
    centered_pixels = np.array(preview_canvas, dtype=np.uint8)
    weights = centered_pixels.astype("float32")
    total_weight = float(weights.sum())
    if total_weight > 0.0:
        yy, xx = np.indices(centered_pixels.shape)
        center_y = float((yy * weights).sum() / total_weight)
        center_x = float((xx * weights).sum() / total_weight)
        centered_pixels = shift_image(
            centered_pixels,
            int(round(13.5 - center_x)),
            int(round(13.5 - center_y)),
        )
        preview_canvas = Image.fromarray(centered_pixels)

    preview_canvas = preview_canvas.filter(ImageFilter.GaussianBlur(radius=0.4))
    preview_tensor = (
        np.array(preview_canvas, dtype=np.float32).reshape(1, 28, 28, 1) / 255.0
    )
    return ProcessedDigitImage(tensor=preview_tensor, preview=preview_canvas)


def convert_dataset_directory(
    source_dir: str | Path,
    destination_dir: str | Path,
    *,
    threshold: float = 0.22,
) -> int:
    """Convert a class-organized image directory into 28x28 MNIST-like images."""
    source_root = Path(source_dir)
    destination_root = Path(destination_dir)
    destination_root.mkdir(parents=True, exist_ok=True)
    converted_images = 0

    # Duyet tung thu muc lop va luu lai anh da chuan hoa theo ten goc.
    for digit_label in DIGIT_LABELS:
        source_digit_dir = source_root / digit_label
        destination_digit_dir = destination_root / digit_label
        destination_digit_dir.mkdir(parents=True, exist_ok=True)

        if not source_digit_dir.is_dir():
            continue

        for image_path in list_image_files(source_digit_dir):
            processed_image = preprocess_handwritten_image(
                image_path,
                threshold=threshold,
            )
            processed_image.preview.save(destination_digit_dir / image_path.name)
            converted_images += 1

    return converted_images
