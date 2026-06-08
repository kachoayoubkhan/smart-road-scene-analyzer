"""Utility helpers for image loading, conversion, validation, and saving."""

from __future__ import annotations

from io import BytesIO
from datetime import datetime
from pathlib import Path
import re
from typing import Iterable

import cv2
import numpy as np
from PIL import Image

from config import OUTPUT_DIR


def ensure_output_dir(output_dir: Path = OUTPUT_DIR) -> Path:
    """Create and return the output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_image(uploaded_file) -> np.ndarray:
    """Load a Streamlit uploaded image as an RGB NumPy array."""
    image = Image.open(uploaded_file).convert("RGB")
    return np.array(image)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert a BGR OpenCV image to RGB."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    """Convert an RGB image to BGR for OpenCV writing and video output."""
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def validate_file_type(file_name: str, supported_types: Iterable[str]) -> bool:
    """Return True when the file extension is supported."""
    suffix = Path(file_name).suffix.lower().lstrip(".")
    return suffix in {item.lower().lstrip(".") for item in supported_types}


def save_output_image(
    image_rgb: np.ndarray,
    original_name: str = "road_scene",
    output_dir: Path = OUTPUT_DIR,
) -> Path:
    """Save an RGB output image and return the created path."""
    ensure_output_dir(output_dir)
    safe_stem = _safe_file_stem(original_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"{safe_stem}_{timestamp}.jpg"

    image_rgb = _as_uint8_rgb(image_rgb)
    saved = cv2.imwrite(str(output_path), rgb_to_bgr(image_rgb))
    if not saved or not output_path.exists() or output_path.stat().st_size == 0:
        Image.fromarray(image_rgb).save(output_path, format="JPEG", quality=92)

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(f"Could not save processed image to {output_path}")

    return output_path


def image_to_jpeg_bytes(image_rgb: np.ndarray, quality: int = 92) -> bytes:
    """Encode an RGB image as JPEG bytes for Streamlit downloads."""
    buffer = BytesIO()
    Image.fromarray(_as_uint8_rgb(image_rgb)).save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def empty_counts(labels: Iterable[str]) -> dict[str, int]:
    """Create a zero-filled count dictionary for display stability."""
    return {label: 0 for label in labels}


def _safe_file_stem(file_name: str) -> str:
    stem = Path(file_name).stem or "road_scene"
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._-")
    return stem or "road_scene"


def _as_uint8_rgb(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    return np.clip(image, 0, 255).astype(np.uint8)
