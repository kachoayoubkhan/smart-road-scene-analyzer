"""Road-region segmentation helpers."""

from __future__ import annotations

import cv2
import numpy as np


class RoadSegmenter:
    """Approximate drivable road regions with a lightweight classical mask."""

    def __init__(self, mode: str = "classical") -> None:
        self.mode = mode

    def segment_road(self, image: np.ndarray) -> tuple[np.ndarray, str]:
        """Return a binary road mask and a status message."""
        if self.mode != "classical":
            return (
                np.zeros(image.shape[:2], dtype=np.uint8),
                "Deep learning road segmentation is unavailable. Use the classical mode or add a model later.",
            )

        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        _, saturation, value = cv2.split(hsv)

        # Roads are often low-saturation surfaces in the lower part of the frame.
        low_saturation = saturation < 95
        usable_brightness = (value > 35) & (value < 235)
        road_like = (low_saturation & usable_brightness).astype(np.uint8) * 255

        lower_region = self._lower_road_region(image.shape[:2])
        mask = cv2.bitwise_and(road_like, lower_region)

        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        area_ratio = float(np.count_nonzero(mask)) / float(mask.size)
        if area_ratio < 0.025:
            return (
                np.zeros(image.shape[:2], dtype=np.uint8),
                "Road segmentation unavailable: the classical heuristic could not isolate the road region.",
            )

        return mask, "Classical road segmentation generated a road-region approximation."

    def overlay_mask(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Overlay a translucent road mask on an RGB image."""
        if mask is None or np.count_nonzero(mask) == 0:
            return image.copy()

        overlay = image.copy()
        road_color = np.zeros_like(image)
        road_color[:, :] = (17, 138, 178)
        alpha_mask = mask.astype(bool)
        overlay[alpha_mask] = cv2.addWeighted(
            image[alpha_mask],
            0.55,
            road_color[alpha_mask],
            0.45,
            0.0,
        )
        return overlay

    @staticmethod
    def _lower_road_region(shape: tuple[int, int]) -> np.ndarray:
        height, width = shape
        polygon = np.array(
            [
                [
                    (0, height),
                    (int(width * 0.2), int(height * 0.52)),
                    (int(width * 0.8), int(height * 0.52)),
                    (width, height),
                ]
            ],
            dtype=np.int32,
        )
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, polygon, 255)
        return mask
