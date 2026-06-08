"""Optional custom YOLO pothole detection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from config import DEFAULT_CONFIDENCE, POTHOLE_MODEL_PATH


class PotholeDetector:
    """Load and run a custom pothole model when weights are available."""

    def __init__(
        self,
        model_path: str | Path = POTHOLE_MODEL_PATH,
        confidence: float = DEFAULT_CONFIDENCE,
    ) -> None:
        self.model_path = Path(model_path)
        self.confidence = confidence
        self.model: Any | None = None
        self.class_names: dict[int, str] = {}
        self.error_message: str | None = None

        if not self.model_path.exists():
            self.error_message = (
                "Custom pothole model not found. Add your trained YOLO pothole "
                "model inside the models folder."
            )
            return

        try:
            from ultralytics import YOLO
        except Exception as exc:  # pragma: no cover - depends on local install
            self.error_message = (
                "Ultralytics is not installed. Run `pip install -r requirements.txt` "
                f"and restart the app. Details: {exc}"
            )
            return

        try:
            self.model = YOLO(str(self.model_path))
            names = getattr(self.model, "names", {}) or {}
            if isinstance(names, dict):
                self.class_names = {int(key): str(value) for key, value in names.items()}
            else:
                self.class_names = {index: str(value) for index, value in enumerate(names)}
        except Exception as exc:  # pragma: no cover - model loading is environment-specific
            self.error_message = f"Could not load pothole model. Details: {exc}"

    @property
    def is_available(self) -> bool:
        """Return True when the custom pothole model is ready."""
        return self.model is not None and self.error_message is None

    def detect(self, image: np.ndarray) -> list[Any]:
        """Run pothole detection on an RGB image."""
        if not self.is_available:
            return []
        try:
            return self.model.predict(
                source=image,
                conf=self.confidence,
                verbose=False,
            )
        except Exception as exc:  # pragma: no cover - inference failures vary
            self.error_message = f"Pothole detection failed gracefully. Details: {exc}"
            return []

    def draw_potholes(self, image: np.ndarray, results: list[Any]) -> np.ndarray:
        """Draw pothole detections on a copy of the RGB image."""
        annotated = image.copy()
        for box, confidence, class_name in self._iter_detections(results):
            x1, y1, x2, y2 = [int(value) for value in box]
            label = f"{class_name} {confidence:.2f}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (214, 40, 40), 3)
            cv2.putText(
                annotated,
                label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (214, 40, 40),
                2,
                cv2.LINE_AA,
            )
        return annotated

    def count_potholes(self, results: list[Any]) -> int:
        """Count pothole boxes from model results."""
        return len(self._iter_detections(results))

    def _iter_detections(
        self,
        results: list[Any],
    ) -> list[tuple[np.ndarray, float, str]]:
        detections: list[tuple[np.ndarray, float, str]] = []
        for result in results or []:
            boxes = getattr(result, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue

            xyxy = self._to_numpy(boxes.xyxy)
            confidences = self._to_numpy(boxes.conf)
            class_ids = self._to_numpy(boxes.cls).astype(int)

            for box, confidence, class_id in zip(xyxy, confidences, class_ids):
                class_name = self.class_names.get(int(class_id), "pothole")
                detections.append((box, float(confidence), class_name))
        return detections

    @staticmethod
    def _to_numpy(values: Any) -> np.ndarray:
        if hasattr(values, "detach"):
            values = values.detach()
        if hasattr(values, "cpu"):
            values = values.cpu()
        if hasattr(values, "numpy"):
            return values.numpy()
        return np.asarray(values)
