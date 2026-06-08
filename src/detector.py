"""YOLO-based road object detection."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from config import DEFAULT_CONFIDENCE, DEFAULT_YOLO_MODEL, ROAD_OBJECT_CLASSES


class RoadObjectDetector:
    """Thin, failure-safe wrapper around Ultralytics YOLO detection."""

    def __init__(
        self,
        model_name: str = DEFAULT_YOLO_MODEL,
        confidence: float = DEFAULT_CONFIDENCE,
    ) -> None:
        self.model_name = model_name
        self.confidence = confidence
        self.model: Any | None = None
        self.class_names: dict[int, str] = {}
        self.error_message: str | None = None
        self.target_classes = set(ROAD_OBJECT_CLASSES)

        try:
            from ultralytics import YOLO
        except Exception as exc:  # pragma: no cover - depends on local install
            self.error_message = (
                "Ultralytics is not installed. Run `pip install -r requirements.txt` "
                f"and restart the app. Details: {exc}"
            )
            return

        try:
            self.model = YOLO(model_name)
            names = getattr(self.model, "names", {}) or {}
            if isinstance(names, dict):
                self.class_names = {int(key): str(value) for key, value in names.items()}
            else:
                self.class_names = {index: str(value) for index, value in enumerate(names)}
        except Exception as exc:  # pragma: no cover - model loading is environment-specific
            self.error_message = (
                f"Could not load YOLO model '{model_name}'. Check your internet "
                f"connection or place the model file locally. Details: {exc}"
            )

    @property
    def is_available(self) -> bool:
        """Return True when the model is ready for inference."""
        return self.model is not None and self.error_message is None

    def detect(self, image: np.ndarray) -> list[Any]:
        """Run object detection on an RGB image."""
        if not self.is_available:
            return []

        try:
            return self.model.predict(
                source=image,
                conf=self.confidence,
                verbose=False,
            )
        except Exception as exc:  # pragma: no cover - inference failures vary
            self.error_message = f"Object detection failed gracefully. Details: {exc}"
            return []

    def draw_detections(self, image: np.ndarray, results: list[Any]) -> np.ndarray:
        """Draw filtered road-object detections on a copy of the RGB image."""
        annotated = image.copy()
        thickness = max(2, int(round(min(image.shape[:2]) / 300)))

        for box, confidence, class_name in self._iter_filtered_detections(results):
            x1, y1, x2, y2 = [int(value) for value in box]
            color = self._color_for_class(class_name)
            label = f"{class_name} {confidence:.2f}"

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
            label_size, baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                2,
            )
            label_y = max(y1, label_size[1] + baseline + 6)
            cv2.rectangle(
                annotated,
                (x1, label_y - label_size[1] - baseline - 6),
                (x1 + label_size[0] + 8, label_y + baseline - 2),
                color,
                -1,
            )
            cv2.putText(
                annotated,
                label,
                (x1 + 4, label_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        return annotated

    def get_object_counts(self, results: list[Any]) -> dict[str, int]:
        """Count only the target road-scene classes."""
        counts = {class_name: 0 for class_name in ROAD_OBJECT_CLASSES}
        for _, _, class_name in self._iter_filtered_detections(results):
            counts[class_name] += 1
        return {class_name: count for class_name, count in counts.items() if count > 0}

    def _iter_filtered_detections(
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
                class_name = self.class_names.get(int(class_id), str(class_id))
                if class_name in self.target_classes:
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

    @staticmethod
    def _color_for_class(class_name: str) -> tuple[int, int, int]:
        """Return an RGB color for each class family."""
        if class_name == "person":
            return (230, 57, 70)
        if class_name in {"car", "bus", "truck", "motorcycle"}:
            return (42, 157, 143)
        if class_name == "bicycle":
            return (244, 162, 97)
        return (69, 123, 157)
