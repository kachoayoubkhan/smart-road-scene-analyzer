"""Classical computer vision lane detection with OpenCV."""

from __future__ import annotations

import cv2
import numpy as np


class LaneDetector:
    """Detect and overlay lane lines using Canny edges and Hough lines."""

    def detect_lanes(self, image: np.ndarray) -> tuple[np.ndarray, bool]:
        """Return an RGB image with lane overlay and a detection flag."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        cropped_edges = self.region_of_interest(edges)

        lines = cv2.HoughLinesP(
            cropped_edges,
            rho=2,
            theta=np.pi / 180,
            threshold=45,
            minLineLength=40,
            maxLineGap=120,
        )
        averaged_lines = self.average_slope_intercept(image, lines)
        lane_overlay = self.draw_lanes(image, averaged_lines)
        processed = cv2.addWeighted(image, 0.82, lane_overlay, 1.0, 0.0)
        return processed, bool(averaged_lines)

    def region_of_interest(self, edges: np.ndarray) -> np.ndarray:
        """Mask edges outside the likely road-lane region."""
        height, width = edges.shape[:2]
        polygon = np.array(
            [
                [
                    (int(width * 0.06), height),
                    (int(width * 0.44), int(height * 0.58)),
                    (int(width * 0.56), int(height * 0.58)),
                    (int(width * 0.94), height),
                ]
            ],
            dtype=np.int32,
        )

        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, polygon, 255)
        return cv2.bitwise_and(edges, mask)

    def average_slope_intercept(
        self,
        image: np.ndarray,
        lines: np.ndarray | None,
    ) -> list[tuple[int, int, int, int]]:
        """Average Hough line segments into one left and one right lane line."""
        if lines is None:
            return []

        left_lines: list[tuple[float, float]] = []
        right_lines: list[tuple[float, float]] = []

        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            if x1 == x2:
                continue

            slope, intercept = np.polyfit((x1, x2), (y1, y2), 1)
            if abs(slope) < 0.45 or abs(slope) > 2.5:
                continue

            if slope < 0:
                left_lines.append((float(slope), float(intercept)))
            else:
                right_lines.append((float(slope), float(intercept)))

        averaged_lines: list[tuple[int, int, int, int]] = []
        for lane_lines in (left_lines, right_lines):
            if not lane_lines:
                continue
            slope, intercept = np.average(lane_lines, axis=0)
            averaged_lines.append(self._make_coordinates(image, slope, intercept))

        return averaged_lines

    def draw_lanes(
        self,
        image: np.ndarray,
        lines: list[tuple[int, int, int, int]],
    ) -> np.ndarray:
        """Draw lane lines on a blank RGB image matching the input size."""
        line_image = np.zeros_like(image)
        for x1, y1, x2, y2 in lines:
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 209, 102), 10)
            cv2.line(line_image, (x1, y1), (x2, y2), (7, 59, 76), 3)
        return line_image

    @staticmethod
    def _make_coordinates(
        image: np.ndarray,
        slope: float,
        intercept: float,
    ) -> tuple[int, int, int, int]:
        height, width = image.shape[:2]
        y1 = height
        y2 = int(height * 0.62)
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        return (
            int(np.clip(x1, 0, width - 1)),
            y1,
            int(np.clip(x2, 0, width - 1)),
            y2,
        )
