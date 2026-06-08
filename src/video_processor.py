"""Frame-by-frame road-scene video processing."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2

from config import FRAME_SKIP, MAX_VIDEO_FRAMES, OUTPUT_DIR
from src.risk_assessment import calculate_risk
from src.utils import bgr_to_rgb, ensure_output_dir, rgb_to_bgr


class VideoProcessor:
    """Apply selected road-analysis modules to uploaded videos."""

    def __init__(self, output_dir: Path = OUTPUT_DIR) -> None:
        self.output_dir = ensure_output_dir(output_dir)

    def process_video(
        self,
        input_path: str | Path,
        selected_modules: Mapping[str, bool],
        frame_skip: int = FRAME_SKIP,
        max_frames: int = MAX_VIDEO_FRAMES,
        detector: Any | None = None,
        lane_detector: Any | None = None,
        segmenter: Any | None = None,
        pothole_detector: Any | None = None,
        progress_callback: Callable[[float], None] | None = None,
    ) -> tuple[Path, dict[str, Any]]:
        """Process a video and return the output path plus aggregate summary."""
        input_path = Path(input_path)
        capture = cv2.VideoCapture(str(input_path))
        if not capture.isOpened():
            raise RuntimeError("Could not open the uploaded video file.")

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = capture.get(cv2.CAP_PROP_FPS) or 20.0
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or max_frames)
        frame_limit = max(1, min(max_frames, total_frames if total_frames > 0 else max_frames))

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"{input_path.stem}_processed_{timestamp}.mp4"
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            capture.release()
            raise RuntimeError("Could not create the processed video writer.")

        object_counts: dict[str, int] = {}
        pothole_count = 0
        lane_frames = 0
        processed_frames = 0
        segmentation_status = "Not selected"

        try:
            for frame_index in range(frame_limit):
                ok, frame_bgr = capture.read()
                if not ok:
                    break

                frame_rgb = bgr_to_rgb(frame_bgr)
                annotated = frame_rgb.copy()
                should_process = frame_index % max(1, frame_skip) == 0

                if should_process:
                    processed_frames += 1

                    if selected_modules.get("lane_detection") and lane_detector is not None:
                        annotated, lane_detected = lane_detector.detect_lanes(annotated)
                        if lane_detected:
                            lane_frames += 1

                    if selected_modules.get("road_segmentation") and segmenter is not None:
                        mask, segmentation_status = segmenter.segment_road(frame_rgb)
                        annotated = segmenter.overlay_mask(annotated, mask)

                    if selected_modules.get("object_detection") and detector is not None:
                        results = detector.detect(frame_rgb)
                        frame_counts = detector.get_object_counts(results)
                        for label, count in frame_counts.items():
                            object_counts[label] = object_counts.get(label, 0) + int(count)
                        annotated = detector.draw_detections(annotated, results)

                    if selected_modules.get("pothole_detection") and pothole_detector is not None:
                        results = pothole_detector.detect(frame_rgb)
                        pothole_count += pothole_detector.count_potholes(results)
                        annotated = pothole_detector.draw_potholes(annotated, results)

                writer.write(rgb_to_bgr(annotated))

                if progress_callback is not None:
                    progress_callback((frame_index + 1) / frame_limit)
        finally:
            capture.release()
            writer.release()

        lane_selected = bool(selected_modules.get("lane_detection"))
        lane_detected: bool | None = None
        if lane_selected:
            lane_detected = lane_frames > 0

        if not selected_modules.get("road_segmentation"):
            segmentation_status = "Not selected"

        risk_lane_flag = True if lane_detected is None else lane_detected
        risk_score, risk_level, explanation = calculate_risk(
            object_counts=object_counts,
            pothole_count=pothole_count,
            lane_detected=risk_lane_flag,
            segmentation_status=segmentation_status,
        )

        summary: dict[str, Any] = {
            "object_counts": object_counts,
            "pothole_count": pothole_count,
            "lane_detected": lane_detected,
            "lane_frames": lane_frames,
            "processed_frames": processed_frames,
            "segmentation_status": segmentation_status,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "risk_explanation": explanation,
        }
        return output_path, summary
