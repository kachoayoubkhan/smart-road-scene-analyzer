"""Streamlit dashboard for Smart Road Scene Analyzer."""

from __future__ import annotations

from io import BytesIO
import tempfile
from pathlib import Path
from typing import Any

import streamlit as st
from PIL import Image

from config import (
    DEFAULT_CONFIDENCE,
    DEFAULT_YOLO_MODEL,
    FRAME_SKIP,
    MAX_VIDEO_FRAMES,
    POTHOLE_MODEL_PATH,
    ROAD_OBJECT_CLASSES,
    SUPPORTED_IMAGE_TYPES,
    SUPPORTED_VIDEO_TYPES,
)
from src.detector import RoadObjectDetector
from src.lane_detection import LaneDetector
from src.pothole_detection import PotholeDetector
from src.report_generator import generate_report
from src.risk_assessment import calculate_risk
from src.segmentation import RoadSegmenter
from src.utils import load_image, save_output_image
from src.video_processor import VideoProcessor


MODULE_LABELS = {
    "object_detection": "Object Detection",
    "lane_detection": "Lane Detection",
    "road_segmentation": "Road Segmentation",
    "pothole_detection": "Pothole Detection",
    "risk_assessment": "Risk Assessment",
}


@st.cache_resource(show_spinner=False)
def get_object_detector(model_name: str, confidence: float) -> RoadObjectDetector:
    """Cache YOLO object detector instances between Streamlit reruns."""
    return RoadObjectDetector(model_name=model_name, confidence=confidence)


@st.cache_resource(show_spinner=False)
def get_pothole_detector(model_path: str, confidence: float) -> PotholeDetector:
    """Cache optional pothole detector instances between Streamlit reruns."""
    return PotholeDetector(model_path=model_path, confidence=confidence)


@st.cache_resource(show_spinner=False)
def get_segmenter() -> RoadSegmenter:
    """Cache the lightweight road segmenter."""
    return RoadSegmenter(mode="classical")


def main() -> None:
    """Render and run the Streamlit application."""
    st.set_page_config(
        page_title="Smart Road Scene Analyzer",
        layout="wide",
    )
    _apply_page_style()

    st.title("Smart Road Scene Analyzer")
    st.caption("Deep learning and classical computer vision for road images and videos.")

    settings = _render_sidebar()
    image_tab, video_tab = st.tabs(["Image Analysis", "Video Analysis"])

    with image_tab:
        _render_image_workflow(settings)

    with video_tab:
        _render_video_workflow(settings)


def _render_sidebar() -> dict[str, Any]:
    st.sidebar.header("Settings")
    model_name = st.sidebar.text_input("YOLO model", value=DEFAULT_YOLO_MODEL)
    confidence = st.sidebar.slider(
        "Confidence threshold",
        min_value=0.10,
        max_value=0.95,
        value=float(DEFAULT_CONFIDENCE),
        step=0.05,
    )

    st.sidebar.subheader("Modules")
    selected_modules = {
        "object_detection": st.sidebar.checkbox("Object Detection", value=True),
        "lane_detection": st.sidebar.checkbox("Lane Detection", value=True),
        "road_segmentation": st.sidebar.checkbox("Road Segmentation", value=False),
        "pothole_detection": st.sidebar.checkbox("Pothole Detection", value=False),
        "risk_assessment": st.sidebar.checkbox("Risk Assessment", value=True),
    }

    st.sidebar.subheader("Video")
    frame_skip = st.sidebar.number_input(
        "Process every N frames",
        min_value=1,
        max_value=60,
        value=int(FRAME_SKIP),
        step=1,
    )
    max_frames = st.sidebar.number_input(
        "Maximum frames",
        min_value=10,
        max_value=5000,
        value=int(MAX_VIDEO_FRAMES),
        step=10,
    )

    if st.sidebar.button("Clear model cache"):
        get_object_detector.clear()
        get_pothole_detector.clear()
        get_segmenter.clear()
        st.sidebar.success("Model cache cleared.")

    return {
        "model_name": model_name,
        "confidence": confidence,
        "selected_modules": selected_modules,
        "frame_skip": int(frame_skip),
        "max_frames": int(max_frames),
    }


def _render_image_workflow(settings: dict[str, Any]) -> None:
    uploaded_image = st.file_uploader(
        "Upload a road image",
        type=list(SUPPORTED_IMAGE_TYPES),
    )
    if uploaded_image is None:
        st.info("Upload an image to start the road-scene analysis.")
        return

    image_rgb = load_image(uploaded_image)
    left_col, right_col = st.columns(2)
    with left_col:
        st.subheader("Original Image")
        st.image(image_rgb, use_container_width=True)

    if not st.button("Process Image", type="primary"):
        return

    with st.spinner("Analyzing image..."):
        processed_image, summary = _process_image(
            image_rgb=image_rgb,
            file_name=uploaded_image.name,
            settings=settings,
        )
        image_bytes = _image_to_jpeg_bytes(processed_image)
        try:
            output_path = save_output_image(processed_image, uploaded_image.name)
        except Exception as exc:
            output_path = None
            st.warning(
                "The processed image could not be saved to disk, but it is still "
                f"available for download from memory. Details: {exc}"
            )

    with right_col:
        st.subheader("Processed Output")
        st.image(processed_image, use_container_width=True)

    _render_summary(summary)
    _render_downloads(
        file_name=uploaded_image.name,
        selected_modules=_selected_module_names(settings["selected_modules"]),
        summary=summary,
        output_path=output_path,
        media_label="Download processed image",
        media_mime="image/jpeg",
        media_bytes=image_bytes,
    )


def _render_video_workflow(settings: dict[str, Any]) -> None:
    uploaded_video = st.file_uploader(
        "Upload a road video",
        type=list(SUPPORTED_VIDEO_TYPES),
    )
    if uploaded_video is None:
        st.info("Upload a video to process frames with the selected modules.")
        return

    st.video(uploaded_video)
    if not st.button("Process Video", type="primary"):
        return

    temp_path = _write_uploaded_video_to_temp(uploaded_video)
    progress_bar = st.progress(0.0, text="Starting video processing")

    selected_modules = settings["selected_modules"]
    detector = _load_object_detector_if_needed(settings)
    pothole_detector = _load_pothole_detector_if_needed(settings)
    lane_detector = LaneDetector() if selected_modules.get("lane_detection") else None
    segmenter = get_segmenter() if selected_modules.get("road_segmentation") else None

    try:
        with st.spinner("Processing video..."):
            processor = VideoProcessor()
            output_path, summary = processor.process_video(
                input_path=temp_path,
                selected_modules=selected_modules,
                frame_skip=settings["frame_skip"],
                max_frames=settings["max_frames"],
                detector=detector,
                lane_detector=lane_detector,
                segmenter=segmenter,
                pothole_detector=pothole_detector,
                progress_callback=lambda value: progress_bar.progress(
                    min(1.0, value),
                    text=f"Processing video: {int(value * 100)}%",
                ),
            )
    except Exception as exc:
        st.error(f"Video processing failed: {exc}")
        return
    finally:
        temp_path.unlink(missing_ok=True)

    progress_bar.progress(1.0, text="Video processing complete")
    st.subheader("Processed Video")
    st.video(str(output_path))
    _render_summary(summary)
    _render_downloads(
        file_name=uploaded_video.name,
        selected_modules=_selected_module_names(settings["selected_modules"]),
        summary=summary,
        output_path=output_path,
        media_label="Download processed video",
        media_mime="video/mp4",
    )


def _process_image(
    image_rgb,
    file_name: str,
    settings: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    selected_modules = settings["selected_modules"]
    processed = image_rgb.copy()
    object_counts: dict[str, int] = {}
    pothole_count = 0
    lane_detected: bool | None = None
    segmentation_status = "Not selected"

    if selected_modules.get("lane_detection"):
        processed, lane_detected = LaneDetector().detect_lanes(processed)

    if selected_modules.get("road_segmentation"):
        segmenter = get_segmenter()
        mask, segmentation_status = segmenter.segment_road(image_rgb)
        processed = segmenter.overlay_mask(processed, mask)

    if selected_modules.get("object_detection"):
        detector = _load_object_detector_if_needed(settings)
        if detector is not None:
            results = detector.detect(image_rgb)
            object_counts = detector.get_object_counts(results)
            processed = detector.draw_detections(processed, results)

    if selected_modules.get("pothole_detection"):
        pothole_detector = _load_pothole_detector_if_needed(settings)
        if pothole_detector is not None:
            pothole_results = pothole_detector.detect(image_rgb)
            pothole_count = pothole_detector.count_potholes(pothole_results)
            processed = pothole_detector.draw_potholes(processed, pothole_results)

    if selected_modules.get("risk_assessment"):
        risk_lane_flag = True if lane_detected is None else lane_detected
        risk_score, risk_level, explanation = calculate_risk(
            object_counts=object_counts,
            pothole_count=pothole_count,
            lane_detected=risk_lane_flag,
            segmentation_status=segmentation_status,
        )
    else:
        risk_score, risk_level, explanation = (
            0,
            "Not evaluated",
            ["Risk assessment was not selected."],
        )

    return processed, {
        "file_name": file_name,
        "object_counts": object_counts,
        "pothole_count": pothole_count,
        "lane_detected": lane_detected,
        "segmentation_status": segmentation_status,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "risk_explanation": explanation,
    }


def _load_object_detector_if_needed(settings: dict[str, Any]) -> RoadObjectDetector | None:
    if not settings["selected_modules"].get("object_detection"):
        return None

    detector = get_object_detector(
        model_name=settings["model_name"],
        confidence=float(settings["confidence"]),
    )
    if detector.error_message:
        st.warning(detector.error_message)
        return None
    return detector


def _load_pothole_detector_if_needed(settings: dict[str, Any]) -> PotholeDetector | None:
    if not settings["selected_modules"].get("pothole_detection"):
        return None

    detector = get_pothole_detector(
        model_path=str(POTHOLE_MODEL_PATH),
        confidence=float(settings["confidence"]),
    )
    if detector.error_message:
        st.warning(detector.error_message)
        return None
    return detector


def _render_summary(summary: dict[str, Any]) -> None:
    st.subheader("Analysis Summary")
    object_counts = summary.get("object_counts", {})
    total_objects = sum(int(value) for value in object_counts.values())
    pothole_count = int(summary.get("pothole_count", 0))
    lane_detected = summary.get("lane_detected")
    lane_status = _lane_status_text(lane_detected)
    risk_score = int(summary.get("risk_score", 0))
    risk_level = str(summary.get("risk_level", "Not evaluated"))

    metric_cols = st.columns(4)
    metric_cols[0].metric("Detected objects", total_objects)
    metric_cols[1].metric("Lane status", lane_status)
    metric_cols[2].metric("Potholes", pothole_count)
    metric_cols[3].metric("Risk level", risk_level, f"{risk_score}/100")

    st.progress(min(100, max(0, risk_score)) / 100)

    detail_cols = st.columns([1, 1])
    with detail_cols[0]:
        st.markdown("#### Object Counts")
        if object_counts:
            st.table(
                [
                    {"Object": label, "Count": object_counts.get(label, 0)}
                    for label in ROAD_OBJECT_CLASSES
                    if object_counts.get(label, 0) > 0
                ]
            )
        else:
            st.write("No target objects detected.")

    with detail_cols[1]:
        st.markdown("#### Risk Explanation")
        for item in summary.get("risk_explanation", []):
            st.write(f"- {item}")
        st.caption(str(summary.get("segmentation_status", "Not selected")))


def _render_downloads(
    file_name: str,
    selected_modules: list[str],
    summary: dict[str, Any],
    output_path: Path | None,
    media_label: str,
    media_mime: str,
    media_bytes: bytes | None = None,
) -> None:
    report_text = generate_report(
        file_name=file_name,
        selected_modules=selected_modules,
        object_counts=summary.get("object_counts", {}),
        pothole_count=int(summary.get("pothole_count", 0)),
        lane_detected=summary.get("lane_detected"),
        segmentation_status=str(summary.get("segmentation_status", "Not selected")),
        risk_score=int(summary.get("risk_score", 0)),
        risk_level=str(summary.get("risk_level", "Not evaluated")),
        explanation=summary.get("risk_explanation", []),
    )

    download_cols = st.columns(2)
    if media_bytes is None and output_path is not None and output_path.exists():
        try:
            media_bytes = output_path.read_bytes()
        except OSError as exc:
            st.warning(f"Processed media could not be read for download. Details: {exc}")

    if media_bytes is not None:
        download_cols[0].download_button(
            media_label,
            data=media_bytes,
            file_name=output_path.name if output_path is not None else _download_file_name(file_name, media_mime),
            mime=media_mime,
        )
    else:
        download_cols[0].warning("Processed media file is not available for download.")

    download_cols[1].download_button(
        "Download report",
        data=report_text,
        file_name=f"{Path(file_name).stem}_road_analysis_report.md",
        mime="text/markdown",
    )


def _download_file_name(file_name: str, media_mime: str) -> str:
    stem = Path(file_name).stem or "road_scene"
    if media_mime == "image/jpeg":
        return f"{stem}_processed.jpg"
    if media_mime == "video/mp4":
        return f"{stem}_processed.mp4"
    return f"{stem}_processed"


def _image_to_jpeg_bytes(image_rgb) -> bytes:
    """Encode a processed RGB image for download without depending on module reloads."""
    buffer = BytesIO()
    Image.fromarray(image_rgb).save(buffer, format="JPEG", quality=92)
    return buffer.getvalue()


def _selected_module_names(selected_modules: dict[str, bool]) -> list[str]:
    return [label for key, label in MODULE_LABELS.items() if selected_modules.get(key)]


def _lane_status_text(lane_detected: bool | None) -> str:
    if lane_detected is None:
        return "Not evaluated"
    return "Detected" if lane_detected else "Not detected"


def _write_uploaded_video_to_temp(uploaded_video) -> Path:
    suffix = Path(uploaded_video.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_video.getbuffer())
        return Path(temp_file.name)


def _apply_page_style() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        [data-testid="stMetric"] {
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 0.85rem 1rem;
            background: #ffffff;
        }
        div[data-testid="stDownloadButton"] button,
        div[data-testid="stButton"] button {
            border-radius: 8px;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
