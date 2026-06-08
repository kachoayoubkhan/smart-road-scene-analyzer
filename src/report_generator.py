"""Generate downloadable road-scene analysis reports."""

from __future__ import annotations

from datetime import datetime
from typing import Iterable, Mapping


def generate_report(
    file_name: str,
    selected_modules: Iterable[str],
    object_counts: Mapping[str, int],
    pothole_count: int,
    lane_detected: bool | None,
    segmentation_status: str,
    risk_score: int,
    risk_level: str,
    explanation: Iterable[str],
) -> str:
    """Return a markdown report for download."""
    modules = ", ".join(selected_modules) if selected_modules else "None"
    counts = "\n".join(
        f"- {label}: {count}" for label, count in object_counts.items() if count > 0
    )
    if not counts:
        counts = "- No target objects detected"

    if lane_detected is None:
        lane_status = "Not evaluated"
    else:
        lane_status = "Detected" if lane_detected else "Not detected"

    explanation_text = "\n".join(f"- {item}" for item in explanation)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return f"""# Smart Road Scene Analyzer Report

Generated: {generated_at}
Input file: {file_name}

## Selected Modules

{modules}

## Detection Summary

{counts}

Pothole count: {pothole_count}
Lane status: {lane_status}
Segmentation status: {segmentation_status}

## Risk Assessment

Risk score: {risk_score}/100
Risk level: {risk_level}

## Risk Explanation

{explanation_text}
"""
