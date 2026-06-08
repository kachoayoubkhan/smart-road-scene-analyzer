"""Rule-based road-scene risk scoring."""

from __future__ import annotations

from typing import Mapping


def calculate_risk(
    object_counts: Mapping[str, int],
    pothole_count: int,
    lane_detected: bool,
    segmentation_status: str,
) -> tuple[int, str, list[str]]:
    """Calculate a 0-100 risk score, label, and human-readable explanations."""
    score = 0
    explanation: list[str] = []

    pedestrian_count = int(object_counts.get("person", 0))
    vehicle_count = sum(
        int(object_counts.get(label, 0))
        for label in ("car", "motorcycle", "bus", "truck")
    )
    bike_count = int(object_counts.get("bicycle", 0))
    heavy_vehicle_count = int(object_counts.get("bus", 0)) + int(object_counts.get("truck", 0))

    if pedestrian_count > 0 and vehicle_count > 0:
        score += 25
        explanation.append("Pedestrians and moving vehicles appear in the same scene.")
    elif pedestrian_count > 0:
        score += 12
        explanation.append("Pedestrians are present near the road area.")

    if pothole_count > 0:
        pothole_risk = min(35, 15 + pothole_count * 5)
        score += pothole_risk
        explanation.append(f"{pothole_count} pothole or road-damage region(s) were detected.")

    if not lane_detected:
        score += 20
        explanation.append("Lane markings were not confidently detected.")

    if vehicle_count >= 6:
        score += 25
        explanation.append("Traffic density appears high.")
    elif vehicle_count >= 3:
        score += 15
        explanation.append("Multiple vehicles are present.")
    elif vehicle_count >= 1:
        score += 6
        explanation.append("At least one vehicle is present.")

    if heavy_vehicle_count > 0:
        score += 12
        explanation.append("A bus or truck increases stopping-distance and visibility risk.")

    if bike_count > 0 and vehicle_count > 0:
        score += 8
        explanation.append("Bicycles or motorcycles near vehicles increase interaction risk.")

    status_lower = (segmentation_status or "").lower()
    if "unavailable" in status_lower or "not selected" in status_lower:
        score += 5
        explanation.append("Road segmentation was unavailable, so the score includes uncertainty.")

    score = max(0, min(100, int(score)))

    if score <= 35:
        risk_level = "Low Risk"
    elif score <= 70:
        risk_level = "Moderate Risk"
    else:
        risk_level = "High Risk"

    if not explanation:
        explanation.append("No major hazards were detected by the enabled modules.")

    return score, risk_level, explanation
