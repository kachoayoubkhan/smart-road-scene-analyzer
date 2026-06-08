# Methodology

## YOLO Object Detection

YOLO is a convolutional neural network detector that predicts object classes and bounding boxes in one forward pass. This project uses an Ultralytics YOLO model for common COCO road-scene classes such as person, bicycle, car, motorcycle, bus, truck, traffic light, and stop sign.

The detector is wrapped in `RoadObjectDetector` so the Streamlit app does not crash if the model cannot be loaded. The wrapper filters detections to road-relevant classes and returns object counts for the dashboard and report.

## Canny Edge Detection

Lane detection begins by converting the image to grayscale and smoothing it with a Gaussian blur. Canny edge detection then highlights strong intensity changes, which often correspond to lane markings, road boundaries, or nearby object edges.

## Hough Transform

After edge detection, the app masks the lower road region and applies the probabilistic Hough Line Transform. This transform finds line segments by voting for likely straight-line parameters in edge space.

## Lane Detection Pipeline

The lane pipeline follows these steps:

1. Convert RGB image to grayscale.
2. Apply Gaussian blur to reduce noise.
3. Run Canny edge detection.
4. Keep only a trapezoid-shaped road region.
5. Detect line segments using Hough Lines.
6. Filter unstable slopes.
7. Average left and right lane candidates.
8. Draw lane overlays on the original image.

If no stable lane candidate is found, the app returns a `lane_detected=False` status rather than failing.

## Road Segmentation Idea

The current segmentation module uses a lightweight classical approximation. It searches for low-saturation, road-like pixels in the lower part of the frame and cleans the mask with morphology. This is useful for a portfolio MVP because it shows segmentation thinking without requiring a large semantic segmentation model.

A stronger future version could use DeepLabV3, YOLO segmentation, SegFormer, SAM, or a road-specific segmentation network.

## Pothole Detection Idea

Potholes are not part of the standard COCO object classes, so the project supports an optional custom YOLO pothole model. If `models/pothole_yolo.pt` exists, the app loads it and runs pothole detection. If the file is missing, the app shows a friendly message and continues running.

## Risk Scoring Rules

The risk score is rule-based and interpretable. It increases when:

- Pedestrians and vehicles are detected together.
- Potholes or road-damage regions are detected.
- Lane markings are missing or unclear.
- Many vehicles appear in the scene.
- Buses or trucks appear in the scene.
- Road segmentation is unavailable, adding uncertainty.

Scores are mapped to:

- `0-35`: Low Risk
- `36-70`: Moderate Risk
- `71-100`: High Risk
