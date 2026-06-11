# Smart Road Scene Analyzer using Deep Learning and Computer Vision

A professional Computer Vision web application for analyzing road images and videos. It combines Ultralytics YOLO, OpenCV lane detection, road-region approximation, optional pothole detection, and rule-based risk scoring in a Streamlit dashboard.

## Features

- Image upload and processing
- Video upload with frame-by-frame processing
- YOLO object detection for pedestrians, vehicles, bicycles, buses, trucks, traffic lights, and stop signs
- Classical lane detection with Canny edges and Hough Transform
- Lightweight road-region segmentation approximation
- Optional custom YOLO pothole detection
- Risk score from 0 to 100 with Low Risk, Moderate Risk, or High Risk labels
- Downloadable processed output and markdown report
- Modular Python code designed for GitHub and portfolio use

## Tech Stack

- Python
- Streamlit
- OpenCV
- Ultralytics YOLO
- NumPy
- Pillow
- pathlib
- tempfile
- typing

## Folder Structure

```text
smart-road-scene-analyzer/
|
|-- app.py
|-- config.py
|-- requirements.txt
|-- README.md
|-- .gitignore
|
|-- assets/
|   |-- sample_images/
|   |-- sample_videos/
|   `-- outputs/
|
|-- models/
|   `-- README.md
|
|-- src/
|   |-- __init__.py
|   |-- detector.py
|   |-- lane_detection.py
|   |-- segmentation.py
|   |-- pothole_detection.py
|   |-- risk_assessment.py
|   |-- video_processor.py
|   |-- report_generator.py
|   `-- utils.py
|
`-- docs/
    |-- project_overview.md
    `-- methodology.md
```

## Installation

```bash
cd smart-road-scene-analyzer
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS/Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## How to Run

```bash
streamlit run app.py
```

The app opens in your browser. If the default YOLO model is not already cached, Ultralytics may download it on first use.

## How to Use

1. Select modules in the sidebar.
2. Adjust the confidence threshold.
3. Upload a road image or video
4. Click **Process Image** or **Process Video**.
5. Review detections, lane status, segmentation status, pothole count, and risk score.
6. Download the processed output or markdown report.

## Custom Pothole Model

Pothole detection needs a custom model because potholes are not included in the default COCO classes.

To enable it:

1. Train a YOLO pothole model on pothole or road-damage images.
2. Export the best weights file.
3. Place it here:

```text
models/pothole_yolo.pt
```

4. Restart Streamlit or press **Clear model cache**.
5. Enable **Pothole Detection** in the sidebar.

To use another file name, edit `POTHOLE_MODEL_PATH` in `config.py`.

## Future Improvements

- Add a trained semantic road segmentation model.
- Add object tracking across video frames.
- Estimate vehicle density per lane.
- Add speed estimation with camera calibration.
- Train a dedicated road-damage model.
- Add transformer-based segmentation or attention-based scene understanding.
- Export PDF reports.
- Add automated tests and CI.

## Resume Bullet

Built a Smart Road Scene Analyzer using Python, OpenCV, Streamlit, and YOLO to detect vehicles, pedestrians, lanes, potholes, and drivable regions from road images/videos, with automated risk scoring and visual reports.

## References

- Ultralytics YOLO11 documentation: https://docs.ultralytics.com/models/yolo11/
- Ultralytics Python usage documentation: https://docs.ultralytics.com/usage/python/
