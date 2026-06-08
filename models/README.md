# Models

This project uses the lightweight Ultralytics YOLO model configured in `config.py` for general road-object detection.

## Default Object Model

The default value is:

```python
DEFAULT_YOLO_MODEL = "yolo11n.pt"
```

On first use, Ultralytics may download the model weights if they are not already available locally.

## Custom Pothole Model

Pothole detection is optional because potholes require a custom trained model. To enable it:

1. Train or download a YOLO pothole detector.
2. Save the weights as:

```text
models/pothole_yolo.pt
```

3. Restart Streamlit or press **Clear model cache** in the sidebar.
4. Enable **Pothole Detection** in the dashboard.

You can also change `POTHOLE_MODEL_PATH` in `config.py` if your model uses another file name.
