"""Project configuration for Smart Road Scene Analyzer."""

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
OUTPUT_DIR = ASSETS_DIR / "outputs"
MODELS_DIR = BASE_DIR / "models"

DEFAULT_YOLO_MODEL = "yolo11n.pt"
DEFAULT_CONFIDENCE = 0.35
POTHOLE_MODEL_PATH = MODELS_DIR / "pothole_yolo.pt"

SUPPORTED_IMAGE_TYPES = ("jpg", "jpeg", "png", "bmp", "webp")
SUPPORTED_VIDEO_TYPES = ("mp4", "avi", "mov", "mkv")

FRAME_SKIP = 5
MAX_VIDEO_FRAMES = 300

ROAD_OBJECT_CLASSES = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "traffic light",
    "stop sign",
)
