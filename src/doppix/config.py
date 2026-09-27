"""Yechimning barcha sozlamalari bir joyda."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = ROOT / "weights" / "detector.pt"
TRACKER_CFG = ROOT / "configs" / "bytetrack.yaml"
SCENE_CFG = ROOT / "configs" / "scene.json"

SEED = 42

# Detektor klasslari (weights/detector.pt bilan bir xil tartib)
CLASS_NAMES = ["person", "bicycle", "car", "motorcycle", "bus", "truck", "light_red", "light_not_red"]
PERSON, BICYCLE, CAR, MOTORCYCLE, BUS, TRUCK, LIGHT_RED, LIGHT_NOT_RED = range(8)
VEHICLES = (CAR, MOTORCYCLE, BUS, TRUCK)
LIGHTS = (LIGHT_RED, LIGHT_NOT_RED)

# Perception: 4K kadr -> FRAME_W ga kichraytiriladi -> detektor IMGSZ da ishlaydi
FRAME_W = 1920
IMGSZ = 1280
DET_CONF = 0.1           # ByteTrack ikkinchi bosqichi past ishonchli boxlardan ham foydalanadi
PART_A_STRIDE_SEC = 0.1  # Part A: ~10 fps (29.97 fps videoda har 3-kadr)
