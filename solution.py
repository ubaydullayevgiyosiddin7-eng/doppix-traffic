"""
solution.py — Do'ppiX (WIUT Hackathon 2026, CV track).

    detect_events(video_path)  -> [[start_sec, end_sec, label], ...]    # Part A
    RiskEstimator().reset(meta); .step(frame, t_sec) -> float           # Part B

Pipeline: fine-tune qilingan YOLO26m (yo'l ishtirokchilari + svetofor rangi) -> ByteTrack ->
sahna sxemasi (configs/scene.json) bo'yicha qoidalar -> segmentlar. Batafsil: README.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

CLASSES: list[str] = [
    "accident",
    "near_miss",
    "red_light",
    "wrong_way",
    "illegal_u_turn",
    "stopped_vehicle",
    "jaywalking",
    "failure_to_yield",
    "illegal_turn",
    "solid_line_crossing",
    "stop_line",
    "congestion",
    "road_obstacle",
    "fire_smoke",
]

RISK_HORIZON_SEC = 5.0


def _seed() -> None:
    import random

    import torch

    from doppix import config
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)


# Vaqt limiti: harness bitta video uchun Part A + Part B ga 3 x davomiylik beradi. Part A boshlangan
# paytni eslab qolamiz — Part B qolgan vaqtga moslashadi (doppix.risk.RiskModel._budget).
_part_a_start: dict[str, float] = {}


def _duration(video_path: str) -> float:
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    fps, n = cap.get(cv2.CAP_PROP_FPS) or 25.0, cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return n / fps if n > 0 else 0.0


def detect_events(video_path: str) -> list[list]:
    """Part A — traffic event detection."""
    import time

    from doppix import config, perception, rules
    _part_a_start[Path(video_path).name] = time.perf_counter()
    _seed()
    dur = _duration(video_path)
    # qoidalar uchun ~0.1 x davomiylik zaxira qoldiramiz (5 daqiqalik videoda ~15 s)
    p = perception.run(video_path, budget_sec=(config.PART_A_BUDGET - 0.1) * dur if dur else None)
    return rules.detect(p)


class RiskEstimator:
    """Part B — causal accident anticipation (TTC bo'yicha). Videoni o'zi ochmaydi."""

    _model = None   # og'irliklar bir marta yuklanadi, videolar orasida qayta ishlatiladi

    def reset(self, meta: dict) -> None:
        import time

        from doppix import config
        from doppix.risk import RiskModel
        _seed()
        if RiskEstimator._model is None:
            RiskEstimator._model = RiskModel()
        fps = float(meta.get("fps") or 25.0)
        n = int(meta.get("n_frames") or 0)
        t0 = _part_a_start.get(str(meta.get("video_id")), time.perf_counter())
        deadline = t0 + config.TIME_FACTOR * config.TIME_SAFETY * n / fps if n else None
        RiskEstimator._model.reset(fps, n, deadline)

    def step(self, frame: np.ndarray, t_sec: float) -> float:
        return RiskEstimator._model.step(frame, t_sec)
