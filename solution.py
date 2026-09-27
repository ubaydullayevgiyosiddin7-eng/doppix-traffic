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


def detect_events(video_path: str) -> list[list]:
    """Part A — traffic event detection."""
    from doppix import perception, rules
    _seed()
    p = perception.run(video_path)
    return rules.detect(p)


class RiskEstimator:
    """Part B — causal accident anticipation (TTC bo'yicha). Videoni o'zi ochmaydi."""

    _model = None   # og'irliklar bir marta yuklanadi, videolar orasida qayta ishlatiladi

    def reset(self, meta: dict) -> None:
        from doppix.risk import RiskModel
        _seed()
        if RiskEstimator._model is None:
            RiskEstimator._model = RiskModel()
        RiskEstimator._model.reset(float(meta.get("fps") or 25.0))

    def step(self, frame: np.ndarray, t_sec: float) -> float:
        return RiskEstimator._model.step(frame, t_sec)
