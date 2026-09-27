"""Sahna sxemasi (configs/scene.json): yo'l, zebralar, stop-chiziq, svetofor hududlari.

Kamera qimirlamaydi — sxema bir marta chiziladi (labeler'ning "Sahna" bo'limida) va hamma
videolar uchun amal qiladi. Koordinatalar 0..1; bu yerda FRAME_W x H pikselga o'giriladi.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import config

W = config.FRAME_W
H = round(W * 9 / 16)


@dataclass
class Scene:
    carriageway: list[np.ndarray] = field(default_factory=list)
    crosswalks: list[np.ndarray] = field(default_factory=list)
    stop_lines: list[np.ndarray] = field(default_factory=list)
    lights: list[np.ndarray] = field(default_factory=list)
    islands: list[np.ndarray] = field(default_factory=list)   # yo'l ichidagi orolcha/trotuar — yo'l emas

    @classmethod
    def load(cls, path: Path = config.SCENE_CFG) -> "Scene":
        s = cls()
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for sh in data["shapes"]:
            pts = (np.asarray(sh["points"], dtype=np.float32) * [W, H]).astype(np.float32)
            {"carriageway": s.carriageway, "crosswalk": s.crosswalks,
             "stop_line": s.stop_lines, "traffic_light": s.lights, "island": s.islands}.get(sh["type"], []).append(pts)
        return s

    def transformed(self, M: np.ndarray | None) -> "Scene":
        """Barcha shakllarni 2x3 affine M bilan o'zgartirilgan nusxa (kamera siljishi uchun)."""
        if M is None:
            return self
        f = lambda polys: [cv2.transform(p.reshape(-1, 1, 2), M).reshape(-1, 2) for p in polys]
        return Scene(f(self.carriageway), f(self.crosswalks), f(self.stop_lines), f(self.lights), f(self.islands))

    def on_road(self, pts: np.ndarray, margin: float = 0.0) -> np.ndarray:
        """Qatnov qismida (chegaradan `margin` px ichkarida) va orolcha/trotuarda emas."""
        road = inside_any(self.carriageway, pts, margin)
        if self.islands:
            road &= ~inside_any(self.islands, pts, -margin)
        return road


def inside(poly: np.ndarray, pts: np.ndarray, margin: float = 0.0) -> np.ndarray:
    """(n,2) nuqtalar poligon ichidami. margin > 0 — chegaradan shuncha px ichkarida bo'lishi shart,
    margin < 0 — chegaradan shuncha px tashqarida bo'lsa ham ichida hisoblanadi."""
    c = poly.reshape(-1, 1, 2)
    d = np.array([cv2.pointPolygonTest(c, (float(x), float(y)), True) for x, y in pts])
    return d >= margin


def inside_any(polys: list[np.ndarray], pts: np.ndarray, margin: float = 0.0) -> np.ndarray:
    out = np.zeros(len(pts), dtype=bool)
    for p in polys:
        out |= inside(p, pts, margin)
    return out


def side_of_line(line: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Chiziqning qaysi tomonida: ishorasi bilan (musbat — kamera tomoni / pastda)."""
    a, b = line[0], line[-1]
    d = b - a
    s = (pts[:, 0] - a[0]) * d[1] - (pts[:, 1] - a[1]) * d[0]
    return -s if d[0] > 0 else s
