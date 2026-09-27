"""Perception: video -> kadrlar bo'yicha trayektoriyalar (detektor + ByteTrack).

Natija — `Perception`: har bir qayta ishlangan kadr uchun vaqt va o'sha kadrdagi
barcha trek qilingan obyektlar. Koordinatalar 0..1 ga normallashtirilgan (sahna
sxemasi ham shunday), shuning uchun kadr o'lchamiga bog'liq emas.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import config

# dets ustunlari
K, TID, CLS, CONF, X1, Y1, X2, Y2 = range(8)


@dataclass
class Perception:
    fps: float
    n_frames: int
    width: int
    height: int
    frame_idx: np.ndarray   # (N,) qayta ishlangan kadrlar raqami
    dets: np.ndarray        # (M, 8): [k, track_id, cls, conf, x1, y1, x2, y2]; k — frame_idx dagi indeks
    bg: np.ndarray | None = None   # videoning fon kadri (kulrang, align.REF_W) — sahnani moslash uchun

    @property
    def duration(self) -> float:
        return self.n_frames / self.fps if self.fps else 0.0

    @property
    def times(self) -> np.ndarray:
        return self.frame_idx / self.fps

    def save(self, path: Path) -> None:
        extra = {"bg": self.bg} if self.bg is not None else {}
        np.savez_compressed(path, fps=self.fps, n_frames=self.n_frames, width=self.width, height=self.height,
                            frame_idx=self.frame_idx, dets=self.dets, **extra)

    @classmethod
    def load(cls, path: Path) -> "Perception":
        z = np.load(path)
        return cls(float(z["fps"]), int(z["n_frames"]), int(z["width"]), int(z["height"]), z["frame_idx"], z["dets"],
                   z["bg"] if "bg" in z.files else None)


_model = None


def load_detector():
    """Model bir marta yuklanadi (bir nechta video uchun qayta ishlatiladi)."""
    global _model
    if _model is None:
        import torch
        from ultralytics import YOLO
        torch.manual_seed(config.SEED)
        _model = YOLO(str(config.WEIGHTS))
    return _model


def _device():
    import torch
    return 0 if torch.cuda.is_available() else "cpu"


def run(video_path: str, stride_sec: float = config.PART_A_STRIDE_SEC, progress=None,
        max_sec: float | None = None, imgsz: int = config.IMGSZ) -> Perception:
    """Videoni boshidan oxirigacha o'qiydi; har `stride_sec` da bir kadrni detektor+tracker'dan o'tkazadi.

    H.264 da kadrga sakrash (seek) qimmat va noaniq — shuning uchun ketma-ket o'qiymiz,
    keraksiz kadrlar faqat grab() qilinadi (rasmga aylantirilmaydi).
    `max_sec` — faqat boshidagi shuncha soniya (sayt demosi uchun); `imgsz` — detektor kirish o'lchami.
    """
    model = load_detector()
    dev = _device()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    limit = int(max_sec * fps) if max_sec is not None else None   # None — video oxirigacha
    if limit is not None and n_frames > 0:
        n_frames = min(n_frames, limit)
    stride = max(1, round(stride_sec * fps))
    half_kw = {"half": True} if dev != "cpu" else {}   # GPU'da fp16 — ~2x tez

    # ByteTrack holati har video uchun yangidan boshlanadi
    if getattr(model, "predictor", None) is not None and hasattr(model.predictor, "trackers"):
        del model.predictor.trackers

    frame_ids, rows = [], []
    bg_frames, bg_every = [], max(1, n_frames // 25)     # ~25 ta kadr fon uchun
    idx = 0
    try:
        while limit is None or idx < limit:
            if idx % stride:
                if not cap.grab():
                    break
                idx += 1
                continue
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            if w > config.FRAME_W:
                frame = cv2.resize(frame, (config.FRAME_W, round(h * config.FRAME_W / w)), interpolation=cv2.INTER_AREA)
            if (idx // stride) % max(1, bg_every // stride) == 0 and len(bg_frames) < 30:
                bg_frames.append(frame.copy())
            res = model.track(frame, persist=True, tracker=str(config.TRACKER_CFG), imgsz=imgsz,
                              conf=config.DET_CONF, device=dev, verbose=False, **half_kw)[0]
            k = len(frame_ids)
            frame_ids.append(idx)
            b = res.boxes
            if b is not None and len(b) and b.id is not None:
                xyxyn = b.xyxyn.cpu().numpy()
                ids = b.id.cpu().numpy()
                cls = b.cls.cpu().numpy()
                conf = b.conf.cpu().numpy()
                for i in range(len(ids)):
                    rows.append((k, ids[i], cls[i], conf[i], *xyxyn[i]))
            if progress and k % 10 == 0:
                progress(idx, n_frames)
            idx += 1
    finally:
        cap.release()

    from .align import background
    dets = np.asarray(rows, dtype=np.float32).reshape(-1, 8)
    bg = background(bg_frames) if bg_frames else None
    return Perception(float(fps), n_frames, width, height, np.asarray(frame_ids, dtype=np.int32), dets, bg)
