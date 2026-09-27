"""Kamera siljishini tuzatish: har bir videoning fonini sahna chizilgan etalon kadr bilan solishtirib,
sahna sxemasini shu videoga moslab suradi.

Kamera "qimirlamaydi", lekin videolar orasida bir necha o'n piksel siljishi kuzatildi
(C3902 da ~40 px). Sahna esa bitta kadrda chizilgan — shuning uchun har video uchun
etalon -> joriy video o'zgartirishi (siljish + ozgina burilish/masshtab) baholanadi.
"""
from __future__ import annotations

import cv2
import numpy as np

from . import config

REF_W = 960                       # solishtirish shu kenglikda (tez va shovqinga chidamli)
REF_PATH = config.ROOT / "configs" / "scene_ref.png"


def background(frames: list[np.ndarray]) -> np.ndarray:
    """Bir necha kadrning medianasi — harakatlanuvchi mashina/odamlar yo'qoladi, yo'l qoladi."""
    small = [cv2.cvtColor(cv2.resize(f, (REF_W, round(f.shape[0] * REF_W / f.shape[1])), interpolation=cv2.INTER_AREA),
                          cv2.COLOR_BGR2GRAY) for f in frames]
    return np.median(np.stack(small), axis=0).astype(np.uint8)


def estimate(ref: np.ndarray, cur: np.ndarray) -> np.ndarray | None:
    """ref -> cur o'zgartirish (2x3, FRAME_W koordinatalarida). Ishonchsiz bo'lsa None."""
    orb = cv2.ORB_create(4000)
    k1, d1 = orb.detectAndCompute(ref, None)
    k2, d2 = orb.detectAndCompute(cur, None)
    if d1 is None or d2 is None or len(k1) < 50 or len(k2) < 50:
        return None
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d1, d2)
    if len(matches) < 30:
        return None
    src = np.float32([k1[m.queryIdx].pt for m in matches])
    dst = np.float32([k2[m.trainIdx].pt for m in matches])
    M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.0, maxIters=5000)
    if M is None or inl is None or inl.sum() < 25:
        return None
    scale = float(np.hypot(M[0, 0], M[1, 0]))
    if not (0.9 < scale < 1.1) or np.abs(M[:, 2]).max() > 0.15 * REF_W:   # aqlga sig'maydigan o'zgarish
        return None
    k = config.FRAME_W / REF_W           # REF_W -> FRAME_W koordinatalari (siljish masshtablanadi)
    M = M.copy()
    M[:, 2] *= k
    return M


def load_ref() -> np.ndarray | None:
    return cv2.imread(str(REF_PATH), cv2.IMREAD_GRAYSCALE) if REF_PATH.exists() else None
