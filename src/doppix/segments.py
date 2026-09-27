"""Vaqt bayroqlari / intervallar -> baholanadigan segmentlar."""
from __future__ import annotations

import numpy as np


def mask_to_intervals(times: np.ndarray, mask: np.ndarray) -> list[tuple[float, float]]:
    """Ketma-ket True bo'lgan joylar -> [(boshi, oxiri)]. Oxiri — oxirgi True kadr vaqti."""
    out, start, prev = [], None, None
    for t, m in zip(times, mask):
        if m and start is None:
            start = t
        if not m and start is not None:
            out.append((start, prev))
            start = None
        prev = t
    if start is not None:
        out.append((start, times[-1]))
    return out


def merge(intervals: list[tuple[float, float]], gap: float = 0.0) -> list[tuple[float, float]]:
    """Ustma-ust yoki `gap` dan yaqin intervallarni birlashtiradi (bir xil klass ustma-ust tushmasligi shart)."""
    out = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1] + gap:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def finalize(intervals: list[tuple[float, float]], duration: float, min_len: float = 0.5,
             gap: float = 1.0, pad: float = 0.0) -> list[tuple[float, float]]:
    """Birlashtirish -> qisqa bo'laklarni tashlash -> video chegarasiga qirqish."""
    res = []
    for s, e in merge([(s - pad, e + pad) for s, e in intervals], gap):
        s, e = max(0.0, s), min(duration, e)
        if e - s >= min_len:
            res.append((round(s, 3), round(e, 3)))
    return res
