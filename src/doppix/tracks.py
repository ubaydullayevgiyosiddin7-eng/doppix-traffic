"""Perception natijasidan treklar: har bir obyektning vaqt bo'yicha holati, tezligi va svetofor fazasi."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config
from .perception import CLS, CONF, K, TID, X1, X2, Y1, Y2, Perception
from .scene import H, W, Scene, inside


@dataclass
class Track:
    tid: int
    cls: int
    t: np.ndarray        # (n,) sekund
    box: np.ndarray      # (n,4) px x1,y1,x2,y2 (FRAME_W o'lchamida)
    conf: np.ndarray

    @property
    def foot(self) -> np.ndarray:
        """Yerga tegib turgan nuqta: box pastki o'rtasi — yo'l tekisligidagi joylashuv."""
        return np.stack([(self.box[:, 0] + self.box[:, 2]) / 2, self.box[:, 3]], 1)

    @property
    def center(self) -> np.ndarray:
        return np.stack([(self.box[:, 0] + self.box[:, 2]) / 2, (self.box[:, 1] + self.box[:, 3]) / 2], 1)

    @property
    def height(self) -> np.ndarray:
        return self.box[:, 3] - self.box[:, 1]

    def speed(self, window: float = 1.0) -> np.ndarray:
        """px/s, `window` sekund oralig'idagi siljish bo'yicha (bitta kadrdagi titrashga chidamli).
        Uzoqdagi obyekt kichik ko'rinadi — shuning uchun box balandligiga nisbatan ham beriladi."""
        f = self.foot
        n = len(self.t)
        out = np.zeros(n)
        j0 = 0
        for i in range(n):
            while self.t[i] - self.t[j0] > window:
                j0 += 1
            j1 = i
            if self.t[j1] - self.t[j0] > 0:
                out[i] = np.linalg.norm(f[j1] - f[j0]) / (self.t[j1] - self.t[j0])
        # boshlang'ich nuqtalar — keyingi oynadan
        if n > 1 and out[0] == 0:
            out[0] = out[1]
        return out


def build_tracks(p: Perception, min_len: int = 3) -> list[Track]:
    times = p.times
    d = p.dets
    tracks = []
    for tid in np.unique(d[:, TID]).astype(int):
        r = d[d[:, TID] == tid]
        r = r[np.argsort(r[:, K])]
        if len(r) < min_len:
            continue
        cls_votes = np.bincount(r[:, CLS].astype(int), weights=r[:, CONF], minlength=len(config.CLASS_NAMES))
        cls = int(np.argmax(cls_votes))
        box = r[:, [X1, Y1, X2, Y2]] * [W, H, W, H]
        tracks.append(Track(tid, cls, times[r[:, K].astype(int)], box, r[:, CONF]))
    return tracks


def light_state(p: Perception, scene: Scene, smooth_sec: float = 1.0) -> np.ndarray:
    """Har bir qayta ishlangan kadr uchun sahnadagi 1-svetofor holati: 1 qizil, 0 qizil emas, -1 noma'lum.

    Svetofor detektorda klass sifatida chiqadi (light_red / light_not_red); sahnadagi svetofor
    hududiga tushgan eng ishonchli detektsiya olinadi, keyin qisqa sakrashlar median bilan tekislanadi."""
    n = len(p.frame_idx)
    state = np.full(n, -1.0)
    if not scene.lights:
        return state
    poly = scene.lights[0]
    d = p.dets[np.isin(p.dets[:, CLS], config.LIGHTS)]
    if len(d):
        cx = (d[:, X1] + d[:, X2]) / 2 * W
        cy = (d[:, Y1] + d[:, Y2]) / 2 * H
        ok = inside(poly, np.stack([cx, cy], 1), margin=-25)
        d = d[ok]
        best = {}
        for row in d:
            k = int(row[K])
            if k not in best or row[CONF] > best[k][CONF]:
                best[k] = row
        for k, row in best.items():
            state[k] = 1.0 if int(row[CLS]) == config.LIGHT_RED else 0.0
    # tekislash: noma'lumlarni oldingi holat bilan to'ldirish + median
    last = -1.0
    for i in range(n):
        if state[i] < 0:
            state[i] = last
        else:
            last = state[i]
    if n:
        w = max(1, int(round(smooth_sec * p.fps / max(1, p.frame_idx[1] - p.frame_idx[0] if n > 1 else 1))))
        if w > 1:
            pad = np.pad(state, (w // 2, w // 2), mode="edge")
            state = np.array([np.median(pad[i:i + w]) for i in range(n)])
    return state
