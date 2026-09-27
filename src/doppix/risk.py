"""Part B: avariya xavfi (causal) — to'qnashuvgacha qolgan vaqt (TTC) bo'yicha.

Faqat hozirgacha kelgan kadrlar ishlatiladi: detektor + o'z ByteTrack'i har `stride` kadrda,
qolgan kadrlarda oxirgi baho qaytariladi. Part A natijasidan foydalanilmaydi (qoida).

G'oya: ikki yo'l ishtirokchisi bir-biriga tez yaqinlashayotgan bo'lsa va hozirgi tezlik bilan
ularning boxlari yaqin soniyalarda ustma-ust tushadigan bo'lsa — xavf baland. Navbatda turgan
(yaqin, lekin harakatsiz) mashinalar xavf bermaydi.
"""
from __future__ import annotations

from collections import defaultdict, deque

import cv2
import numpy as np

from . import config

HORIZON = 5.0         # metrika gorizonti, s
HIST_SEC = 1.0        # tezlikni shuncha soniyalik tarix bo'yicha baholaymiz
TTC_MAX = 3.0         # shundan uzoqroq to'qnashuv — xavf emas (chorrahada hamma bir-biriga "yaqinlashadi")
MIN_CLOSING = 150.0   # px/s — sekin yaqinlashish (navbat, burilish) xavf emas
HOLD = 3              # xavf ketma-ket shuncha qadamda bo'lsagina ko'tariladi (detektor titrashiga qarshi)
EMA = 0.5             # bahoni tekislash
NORMAL_MAX = 0.93     # oddiy harakatdagi xom bahoning ~99.9-foizili (tools/risk_stats.py, 4 ta namuna video)


class RiskState:
    """Detektorga bog'liq bo'lmagan qism: kuzatuvlar tarixi -> xavf bahosi (causal).
    RiskModel (jonli, kadrma-kadr) va replay() (tayyor treklar ustida, sayt/demo uchun) ikkalasi ishlatadi."""

    def __init__(self):
        self.hist: dict[int, deque] = defaultdict(lambda: deque(maxlen=16))
        self.score = 0.0
        self.streak = 0

    def update(self, t: float, objs) -> float:
        """objs: [(track_id, cls, box_xyxy_px)] — shu paytdagi (conf >= 0.3) obyektlar."""
        for tid, _, box in objs:
            self.hist[tid].append((t, np.asarray(box, dtype=np.float32)))
        states = [(cls, *s) for tid, cls, _ in objs if (s := self._state(tid, t)) is not None]
        raw = calibrate(pair_risk(states))
        self.streak = self.streak + 1 if raw > 0 else 0
        raw = raw if self.streak >= HOLD else 0.0
        self.score = float(np.clip(EMA * raw + (1 - EMA) * self.score, 0.0, 1.0))
        return self.score

    def _state(self, tid: int, t: float):
        """Hozirgi box, markaz va tezlik (px/s) — so'nggi HIST_SEC soniya bo'yicha."""
        h = [x for x in self.hist[tid] if t - x[0] <= HIST_SEC + 1e-6]
        if len(h) < 3:
            return None
        (t0, b0), (t1, b1) = h[0], h[-1]
        if t1 - t0 < 0.3:
            return None
        c0 = np.array([(b0[0] + b0[2]) / 2, (b0[1] + b0[3]) / 2])
        c1 = np.array([(b1[0] + b1[2]) / 2, (b1[1] + b1[3]) / 2])
        return b1, c1, (c1 - c0) / (t1 - t0)


class RiskModel:
    def __init__(self, stride_sec: float = 0.2):
        from ultralytics import YOLO
        self.model = YOLO(str(config.WEIGHTS))   # Part A'dagidan alohida — tracker holati aralashmasin
        self.stride_sec = stride_sec
        self.reset(25.0)

    def reset(self, fps: float) -> None:
        self.fps = fps
        self.stride = max(1, round(self.stride_sec * fps))
        self.state = RiskState()
        self.n = 0
        if getattr(self.model, "predictor", None) is not None and hasattr(self.model.predictor, "trackers"):
            del self.model.predictor.trackers

    def step(self, frame: np.ndarray, t: float) -> float:
        n = self.n
        self.n += 1
        if n % self.stride:
            return self.state.score
        h, w = frame.shape[:2]
        if w > config.FRAME_W:
            frame = cv2.resize(frame, (config.FRAME_W, round(h * config.FRAME_W / w)), interpolation=cv2.INTER_AREA)
        import torch
        dev = 0 if torch.cuda.is_available() else "cpu"
        res = self.model.track(frame, persist=True, tracker=str(config.TRACKER_CFG), imgsz=config.IMGSZ,
                               conf=config.DET_CONF, device=dev, verbose=False,
                               classes=[config.PERSON, *config.VEHICLES], **({"quantize": 16} if dev != "cpu" else {}))[0]
        b = res.boxes
        objs = []
        if b is not None and len(b) and b.id is not None:
            for tid, cls, conf, box in zip(b.id.int().tolist(), b.cls.int().tolist(), b.conf.tolist(), b.xyxy.tolist()):
                if conf >= 0.3:
                    objs.append((tid, cls, box))
        return self.state.update(t, objs)


def replay(p, stride_sec: float = 0.2) -> list[tuple[float, float]]:
    """Tayyor Perception (Part A treklari) ustida xuddi shu causal hisob: [(t, score)].
    Sayt va jonli demo uchun — detektorni ikkinchi marta ishlatmaslik uchun."""
    from .perception import CLS, CONF, K, TID, X1, X2, Y1, Y2
    from .scene import H, W
    times = p.times
    keep = np.isin(p.dets[:, CLS], [config.PERSON, *config.VEHICLES]) & (p.dets[:, CONF] >= 0.3)
    d = p.dets[keep]
    by_k: dict[int, list] = defaultdict(list)
    for r in d:
        by_k[int(r[K])].append((int(r[TID]), int(r[CLS]), r[[X1, Y1, X2, Y2]] * [W, H, W, H]))
    dt = float(np.median(np.diff(times))) if len(times) > 1 else stride_sec
    every = max(1, round(stride_sec / dt))
    st = RiskState()
    return [(float(times[k]), st.update(float(times[k]), by_k.get(k, []))) for k in range(0, len(times), every)]


def pair_risk(states) -> float:
    """states: [(cls, box_xyxy, center, velocity_px_s)]. Eng xavfli juftlikning xom bahosi 0..1.

    Xavf: (1) juftlik tez yaqinlashyapti, (2) hozirgi tezlik bilan TTC_MAX ichida markazlari
    "tegish" masofasiga keladi, (3) kamida bittasi mashina."""
    best = 0.0
    for i in range(len(states)):
        ci, bi, pi, vi = states[i]
        for j in range(i + 1, len(states)):
            cj, bj, pj, vj = states[j]
            if ci == config.PERSON and cj == config.PERSON:
                continue
            dp, dv = pj - pi, vj - vi
            dist = float(np.linalg.norm(dp))
            closing = -float(dp @ dv) / (dist + 1e-6)
            if closing < MIN_CLOSING:
                continue
            tca = float(np.clip(-(dp @ dv) / (dv @ dv + 1e-6), 0.0, HORIZON))
            if tca > TTC_MAX:
                continue
            dmin = float(np.linalg.norm(dp + dv * tca))
            reach = 0.5 * (min(bi[2] - bi[0], bi[3] - bi[1]) + min(bj[2] - bj[0], bj[3] - bj[1]))
            if dmin > 0.6 * reach:
                continue
            r = (1.0 - tca / TTC_MAX) * (1.0 - dmin / (0.6 * reach))
            best = max(best, float(np.sqrt(r)))
    return best


def calibrate(raw: float) -> float:
    """Xom baho -> metrika bahosi. 20 daqiqalik oddiy harakatdagi (avariyasiz) xom baholar deyarli
    doim NORMAL_MAX dan past — ular [0, 0.45] ga siqiladi (faqat AP tartibi uchun), alarm (>=0.5)
    faqat undan yuqori, oddiy harakatda ko'rilmagan holatlarda."""
    if raw <= NORMAL_MAX:
        return 0.45 * raw / NORMAL_MAX
    return 0.45 + 0.55 * (raw - NORMAL_MAX) / (1.0 - NORMAL_MAX)
