"""Qoidalar ishlatadigan hududlarni rasm qilib chizish (labeler'dagi "Dastur zonalari" qatlami).

Rasm qoidalardagi o'sha chegaralar (`rules.P`) va o'sha geometriya bilan chiziladi — shuning uchun
qatlamda ko'ringan narsa dastur aynan nimani tekshirayotganini ko'rsatadi.
"""
from __future__ import annotations

import cv2
import numpy as np

from .rules import P
from .scene import H, W, Scene

LEGEND = [
    {"id": "jaywalk", "color": "#ef4444", "name": "Piyoda uchun taqiqlangan joy",
     "desc": "Piyoda shu yerda yursa — jaywalking (qatnov qismi, zebradan uzoqda)"},
    {"id": "grace", "color": "#facc15", "name": "Zebra cheti (kechirim zonasi)",
     "desc": f"Zebradan {P['jw_cross_margin']:.0f} px gacha — bu yerda yurgan piyoda hisoblanmaydi"},
    {"id": "intersection", "color": "#3b82f6", "name": "Chorraha ichi (ko'k chegara ichida)",
     "desc": "Mashina shu yerda ≥10 s tursa — stopped_vehicle; bir nechta tursa — congestion"},
    {"id": "stopzone", "color": "#a855f7", "name": "Stop-chiziq va zebra orasi",
     "desc": "Qizilda shu yerda to'xtab turgan mashina — stop_line"},
    {"id": "stopline", "color": "#f97316", "name": "Stop-chiziq tekshiruv oralig'i",
     "desc": f"Qizilda shu chiziqni kesib o'tsa — red_light (chiziq ±{P['sl_x_ext']:.0f} px kengaytirilgan)"},
    {"id": "light", "color": "#22c55e", "name": "Svetofor qidiruv hududi",
     "desc": "Svetofor rangi shu hududdagi detektsiyadan o'qiladi"},
]
_COLORS = {z["id"]: z["color"] for z in LEGEND}


def _sdf(poly: np.ndarray) -> np.ndarray:
    """Imzoli masofa (px): ichkarida musbat, tashqarida manfiy — scene.inside() bilan bir xil ma'no."""
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [np.round(poly).astype(np.int32)], 1)
    din = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    dout = cv2.distanceTransform(1 - m, cv2.DIST_L2, 5)
    return din - dout


def _rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)   # BGR


def render(scene: Scene) -> np.ndarray:
    """(H, W, 4) BGRA — shaffof fonda zonalar."""
    img = np.zeros((H, W, 4), np.uint8)

    def fill(mask: np.ndarray, zone: str, alpha: int):
        b, g, r = _rgb(_COLORS[zone])
        img[mask] = (b, g, r, alpha)

    road = np.zeros((H, W), bool)
    for p in scene.carriageway:
        road |= _sdf(p) >= P["jw_road_margin"]
    for p in scene.islands:
        road &= _sdf(p) < -P["jw_road_margin"]
    cw_near = np.zeros((H, W), bool)
    cw_in = np.zeros((H, W), bool)
    for p in scene.crosswalks:
        d = _sdf(p)
        cw_near |= d >= -P["jw_cross_margin"]
        cw_in |= d >= 0

    ys = np.arange(H)[:, None].repeat(W, 1)
    xs = np.arange(W)[None, :].repeat(H, 0)
    if scene.crosswalks:
        cw = scene.crosswalks[0]
        o = np.argsort(cw[:, 0])
        edge = np.interp(np.arange(W), cw[o, 0], cw[o, 1])
        below = ys > edge[None, :] + 20
        inter = (road & below).astype(np.uint8)

    fill(road & ~cw_near, "jaywalk", 45)
    fill(cw_near & ~cw_in & road, "grace", 110)

    if scene.stop_lines:
        line = scene.stop_lines[0]
        a, b = line[0], line[-1]
        x0, x1 = line[:, 0].min() - P["sl_x_ext"], line[:, 0].max() + P["sl_x_ext"]
        k = (b[1] - a[1]) / (b[0] - a[0] + 1e-6)
        yl = a[1] + k * (xs - a[0])
        in_x = (xs >= x0) & (xs <= x1)
        if scene.crosswalks:
            zone = in_x & (ys > yl) & (ys < edge[None, :] + 60) & road
            fill(zone, "stopzone", 70)
        p0 = (int(x0), int(a[1] + k * (x0 - a[0])))
        p1 = (int(x1), int(a[1] + k * (x1 - a[0])))
        cv2.line(img, p0, p1, (*_rgb(_COLORS["stopline"]), 255), 5, cv2.LINE_AA)

    if scene.crosswalks:
        # chorraha zonasi qizil zona bilan ustma-ust — ko'rinishi uchun chegara chizig'i bilan
        cnts, _ = cv2.findContours(inter, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnts, -1, (*_rgb(_COLORS["intersection"]), 255), 4, cv2.LINE_AA)

    for p in scene.lights:
        m = (_sdf(p) >= -25).astype(np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnts, -1, (*_rgb(_COLORS["light"]), 255), 3, cv2.LINE_AA)
    return img
