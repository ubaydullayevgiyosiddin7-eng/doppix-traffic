"""Hodisalarni ko'z bilan tekshirish uchun rasm: har bir jaywalking topilmasi — o'rta kadr kesmasi,
sabab bo'lgan odam (qizil box), uning yo'li (sariq), zebralar (oq), orolchalar (to'q sariq).

    python tools/review_events.py --cache ../02_data/cache_v2 --web ../02_data/samples/_web --video C3902 --out review.jpg
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from doppix import config, rules  # noqa: E402
from doppix.perception import Perception  # noqa: E402
from doppix.rules import P, _is_rider, _rel_speed, _riders, mask_to_intervals, merge  # noqa: E402
from doppix.scene import H, W, Scene, inside_any  # noqa: E402
from doppix.tracks import build_tracks  # noqa: E402


def jaywalk_with_tracks(ctx):
    """rules.jaywalking bilan bir xil mantiq, lekin qaysi trek sabab bo'lganini ham qaytaradi."""
    riders = _riders(ctx)
    out = []
    for tr in ctx.by_cls((config.PERSON,)):
        f = tr.foot
        on_road = ctx.scene.on_road(f, P["jw_road_margin"]) & (tr.height >= P["jw_min_height"]) & (tr.box[:, 3] < H - P["edge_px"])
        off = ~inside_any(ctx.scene.crosswalks, f, -P["jw_cross_margin"])
        moving = tr.speed() > P["move_speed"] * 0.6
        nr = np.array([not _is_rider(riders, t, x) for t, x in zip(tr.t, f)])
        core = on_road & off & moving & nr
        bridge = inside_any(ctx.scene.islands, f) & moving & (tr.height >= P["jw_min_height"]) if ctx.scene.islands \
            else np.zeros(len(f), bool)
        rel = _rel_speed(tr)
        for s, e in merge(mask_to_intervals(tr.t, core | bridge), 1.0):
            m = (tr.t >= s) & (tr.t <= e)
            tc = tr.t[m & core]
            if e - s >= P["jw_min_sec"] and len(tc) and tc[-1] - tc[0] >= P["jw_core_sec"] \
                    and np.median(rel[m]) <= P["ped_max_rel_speed"]:
                out.append((s, e, tr))
    return out


ap = argparse.ArgumentParser()
ap.add_argument("--cache", required=True)
ap.add_argument("--web", required=True)
ap.add_argument("--video", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

p = Perception.load(Path(args.cache) / f"{args.video}.npz")
scene = rules.aligned_scene(p, Scene.load())
ctx = rules.Ctx(p, scene, build_tracks(p), np.zeros(len(p.frame_idx)), p.times)
events = sorted(jaywalk_with_tracks(ctx), key=lambda x: x[0])
cap = cv2.VideoCapture(str(next(Path(args.web).glob(f"{args.video}.*"))))
fw, fh = int(cap.get(3)), int(cap.get(4))
k = fw / W
tiles = []
for s, e, tr in events:
    tm = (s + e) / 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(tm * p.fps))
    ok, img = cap.read()
    if not ok:
        continue
    for cw in scene.crosswalks:
        cv2.polylines(img, [np.int32(cw * k)], True, (255, 255, 255), 1, cv2.LINE_AA)
    for il in scene.islands:
        cv2.polylines(img, [np.int32(il * k)], True, (0, 140, 255), 1, cv2.LINE_AA)
    m = (tr.t >= s - 1) & (tr.t <= e + 1)
    cv2.polylines(img, [np.int32(tr.foot[m] * k).reshape(-1, 1, 2)], False, (0, 255, 255), 2, cv2.LINE_AA)
    i = int(np.argmin(np.abs(tr.t - tm)))
    b = np.int32(tr.box[i] * k)
    cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
    cx, cy = (b[0] + b[2]) // 2, (b[1] + b[3]) // 2
    x0 = int(np.clip(cx - 240, 0, fw - 480)); y0 = int(np.clip(cy - 150, 0, fh - 300))
    tile = img[y0:y0 + 300, x0:x0 + 480].copy()
    cv2.putText(tile, f"{s:.1f}-{e:.1f}s tid{tr.tid} h{int(np.median(tr.height))}", (6, 20), 0, 0.55, (0, 0, 0), 3)
    cv2.putText(tile, f"{s:.1f}-{e:.1f}s tid{tr.tid} h{int(np.median(tr.height))}", (6, 20), 0, 0.55, (255, 255, 255), 1)
    tiles.append(tile)
cols = 3
while len(tiles) % cols:
    tiles.append(np.zeros_like(tiles[0]))
grid = np.vstack([np.hstack(tiles[r:r + cols]) for r in range(0, len(tiles), cols)])
cv2.imwrite(args.out, grid, [cv2.IMWRITE_JPEG_QUALITY, 85])
print(f"{len(events)} ta hodisa -> {args.out}")
