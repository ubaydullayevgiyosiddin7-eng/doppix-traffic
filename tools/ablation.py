"""Ablation: har bir qism (kamera moslash, kadr chastotasi, qoidalardagi filtrlar) Score A ga qancha qo'shadi.

Detektorni qayta ishga tushirmaydi — keshlangan perception ustida qoidalar varianti bilan hisoblaydi.
Past kadr chastotasi keshdagi kadrlarni siyraklashtirib taqlid qilinadi (tracker ~10 fps da ishlagan).

    python tools/ablation.py --cache ../02_data/cache_v2 --labels ../02_data/labels/events --out ablation.json
"""
import argparse
import json
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
import evaluate  # noqa: E402
from doppix import rules  # noqa: E402
from doppix.perception import K, Perception  # noqa: E402
from doppix.scene import Scene  # noqa: E402


def subsample(p: Perception, n: int) -> Perception:
    """Har n-kadrni qoldirish (kamroq fps ni taqlid qilish)."""
    keep = np.arange(0, len(p.frame_idx), n)
    remap = -np.ones(len(p.frame_idx), dtype=int)
    remap[keep] = np.arange(len(keep))
    d = p.dets[remap[p.dets[:, K].astype(int)] >= 0].copy()
    d[:, K] = remap[d[:, K].astype(int)]
    return Perception(p.fps, p.n_frames, p.width, p.height, p.frame_idx[keep], d, p.bg)


@contextmanager
def params(**kw):
    old = {k: rules.P[k] for k in kw}
    rules.P.update(kw)
    try:
        yield
    finally:
        rules.P.update(old)


@contextmanager
def no_alignment():
    orig = rules.aligned_scene
    rules.aligned_scene = lambda p, scene=None: scene or Scene.load()
    try:
        yield
    finally:
        rules.aligned_scene = orig


VARIANTS = [
    ("Full pipeline", lambda: params(), 1),
    ("Frame rate ~5 fps (every 2nd processed frame)", lambda: params(), 2),
    ("Frame rate ~3.3 fps (every 3rd)", lambda: params(), 3),
    ("No camera-shift alignment", no_alignment, 1),
    ("failure_to_yield: no signal rule", lambda: params(fty_signal_cw=()), 1),
    ("failure_to_yield: no walking / creeping thresholds", lambda: params(fty_ped_min_rel=0.0, fty_veh_min_rel=0.0), 1),
    ("jaywalking: no 'alongside crosswalk' filter", lambda: params(jw_along_max=1.01), 1),
    ("jaywalking: no scooter-speed filter", lambda: params(ped_max_rel_speed=1e9), 1),
    ("stopped_vehicle: no 'yielding to pedestrians' filter", lambda: params(sv_yield_share=2.0), 1),
]

ap = argparse.ArgumentParser()
ap.add_argument("--cache", required=True)
ap.add_argument("--labels", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

gt, perc = {}, {}
for f in sorted(Path(args.labels).glob("*.json")):
    lab = json.loads(f.read_text(encoding="utf-8"))
    name = f.name[:-5]
    p = Perception.load(Path(args.cache) / f"{name.split('.')[0]}.npz")
    gt[name] = {"duration": p.duration, "fps": p.fps, "events": [[e["start"], e["end"], e["label"]] for e in lab["events"]]}
    perc[name] = p

rows = []
for title, ctx, n in VARIANTS:
    with ctx():
        pred = {v: {"events": rules.detect(subsample(p, n) if n > 1 else p)} for v, p in perc.items()}
    rep = evaluate.evaluate_part_a(gt, pred)
    row = {"variant": title, "score_a": round(rep["score_a"], 3),
           "per_class": {c: round(v["f1_mean"], 3) for c, v in rep["per_class"].items()}}
    rows.append(row)
    print(f"{row['score_a']:.3f}  {title}", flush=True)
Path(args.out).write_text(json.dumps(rows, indent=1), encoding="utf-8")
