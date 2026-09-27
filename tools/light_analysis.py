"""Qaysi svetofor qaysi oqimni boshqaradi? Stop-chiziqni kesib o'tgan mashinalar va o'sha paytdagi
svetofor rangi. Agar kesib o'tishlar deyarli hammasi "qizil emas" paytiga to'g'ri kelsa — shu
svetofor bu oqimni boshqaradi va red_light qoidasi uchun ishonchli.

    python tools/light_analysis.py --cache ../02_data/cache
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from doppix import config  # noqa: E402
from doppix.perception import Perception  # noqa: E402
from doppix.scene import Scene, side_of_line  # noqa: E402
from doppix.rules import aligned_scene  # noqa: E402
from doppix.tracks import build_tracks, light_state  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--cache", required=True)
args = ap.parse_args()
base = Scene.load()

for npz in sorted(Path(args.cache).glob("*.npz")):
    p = Perception.load(npz)
    scene = aligned_scene(p, base)
    line = scene.stop_lines[0]
    x0, x1 = line[:, 0].min() - 150, line[:, 0].max() + 150
    light = light_state(p, scene)
    times = p.times
    known = light >= 0
    red_share = (light[known] == 1).mean() if known.any() else float("nan")
    # fazalar
    phases, prev = [], None
    for t, s in zip(times, light):
        if s != prev:
            phases.append((t, int(s)))
            prev = s
    crossings = []
    for tr in build_tracks(p):
        if tr.cls not in config.VEHICLES:
            continue
        f = tr.foot
        side = side_of_line(line, f)
        for i in range(1, len(f)):
            if side[i - 1] <= 0 < side[i] and x0 <= f[i, 0] <= x1:
                k = int(np.clip(np.searchsorted(times, tr.t[i]), 0, len(times) - 1))
                crossings.append((tr.t[i], int(light[k]), tr.tid))
    st = [c[1] for c in crossings]
    print(f"\n{npz.stem}: svetofor ma'lum {100 * known.mean():.0f}% kadrda, qizil {100 * red_share:.0f}%")
    print("  fazalar:", " ".join(f"{t // 60:.0f}:{t % 60:02.0f}{'R' if s == 1 else 'G' if s == 0 else '?'}" for t, s in phases[:40]))
    print(f"  stop-chiziqni kesib o'tish: {len(st)} ta — qizilda {st.count(1)}, qizil emas {st.count(0)}, noma'lum {st.count(-1)}")
    print("  qizildagi kesib o'tishlar:", [f"{t // 60:.0f}:{t % 60:04.1f}(#{tid})" for t, s, tid in crossings if s == 1][:15])
