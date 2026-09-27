"""Part B mantiqini keshlangan treklar ustida sinash: yolg'on signallar qancha?

    python tools/risk_offline.py --cache ../02_data/cache
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from doppix import risk  # noqa: E402
from doppix.perception import Perception  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--cache", required=True)
args = ap.parse_args()

for npz in sorted(Path(args.cache).glob("*.npz")):
    p = Perception.load(npz)
    curve = risk.replay(p)
    s = np.array([c[1] for c in curve])
    alarms = [curve[i][0] for i in range(len(s)) if s[i] >= 0.5 and (i == 0 or s[i - 1] < 0.5)]
    print(f"{npz.stem}: {p.duration:.0f}s, xavf>=0.5 kadrlar {100 * (s >= 0.5).mean():.1f}%, o'rtacha {s.mean():.3f}, "
          f"max {s.max():.3f}, alarmlar {len(alarms)}: {[f'{a // 60:.0f}:{a % 60:04.1f}' for a in alarms[:12]]}")
