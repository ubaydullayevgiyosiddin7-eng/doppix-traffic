"""Part B kalibrlash: oddiy (avariyasiz) harakatda xom xavf bahosi qanday taqsimlangan.

risk.NORMAL_MAX shu taqsimotning ~99.9-foizilidan olingan: undan past xom baholar [0, 0.45] ga siqiladi.

    python tools/risk_stats.py --cache ../02_data/cache_v2
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from doppix import config, risk  # noqa: E402
from doppix.perception import CLS, CONF, K, TID, X1, X2, Y1, Y2, Perception  # noqa: E402
from doppix.scene import H, W  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--cache", required=True)
args = ap.parse_args()

raw_all = []
for npz in sorted(Path(args.cache).glob("*.npz")):
    p = Perception.load(npz)
    d = p.dets[np.isin(p.dets[:, CLS], [config.PERSON, *config.VEHICLES]) & (p.dets[:, CONF] >= 0.3)]
    by_k = defaultdict(list)
    for r in d:
        by_k[int(r[K])].append((int(r[TID]), int(r[CLS]), r[[X1, Y1, X2, Y2]] * [W, H, W, H]))
    st = risk.RiskState()
    for k in range(0, len(p.times), 2):                  # ~5 fps, RiskModel bilan bir xil
        t, objs = float(p.times[k]), by_k.get(k, [])
        for tid, _, box in objs:
            st.hist[tid].append((t, np.asarray(box, dtype=np.float32)))
        states = [(cls, *s) for tid, cls, _ in objs if (s := st._state(tid, t)) is not None]
        raw_all.append(risk.pair_risk(states))
raw = np.array(raw_all)
nz = raw[raw > 0]
print(f"{len(raw)} qadam, nolga teng bo'lmagani {100 * len(nz) / max(1, len(raw)):.1f}%")
for q in (50, 90, 99, 99.9, 100):
    print(f"  {q:5}-foizil (nol bo'lmaganlar): {np.percentile(nz, q) if len(nz) else 0:.3f}")
print(f"risk.NORMAL_MAX = {risk.NORMAL_MAX}")
