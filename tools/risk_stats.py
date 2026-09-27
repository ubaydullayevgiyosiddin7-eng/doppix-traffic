"""Part B xom xavf taqsimoti (kalibrlash uchun): oddiy harakatda qanday qiymatlar chiqadi."""
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from doppix import config, risk  # noqa: E402
from doppix.perception import CLS, CONF, K, TID, X1, X2, Y1, Y2, Perception  # noqa: E402
from doppix.scene import H, W  # noqa: E402

allraw = []
for npz in sorted(Path(sys.argv[1]).glob("*.npz")):
    p = Perception.load(npz)
    times = p.times
    d = p.dets[np.isin(p.dets[:, CLS], [config.PERSON, *config.VEHICLES]) & (p.dets[:, CONF] >= 0.3)]
    by_k = defaultdict(list)
    for r in d:
        by_k[int(r[K])].append(r)
    hist = defaultdict(lambda: deque(maxlen=16))
    for k in range(0, len(times), 2):
        t = times[k]
        objs = []
        for r in by_k.get(k, []):
            hist[int(r[TID])].append((t, r[[X1, Y1, X2, Y2]] * [W, H, W, H]))
            objs.append((int(r[TID]), int(r[CLS])))
        st = []
        for tid, cls in objs:
            h = [x for x in hist[tid] if t - x[0] <= risk.HIST_SEC + 1e-6]
            if len(h) < 3 or h[-1][0] - h[0][0] < 0.3:
                continue
            (t0, b0), (t1, b1) = h[0], h[-1]
            c0 = np.array([(b0[0] + b0[2]) / 2, (b0[1] + b0[3]) / 2])
            c1 = np.array([(b1[0] + b1[2]) / 2, (b1[1] + b1[3]) / 2])
            st.append((cls, b1, c1, (c1 - c0) / (t1 - t0)))
        allraw.append(risk.pair_risk(st))
a = np.array(allraw)
print(f"qadamlar: {len(a)}, >0: {100 * (a > 0).mean():.1f}%")
for q in [50, 90, 95, 99, 99.5, 99.9, 100]:
    print(f"  {q:5}-foizil: {np.percentile(a, q):.3f}")
