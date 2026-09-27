"""Namuna videolar uchun perception natijasini keshlaydi — qoidalarni tez sinash uchun.

    python tools/cache_perception.py --videos ../02_data/samples --out ../02_data/cache
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from doppix import perception  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--videos", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--force", action="store_true")
args = ap.parse_args()

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
src = Path(args.videos)
videos = [src] if src.is_file() else sorted(p for p in src.iterdir() if p.suffix.lower() == ".mp4")
for v in videos:
    dst = out / f"{v.stem}.npz"
    if dst.exists() and not args.force:
        print(f"skip {v.name}")
        continue
    t0 = time.perf_counter()
    p = perception.run(str(v), progress=lambda i, n: print(f"  {v.name}: {i}/{n} ({time.perf_counter() - t0:.0f}s)", flush=True))
    p.save(dst)
    dt = time.perf_counter() - t0
    print(f"ok {v.name}: {len(p.frame_idx)} kadr, {len(p.dets)} det, {len(set(p.dets[:, 1].astype(int)))} trek, "
          f"{dt:.0f}s ({dt / p.duration:.2f}x video)", flush=True)
