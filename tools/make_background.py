"""Video fonini hisoblash: (1) etalon — configs/scene_ref.png (sahna chizilgan video),
(2) mavjud perception keshlariga `bg` qo'shish (detektorni qayta ishlatmasdan).

    python tools/make_background.py --ref ../02_data/samples/C3896.MP4
    python tools/make_background.py --cache ../02_data/cache --videos ../02_data/samples
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from doppix import align  # noqa: E402
from doppix.perception import Perception  # noqa: E402


def video_bg(path: Path, n: int = 25) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in range(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int((i + 0.5) * total / n))
        ok, f = cap.read()
        if ok:
            frames.append(f)
    cap.release()
    return align.background(frames)


ap = argparse.ArgumentParser()
ap.add_argument("--ref")
ap.add_argument("--cache")
ap.add_argument("--videos")
args = ap.parse_args()

if args.ref:
    bg = video_bg(Path(args.ref))
    cv2.imwrite(str(align.REF_PATH), bg)
    print("etalon:", align.REF_PATH, bg.shape)

if args.cache:
    ref = align.load_ref()
    for npz in sorted(Path(args.cache).glob("*.npz")):
        p = Perception.load(npz)
        vid = next(Path(args.videos).glob(f"{npz.stem}.*"))
        p.bg = video_bg(vid)
        p.save(npz)
        M = align.estimate(ref, p.bg) if ref is not None else None
        shift = "aniqlanmadi" if M is None else f"siljish ({M[0, 2]:+.0f}, {M[1, 2]:+.0f}) px, burilish {np.degrees(np.arctan2(M[1, 0], M[0, 0])):+.2f}°"
        print(f"{npz.stem}: fon qo'shildi — {shift}")
