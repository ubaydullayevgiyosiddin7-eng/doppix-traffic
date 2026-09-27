"""Detektor datasetining rasmlarini namuna videolardan qayta chiqarish.

Labellar (training/dataset/labels/{train,val}/<VIDEO>_<KADR>.txt) repo'da bor; rasmlar esa tashkilotchilar
videolarining kadrlari bo'lgani uchun repo'ga qo'yilmagan. Bu skript aynan o'sha kadrlarni (kadr raqami
bo'yicha, ketma-ket o'qib — seek noaniqligisiz) 1920 px kenglikda saqlaydi.

    python training/extract_frames.py --videos /path/to/samples
    python training/train_detector.py --data training/dataset/data.yaml --epochs 100 --imgsz 1280 --batch 8
"""
import argparse
from collections import defaultdict
from pathlib import Path

import cv2

DS = Path(__file__).resolve().parent / "dataset"

ap = argparse.ArgumentParser()
ap.add_argument("--videos", required=True, help="namuna videolar papkasi (C3896.MP4 ...)")
args = ap.parse_args()

need = defaultdict(dict)                      # video -> {kadr: (split, nom)}
for lbl in DS.glob("labels/*/*.txt"):
    video, frame = lbl.stem.rsplit("_", 1)
    need[video][int(frame)] = (lbl.parent.name, lbl.stem)

for video, frames in sorted(need.items()):
    path = next(Path(args.videos).glob(f"{video}.*"))
    cap = cv2.VideoCapture(str(path))
    last, idx, saved = max(frames), 0, 0
    while idx <= last:
        if idx not in frames:
            if not cap.grab():
                break
            idx += 1
            continue
        ok, img = cap.read()
        if not ok:
            break
        h, w = img.shape[:2]
        if w != 1920:
            img = cv2.resize(img, (1920, round(h * 1920 / w)), interpolation=cv2.INTER_AREA)
        split, name = frames[idx]
        out = DS / "images" / split / f"{name}.jpg"
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved += 1
        idx += 1
    cap.release()
    print(f"{video}: {saved}/{len(frames)} kadr")
