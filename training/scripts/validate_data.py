"""Validate YOLO dataset export without modifying anything.

Usage: python validate_data.py --data <path_to_data.yaml_dir>
"""
import argparse
import sys
from pathlib import Path
from collections import defaultdict

import yaml
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Directory containing data.yaml")
    args = ap.parse_args()

    root = Path(args.data).resolve()
    yaml_path = root / "data.yaml"
    if not yaml_path.exists():
        print(f"XATO: {yaml_path} topilmadi")
        sys.exit(1)

    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    names = cfg["names"]
    nc = cfg["nc"]
    print(f"data.yaml: nc={nc}, names={names}")

    errors = []
    warnings = []

    size_bins = {"small(<32px)": 0, "medium(32-96px)": 0, "large(>96px)": 0}

    for split in ["train", "val"]:
        img_dir = root / cfg[split].replace("images/", "images/") if False else root / cfg[split]
        # cfg[split] like "images/train"
        img_dir = root / cfg[split]
        lbl_dir = root / "labels" / split

        images = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
        labels = sorted([p for p in lbl_dir.iterdir() if p.suffix == ".txt"])

        img_stems = {p.stem for p in images}
        lbl_stems = {p.stem for p in labels}

        missing_labels = img_stems - lbl_stems
        missing_images = lbl_stems - img_stems
        if missing_labels:
            errors.append(f"[{split}] {len(missing_labels)} rasmda label yo'q: {sorted(missing_labels)[:5]}...")
        if missing_images:
            errors.append(f"[{split}] {len(missing_images)} labelda rasm yo'q: {sorted(missing_images)[:5]}...")

        class_box_count = defaultdict(int)
        total_boxes = 0

        for lbl_path in labels:
            img_path = img_dir / (lbl_path.stem + ".jpg")
            if not img_path.exists():
                img_path = img_dir / (lbl_path.stem + ".png")
            if not img_path.exists():
                continue
            try:
                with Image.open(img_path) as im:
                    w, h = im.size
            except Exception as e:
                errors.append(f"[{split}] {img_path.name}: rasmni ochib bo'lmadi ({e})")
                continue

            with open(lbl_path, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f if l.strip()]

            for i, line in enumerate(lines):
                parts = line.split()
                if len(parts) != 5:
                    errors.append(f"[{split}] {lbl_path.name}:{i+1}: 5 ustun emas ({len(parts)}) -> '{line}'")
                    continue
                try:
                    cls_id = int(parts[0])
                    cx, cy, bw, bh = map(float, parts[1:])
                except ValueError:
                    errors.append(f"[{split}] {lbl_path.name}:{i+1}: raqamga o'gira olmadi -> '{line}'")
                    continue

                if cls_id < 0 or cls_id >= nc:
                    errors.append(f"[{split}] {lbl_path.name}:{i+1}: noma'lum class_id={cls_id}")
                    continue

                for v, name in [(cx, "cx"), (cy, "cy"), (bw, "bw"), (bh, "bh")]:
                    if not (0.0 <= v <= 1.0):
                        errors.append(f"[{split}] {lbl_path.name}:{i+1}: {name}={v} 0..1 oralig'idan tashqarida")

                if bw <= 0 or bh <= 0:
                    errors.append(f"[{split}] {lbl_path.name}:{i+1}: bw/bh <= 0")

                class_box_count[cls_id] += 1
                total_boxes += 1

                # size bucket in pixels (based on this image's resolution)
                px_w = bw * w
                px_h = bh * h
                min_side = min(px_w, px_h)
                if split == "train" or split == "val":
                    if min_side < 32:
                        size_bins["small(<32px)"] += 1
                    elif min_side < 96:
                        size_bins["medium(32-96px)"] += 1
                    else:
                        size_bins["large(>96px)"] += 1

        print(f"\n=== {split} ===")
        print(f"rasm: {len(images)}, label fayl: {len(labels)}, jami box: {total_boxes}")
        for cid in range(nc):
            cname = names[cid] if isinstance(names, dict) else names[cid]
            print(f"  {cid}: {cname:<16} {class_box_count.get(cid, 0)}")

    print("\n=== Box o'lcham taqsimoti (train+val jami) ===")
    for k, v in size_bins.items():
        print(f"  {k}: {v}")

    print("\n=== Svetofor klasslari alohida ===")
    light_names = {i: n for i, n in (names.items() if isinstance(names, dict) else enumerate(names))
                   if "light" in n}
    print(f"  light class ids: {light_names}")

    if errors:
        print(f"\n!!! {len(errors)} XATO topildi:")
        for e in errors[:50]:
            print("  -", e)
        if len(errors) > 50:
            print(f"  ... yana {len(errors)-50} ta")
    else:
        print("\nXato topilmadi. Data struktura to'g'ri.")

    if warnings:
        print(f"\n{len(warnings)} ogohlantirish:")
        for w in warnings[:20]:
            print("  -", w)


if __name__ == "__main__":
    main()
