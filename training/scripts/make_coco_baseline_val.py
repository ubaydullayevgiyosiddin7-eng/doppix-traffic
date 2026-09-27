"""Create a temporary copy of the val split with labels remapped to COCO class ids,
so a COCO-pretrained model can be evaluated on our val set without touching the
original dataset.

Mapping (ours -> COCO): person 0->0, bicycle 1->1, car 2->2, motorcycle 3->3,
bus 4->5, truck 5->7. light_red (6) and light_not_red (7) rows are dropped
(COCO has no traffic-light-color classes).

Usage: python make_coco_baseline_val.py --data <dir with data.yaml> --out <output dir>
"""
import argparse
import shutil
from pathlib import Path

COCO_80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

OURS_TO_COCO = {0: 0, 1: 1, 2: 2, 3: 3, 4: 5, 5: 7}  # 6,7 (lights) dropped


def make_coco_val(src_dir, out_dir):
    """Build the temp COCO-remapped val copy. Returns (n_imgs, n_kept, n_dropped)."""
    src = Path(src_dir).resolve()
    out = Path(out_dir).resolve()
    if out.exists():
        shutil.rmtree(out)

    (out / "images" / "val").mkdir(parents=True, exist_ok=True)
    (out / "labels" / "val").mkdir(parents=True, exist_ok=True)

    src_img_dir = src / "images" / "val"
    src_lbl_dir = src / "labels" / "val"

    n_imgs = 0
    n_boxes_kept = 0
    n_boxes_dropped = 0

    for img_path in sorted(src_img_dir.iterdir()):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        shutil.copy2(img_path, out / "images" / "val" / img_path.name)
        n_imgs += 1

        lbl_path = src_lbl_dir / (img_path.stem + ".txt")
        out_lines = []
        if lbl_path.exists():
            with open(lbl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    cls_id = int(parts[0])
                    if cls_id in OURS_TO_COCO:
                        new_cls = OURS_TO_COCO[cls_id]
                        out_lines.append(" ".join([str(new_cls)] + parts[1:]))
                        n_boxes_kept += 1
                    else:
                        n_boxes_dropped += 1
        with open(out / "labels" / "val" / (img_path.stem + ".txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(out_lines) + ("\n" if out_lines else ""))

    yaml_content = f"path: {out.as_posix()}\n" + "train: images/val\n" + "val: images/val\n" + f"nc: {len(COCO_80)}\nnames:\n"
    for i, name in enumerate(COCO_80):
        yaml_content += f"  {i}: {name}\n"
    with open(out / "data.yaml", "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"Vaqtinchalik COCO-remapped val: {out}")
    print(f"  rasm: {n_imgs}")
    print(f"  box saqlandi: {n_boxes_kept} (person/bicycle/car/motorcycle/bus/truck)")
    print(f"  box tashlandi: {n_boxes_dropped} (light_red/light_not_red - COCO'da yo'q)")
    return n_imgs, n_boxes_kept, n_boxes_dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Directory containing data.yaml (source dataset)")
    ap.add_argument("--out", required=True, help="Output directory for the temp COCO-remapped copy")
    args = ap.parse_args()
    make_coco_val(args.data, args.out)


if __name__ == "__main__":
    main()
