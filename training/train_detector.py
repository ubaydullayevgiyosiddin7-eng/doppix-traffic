"""Fine-tune YOLO26m on the Do'ppiX road-incident detector dataset.

Re-runnable: with no --data, auto-uses the newest doppix_det_* export under the
project root, so a new labeler export needs no path edits, just re-run:
    python train_detector.py
Does NOT modify the dataset. Trains from the COCO-pretrained yolo26m.pt checkpoint
and, at the end, freshly re-evaluates stock COCO yolo26m.pt on the current val
split (remapped to COCO classes) to print an always-up-to-date baseline vs
fine-tune comparison table.

Usage:
    python train_detector.py [--data path/to/data.yaml] --epochs 100 --imgsz 1280 \
        --batch 8 --freeze 0 [--name doppix_yolo26m_full]
"""
import argparse
import json
import sys
from pathlib import Path

import yaml
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from make_coco_baseline_val import make_coco_val  # noqa: E402


def find_latest_data_yaml() -> str:
    """Auto-detect the most recently modified doppix_det_* export under the
    project root (handles the export's self-nested folder layout) and return
    the path to its data.yaml. Used when --data is not given, so re-running
    after a new data export needs no path edits.
    """
    candidates = []
    for d in PROJECT_ROOT.glob("doppix_det_*"):
        if not d.is_dir():
            continue
        for yaml_path in d.rglob("data.yaml"):
            candidates.append(yaml_path)
    if not candidates:
        raise FileNotFoundError(
            f"Hech qanday 'doppix_det_*/**/data.yaml' topilmadi {PROJECT_ROOT} ostida. "
            "--data bilan aniq yo'l bering."
        )
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(candidates[0])


def compute_coco_baseline(data_arg: str, imgsz: int) -> dict:
    """Freshly evaluate stock COCO-pretrained yolo26m.pt on the current val split
    (remapped to COCO class ids) so the baseline always matches whatever dataset
    export is currently being trained on, instead of relying on a stale cached
    result from a previous, differently-sized val set.
    """
    dataset_dir = Path(data_arg).resolve().parent
    tmp_out = PROJECT_ROOT / "runs" / "_tmp_baseline_val"
    make_coco_val(dataset_dir, tmp_out)

    baseline_model = YOLO("yolo26m.pt")
    r = baseline_model.val(
        data=str(tmp_out / "data.yaml"),
        imgsz=imgsz,
        classes=[0, 1, 2, 3, 5, 7],
        seed=42,
        name="_tmp_baseline_eval",
    )
    names = r.names
    per_class = {}
    for i, c in enumerate(r.box.ap_class_index):
        cname = names[int(c)]
        per_class[cname] = {
            "AP50": float(r.box.ap50[i]),
            "AP50_95": float(r.box.ap[i]),
            "P": float(r.box.p[i]),
            "R": float(r.box.r[i]),
        }
    return per_class


def resolve_data_yaml(data_yaml_path: str) -> str:
    """Read the original data.yaml (never modified) and write a resolved copy
    with an absolute 'path' next to it, so ultralytics doesn't fall back to an
    unrelated global datasets_dir when the yaml uses a relative 'path: .'.
    """
    src = Path(data_yaml_path).resolve()
    with open(src, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["path"] = str(src.parent)

    cache_dir = Path("runs/_resolved_data")
    cache_dir.mkdir(parents=True, exist_ok=True)
    resolved_path = cache_dir / f"{src.parent.name}.yaml"
    with open(resolved_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    return str(resolved_path.resolve())


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data",
        default=None,
        help="Path to data.yaml. Omit to auto-use the newest doppix_det_* export under the project root.",
    )
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--freeze", type=int, default=0, help="0 = no freeze, N = freeze first N layers (backbone)")
    ap.add_argument("--name", default=None, help="Omit to auto-name from the dataset export folder")
    ap.add_argument("--model", default="yolo26m.pt")
    ap.add_argument("--patience", type=int, default=30)
    return ap.parse_args()


def main():
    args = parse_args()

    data_arg = args.data or find_latest_data_yaml()
    print(f"Ishlatilayotgan dataset: {data_arg}")

    dataset_dir_name = Path(data_arg).resolve().parent.name
    if args.name:
        run_name = args.name
    else:
        run_name = f"doppix_yolo26m_{dataset_dir_name}"
    args.name = run_name

    model = YOLO(args.model)
    resolved_data = resolve_data_yaml(data_arg)

    train_kwargs = dict(
        data=resolved_data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        seed=42,
        deterministic=True,
        flipud=0,
        degrees=0,
        perspective=0,
        fliplr=0.5,
        name=args.name,
    )
    if args.freeze and args.freeze > 0:
        train_kwargs["freeze"] = args.freeze

    print(f"=== Training: {train_kwargs} ===")
    model.train(**train_kwargs)

    save_dir = Path(model.trainer.save_dir)
    best_pt = save_dir / "weights" / "best.pt"
    print(f"\n=== Validating {best_pt} on native 8-class val ===")
    trained = YOLO(str(best_pt))
    result = trained.val(data=resolved_data, imgsz=args.imgsz, seed=42, name=f"{save_dir.name}_val")

    names = result.names
    finetune_per_class = {}
    for i, c in enumerate(result.box.ap_class_index):
        cname = names[int(c)]
        finetune_per_class[cname] = {
            "AP50": float(result.box.ap50[i]),
            "AP50_95": float(result.box.ap[i]),
            "P": float(result.box.p[i]),
            "R": float(result.box.r[i]),
        }

    print("\n=== Fine-tune natijalari (native 8 klass) ===")
    print(f"{'class':<16}{'AP50':>8}{'AP50-95':>10}{'P':>8}{'R':>8}")
    for cname, m in finetune_per_class.items():
        print(f"{cname:<16}{m['AP50']:>8.3f}{m['AP50_95']:>10.3f}{m['P']:>8.3f}{m['R']:>8.3f}")

    print(f"\n=== Baseline (COCO yolo26m) hisoblanmoqda, joriy val to'plamida ({dataset_dir_name}) ===")
    try:
        bpc = compute_coco_baseline(data_arg, args.imgsz)
    except Exception as e:
        print(f"Baseline hisoblanmadi: {e}")
        bpc = None

    if bpc:
        print("\n=== Baseline (COCO yolo26m) vs Fine-tune (bizning val) ===")
        print(f"{'class':<16}{'base AP50':>10}{'ft AP50':>10}{'base mAP5095':>14}{'ft mAP5095':>12}")
        common = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]
        for cname in common:
            b = bpc.get(cname, {})
            ft = finetune_per_class.get(cname, {})
            b_ap50 = b.get("AP50")
            b_ap = b.get("AP50_95")
            ft_ap50 = ft.get("AP50")
            ft_ap = ft.get("AP50_95")
            b_ap50_s = f"{b_ap50:.3f}" if b_ap50 is not None else "n/a"
            b_ap_s = f"{b_ap:.3f}" if b_ap is not None else "n/a"
            ft_ap50_s = f"{ft_ap50:.3f}" if ft_ap50 is not None else "n/a"
            ft_ap_s = f"{ft_ap:.3f}" if ft_ap is not None else "n/a"
            print(f"{cname:<16}{b_ap50_s:>10}{ft_ap50_s:>10}{b_ap_s:>14}{ft_ap_s:>12}")

        print("\nSvetofor klasslari (baseline'da yo'q, faqat fine-tune):")
        for cname in ["light_red", "light_not_red"]:
            ft = finetune_per_class.get(cname, {})
            if ft:
                print(f"  {cname}: AP50={ft['AP50']:.3f} AP50-95={ft['AP50_95']:.3f} P={ft['P']:.3f} R={ft['R']:.3f}")
            else:
                print(f"  {cname}: val'da instance yo'q, o'lchab bo'lmadi")
    else:
        print("\n(Baseline hisoblanmadi, solishtirish o'tkazilmadi)")

    out_json = save_dir / "finetune_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(finetune_per_class, f, indent=2)
    print(f"\nNatijalar saqlandi: {out_json}")
    print(f"best.pt: {best_pt}")


if __name__ == "__main__":
    main()
