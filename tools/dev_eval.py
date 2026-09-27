"""Keshlangan perception + qoidalar -> o'z labellarimiz bilan solishtirish (rasmiy metrika).

    python tools/dev_eval.py --cache ../02_data/cache --labels ../02_data/labels/events [--video C3896] [-v]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from doppix.perception import Perception  # noqa: E402
from doppix.rules import detect  # noqa: E402
from evaluate import evaluate_part_a  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--cache", required=True)
ap.add_argument("--labels", required=True)
ap.add_argument("--video", default="")
ap.add_argument("-v", action="store_true", help="har bir hodisani chiqarish")
args = ap.parse_args()

gt, pred = {}, {}
for npz in sorted(Path(args.cache).glob("*.npz")):
    if args.video and args.video not in npz.stem:
        continue
    name = f"{npz.stem}.MP4"
    lab = Path(args.labels) / f"{name}.json"
    if not lab.exists():
        continue
    p = Perception.load(npz)
    ev = json.loads(lab.read_text(encoding="utf-8"))["events"]
    gt[name] = {"duration": p.duration, "fps": p.fps, "events": [[e["start"], e["end"], e["label"]] for e in ev]}
    pred[name] = {"events": detect(p), "risk": []}
    if args.v:
        print(f"\n== {name}")
        for s, e, l in sorted(gt[name]["events"], key=lambda x: (x[2], x[0])):
            print(f"  GT   {l:18s} {s:7.1f} – {e:7.1f}")
        for s, e, l in sorted(pred[name]["events"], key=lambda x: (x[2], x[0])):
            print(f"  PRED {l:18s} {s:7.1f} – {e:7.1f}")

rep = evaluate_part_a(gt, pred)
print(f"\nScore A = {rep['score_a']:.4f}   (klasslar: {len(rep['classes'])})")
for c in rep["classes"]:
    r = rep["per_class"][c]
    tp, fp, fn = r["0.5"]["tp"], r["0.5"]["fp"], r["0.5"]["fn"]
    print(f"  {c:18s} F1@.3/.5/.7 = {r['0.3']['f1']:.2f} / {r['0.5']['f1']:.2f} / {r['0.7']['f1']:.2f}"
          f"   @.5 tp={tp} fp={fp} fn={fn}")
