"""Dastur topgan hodisalarni labeler'ga taklif sifatida yozadi (02_data/labels/suggestions/).

Annotator ularni "+" (to'g'ri) yoki "✕" (xato) bilan tez ko'rib chiqadi — bu ham ground truth'ni
to'ldiradi, ham qoidalarning xatolarini ko'rsatadi. Id'lar barqaror (klass + boshlanish soniyasi),
shuning uchun qayta ishga tushirilganda rad etilganlar rad etilgan holida qoladi.

    python tools/export_suggestions.py --cache ../02_data/cache --out ../02_data/labels/suggestions
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from doppix.perception import Perception  # noqa: E402
from doppix.rules import detect  # noqa: E402

NOTES = {
    "jaywalking": "Dastur: piyoda zebradan tashqarida qatnov qismida harakatlanmoqda.",
    "failure_to_yield": "Dastur: mashina zebradan o'tayotganda zebrada yaqin masofada piyoda bor.",
    "red_light": "Dastur: mashina svetofor C qizil paytida stop-chiziqni kesib o'tdi.",
    "stop_line": "Dastur: qizilda stop-chiziqdan o'tib to'xtab turgan mashina.",
    "stopped_vehicle": "Dastur: chorrahada >=10 s turib qolgan mashina.",
    "congestion": "Dastur: chorrahada bir vaqtda bir nechta mashina turib qolgan.",
}

ap = argparse.ArgumentParser()
ap.add_argument("--cache", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
for npz in sorted(Path(args.cache).glob("*.npz")):
    name = f"{npz.stem}.MP4"
    path = out / f"{name}.json"
    doc = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"video": name, "suggestions": []}
    keep = [s for s in doc["suggestions"] if not s["id"].startswith("m_")]   # Claude'ning qo'lda takliflari qoladi
    model = []
    for s, e, label in detect(Perception.load(npz)):
        model.append({"id": f"m_{label}_{int(s)}", "label": label, "start": s, "end": e, "confidence": "model",
                      "note": NOTES.get(label, "Dastur topdi.")})
    doc["suggestions"] = keep + model
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{name}: {len(model)} ta dastur taklifi (+ {len(keep)} ta qo'lda)")
