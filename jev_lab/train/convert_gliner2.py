"""
JevBench-JA → GLiNER2 の分類学習データ形式に変換（既存 Jev 級モデルの改良ルート）。

  python -m jev_lab.train.convert_gliner2 --in data/jevbench_ja/train.jsonl --out data/gliner2/train.json

出力 1 件: {"text": state(+context), "classification": [{"task": id, "labels": [...], "true_label": [gold], "multi_label": false}, ...]}
Score は順序段階を単なるラベル集合として扱う（GLiNER2 に順序の概念は無い）。Noul は labels ["yes","no"]。
学習自体は gliner2 パッケージのトレーナ（README の "Training" 節）で行う。
"""
import argparse, json, os
ap = argparse.ArgumentParser(); ap.add_argument("--in", dest="inp", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
out = []
for line in open(a.inp, encoding="utf-8"):
    it = json.loads(line); text = it["state"] if not it.get("context") else "[根拠]\n" + "\n".join("- " + c for c in it["context"]) + "\n[状況]\n" + it["state"]
    cls = []
    for q in it["questions"]:
        labels = q["options"] if q["type"] == "choice" else q["levels"] if q["type"] == "score" else ["yes", "no"]
        g = it["gold"][q["id"]]
        if q["type"] == "noul": g = "yes" if str(g).lower() in ("yes", "true", "1") else "no"
        cls.append({"task": q["id"], "labels": labels, "true_label": [g], "multi_label": False, "prompt": q["instructions"]})
    out.append({"text": text, "classification": cls})
os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False)
print("wrote", len(out), "→", a.out)
