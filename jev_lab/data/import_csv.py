"""
自分の CSV/TSV/Excel を JevBench-JA 形式（JSONL）にする最短ルート。

  python -m jev_lab.data.import_csv --file data/my_survey.csv --family my_survey \
      --text answer \
      --q "sentiment:score:この回答の全体的な感情:否定的|中立|肯定的:sentiment_label" \
      --q "request:noul:要望が含まれているか:has_request" \
      --q "topic:choice:主な話題:時間|教材|人間関係|その他:topic_label" \
      --split 0.7,0.1,0.2

--q の書式  id:type:instructions:選択肢または段階を | 区切り:gold列名   （noul は選択肢なし → id:noul:instructions:gold列）
gold 列の値は選択肢名そのもの（noul は yes/no/1/0/true/false）。
出力: data/<family>/{train,val,test}.jsonl と、Play のバッチ評価にそのまま読める docs/bench/<family>.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import random

import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")


def parse_q(spec: str) -> dict:
    parts = spec.split(":")
    qid, typ = parts[0], parts[1]
    if typ == "noul":
        return {"id": qid, "type": "noul", "instructions": parts[2], "gold_col": parts[3]}
    labels = parts[3].split("|")
    q = {"id": qid, "type": typ, "instructions": parts[2], "gold_col": parts[4]}
    q["options" if typ == "choice" else "levels"] = labels
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--family", required=True)
    ap.add_argument("--text", required=True, help="state にする列。複数は , 区切り（改行で連結）")
    ap.add_argument("--q", action="append", required=True)
    ap.add_argument("--split", default="0.7,0.1,0.2")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    df = pd.read_excel(a.file) if a.file.endswith((".xlsx", ".xls")) else pd.read_csv(a.file, sep="\t" if a.file.endswith(".tsv") else ",")
    qs = [parse_q(s) for s in a.q]
    cols = a.text.split(",")
    items = []
    for i, row in df.iterrows():
        gold = {}
        for q in qs:
            g = row[q["gold_col"]]
            if q["type"] == "noul":
                g = "yes" if str(g).strip().lower() in ("yes", "1", "true", "はい", "あり") else "no"
            gold[q["id"]] = str(g)
        items.append({"id": f"{a.family}-{i}", "family": a.family, "split": "in_domain",
                      "state": "\n".join(str(row[c]) for c in cols),
                      "questions": [{k: v for k, v in q.items() if k != "gold_col"} for q in qs], "gold": gold})
    random.Random(a.seed).shuffle(items)
    tr, va, te = [float(x) for x in a.split.split(",")]
    n = len(items); A = int(n * tr); B = A + int(n * va)
    out = os.path.join(ROOT, "data", a.family); os.makedirs(out, exist_ok=True)
    for name, part in [("train", items[:A]), ("val", items[A:B]), ("test", items[B:])]:
        with open(os.path.join(out, f"{name}.jsonl"), "w", encoding="utf-8") as f:
            for it in part:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
    dd = os.path.join(ROOT, "docs", "bench"); os.makedirs(dd, exist_ok=True)
    with open(os.path.join(dd, f"{a.family}.jsonl"), "w", encoding="utf-8") as f:
        for it in items[B:]:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"{a.family}: train {A} / val {B - A} / test {n - B} → {out}  (端末用: docs/bench/{a.family}.jsonl)")


if __name__ == "__main__":
    main()
