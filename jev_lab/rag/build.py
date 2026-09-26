"""RAG 索引の作成: python -m jev_lab.rag.build --docs data/rag_docs.jsonl --out rag/index_default
docs.jsonl: {"text": "...", "source": "津市地域防災計画 p.12"} 1 行 1 件"""
import argparse, json
from .index import RagIndex
ap = argparse.ArgumentParser(); ap.add_argument("--docs", required=True); ap.add_argument("--out", required=True); ap.add_argument("--export-int8")
a = ap.parse_args()
rows = [json.loads(l) for l in open(a.docs, encoding="utf-8") if l.strip()]
idx = RagIndex.build([r["text"] for r in rows], [{k: v for k, v in r.items() if k != "text"} for r in rows])
idx.save(a.out)
if a.export_int8: idx.export_int8(a.export_int8)
print("saved", a.out, len(rows), "docs")
