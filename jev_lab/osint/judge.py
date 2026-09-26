"""
収集した公開情報（collect.py の JSONL）を複数モデルで一括判断し、モデル間の一致度も出す。

  python -m jev_lab.osint.judge --in data/osint/tsu.jsonl --models jev_ja_30m nli_mdeberta gliner2_multi --out reports/osint-tsu.json

出力: 各 item × 各モデルの答え、モデル間 argmax 一致率（Cohen's κ）、信頼性スコア（osint.js と同じ重み）
"""
from __future__ import annotations

import argparse
import json
import os
from itertools import combinations

from ..adapters import get_adapter
from ..schema import ChoiceQ, NoulQ, ScoreQ

KIND = ["公的機関の発表", "報道", "一般の投稿・目撃", "意見・感想", "宣伝・無関係"]
W = {"official": 0.35, "supported": 0.25, "has_specifics": 0.15, "kind_report": 0.10, "no_spread": 0.10, "relevant": 0.05}


def questions(q: str):
    return [NoulQ(id="relevant", instructions=f"この情報は「{q}」に関係しているか"),
            ChoiceQ(id="kind", instructions="この情報の種類", options=KIND),
            ScoreQ(id="urgency", instructions="今すぐ対応や確認が必要な度合い", levels=["0", "1", "2", "3", "4", "5"]),
            NoulQ(id="has_specifics", instructions="日時・場所・数量など検証可能な具体情報が含まれているか"),
            NoulQ(id="asks_spread", instructions="拡散や転送を呼びかける表現があるか"),
            NoulQ(id="supported", instructions="根拠（context）の内容と整合しているか")]


def trust(it, ans):
    parts = {"official": 1.0 if it.get("official") else 0.0, "supported": ans["supported"]["p_yes"], "has_specifics": ans["has_specifics"]["p_yes"],
             "kind_report": {"公的機関の発表": 1, "報道": .8, "一般の投稿・目撃": .5, "意見・感想": .2, "宣伝・無関係": 0}.get(ans["kind"]["label"], .5),
             "no_spread": 1 - ans["asks_spread"]["p_yes"], "relevant": ans["relevant"]["p_yes"]}
    return sum(W[k] * parts[k] for k in W) / sum(W.values()), parts


def kappa(a: list, b: list) -> float:
    n = len(a); po = sum(x == y for x, y in zip(a, b)) / n
    labs = set(a) | set(b); pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labs)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True); ap.add_argument("--models", nargs="+", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    items = [json.loads(l) for l in open(a.inp, encoding="utf-8") if l.strip()]
    items = [it for it in items if not it.get("error")]
    q = items[0].get("query", "") if items else ""
    official = [f"{it['title']}: {it['text'][:200]}" for it in items if it.get("official") or it["source"] == "wikipedia"][:4]
    qs = questions(q)
    res = {"query": q, "n": len(items), "models": {}}
    for m in a.models:
        ad = get_adapter(m); rows = []
        for it in items:
            ctx = [c for c in official if not c.startswith(it["title"] + ":")]
            r = ad.decide(f"{it['title']}\n{it['text']}", qs, ctx or None)
            ans = {x.id: {"label": getattr(x, "choice", None) or getattr(x, "level", None) or ("yes" if getattr(x, "noul", False) else "no"), "p_yes": getattr(x, "p_yes", None), "score": getattr(x, "score", None), "confidence": getattr(x, "confidence", None)} for x in r.answers}
            t, parts = trust(it, ans)
            rows.append({"url": it.get("url"), "source": it["source"], "answers": ans, "trust": t, "trust_parts": parts, "latency_ms": r.latency_ms})
        res["models"][m] = rows
        print(f"{m}: 平均信頼性 {sum(r['trust'] for r in rows) / max(1, len(rows)):.2f}, p50 {sorted(r['latency_ms'] for r in rows)[len(rows) // 2]:.0f} ms")
    res["agreement"] = {}
    for m1, m2 in combinations(a.models, 2):
        for qid in ["kind", "relevant", "supported"]:
            x = [r["answers"][qid]["label"] for r in res["models"][m1]]; y = [r["answers"][qid]["label"] for r in res["models"][m2]]
            res["agreement"][f"{m1} vs {m2} / {qid}"] = {"agree": sum(i == j for i, j in zip(x, y)) / len(x), "kappa": kappa(x, y)}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for k, v in res["agreement"].items():
        print(f"{k}: 一致 {v['agree']:.2f} κ={v['kappa']:.2f}")
    print("saved", a.out)


if __name__ == "__main__":
    main()
