"""
ベンチ実行: JSONL（1 行 = 1 state + questions + gold）を読み、指定モデル群を評価する。

  python -m jev_lab.bench.run --data bench/data/sample_ja.jsonl --models jev_ja_30m nli_mdeberta

JSONL 1 行の形:
{"id": "...", "family": "disaster_urgency", "split": "in_domain",
 "state": "...", "context": ["..."],
 "questions": [{"type":"choice","id":"q1","instructions":"...","options":[...]}, ...],
 "gold": {"q1": "option_name", "q2": "level_name", "q3": "yes"}}
"""
from __future__ import annotations

import argparse
import json
import os
import time

from ..adapters import get_adapter
from ..schema import DecideRequest, options_of
from .metrics import Aggregator


def load_jsonl(path: str):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def run_model(name: str, items: list[dict], use_context: bool = True) -> dict:
    ad = get_adapter(name)
    ad.load()
    agg = Aggregator()
    for it in items:
        req = DecideRequest(state=it["state"], questions=it["questions"],
                            context=it.get("context") if use_context else None)
        try:
            res = ad.decide(req.state, req.questions, req.context)
        except Exception as e:  # invalid（総和ずれ等）はここに来る
            agg.invalid += 1
            agg.add_state_latency(float("nan"))
            continue
        agg.add_state_latency(res.latency_ms)
        for q, a in zip(req.questions, res.answers):
            gold = it["gold"][q.id]
            if q.type == "noul":
                gold = "yes" if str(gold).lower() in ("yes", "true", "1") else "no"
            agg.add(it.get("family", "?"), q.type, options_of(q), a.distribution.probabilities, gold,
                    values=getattr(q, "values", None))
    s = agg.summary()
    s["model"] = name
    s["probability_kind"] = ad.probability_kind
    s["use_context"] = use_context
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--no-context", action="store_true", help="RAG 文脈を注入しない（アブレーション）")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--publish", action="store_true", help="docs/results/ にもコピーし index.json を更新（git push で公開）")
    a = ap.parse_args()
    items = list(load_jsonl(a.data))
    os.makedirs(a.out, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    results = []
    for m in a.models:
        r = run_model(m, items, use_context=not a.no_context)
        results.append(r)
        print(f"{m:24s} acc={r.get('accuracy', float('nan')):.3f} brier={r.get('brier', float('nan')):.3f} "
              f"ece={r.get('ece', float('nan')):.3f} p50={r.get('latency_p50_ms', float('nan')):.0f}ms")
    path = os.path.join(a.out, f"bench-{stamp}.json")
    json.dump({"data": a.data, "results": results}, open(path, "w"), ensure_ascii=False, indent=2)
    print("saved", path)
    if a.publish:
        pub = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "results")
        os.makedirs(pub, exist_ok=True)
        fn = os.path.basename(path)
        json.dump({"data": a.data, "results": results}, open(os.path.join(pub, fn), "w"), ensure_ascii=False, indent=1)
        ip = os.path.join(pub, "index.json")
        idx = json.load(open(ip)) if os.path.exists(ip) else {"files": []}
        idx["files"] = [fn] + [f for f in idx["files"] if f != fn]
        json.dump(idx, open(ip, "w"), indent=1)
        print("published → docs/results/" + fn)


if __name__ == "__main__":
    main()
