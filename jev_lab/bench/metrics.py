"""
評価指標（JevBench の 5 軸に対応）

  Smart    : accuracy（argmax = gold）, Score は MAE（段階）
  Reliable : Brier（多クラス総和）, ECE（top-label 信頼度、10 等幅ビン、空ビンは除外）
  Fast     : latency p50 / p95
  Cheap    : モデルサイズ（MB）で代用（ローカル実行なので単価はない）
  Valid    : 総和が 1±2% に入った割合
"""
from __future__ import annotations

import math
from collections import defaultdict


def brier(probs: list[float], gold_idx: int) -> float:
    return sum((p - (1.0 if i == gold_idx else 0.0)) ** 2 for i, p in enumerate(probs))


def ece(confidences: list[float], corrects: list[bool], bins: int = 10):
    """(ECE, bin table)"""
    table = []
    total = len(confidences)
    e = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(confidences) if (lo <= c < hi) or (b == bins - 1 and c == 1.0)]
        if not idx:
            table.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": 0})
            continue
        conf = sum(confidences[i] for i in idx) / len(idx)
        acc = sum(corrects[i] for i in idx) / len(idx)
        e += len(idx) / total * abs(conf - acc)
        table.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": len(idx), "confidence": conf, "accuracy": acc})
    return e, table


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = (len(s) - 1) * p
    f, c = math.floor(k), math.ceil(k)
    return s[f] if f == c else s[f] + (s[c] - s[f]) * (k - f)


class Aggregator:
    def __init__(self):
        self.rows = []          # per question
        self.latencies = []     # per state
        self.invalid = 0

    def add_state_latency(self, ms: float):
        self.latencies.append(ms)

    def add(self, family: str, qtype: str, labels: list[str], probs: list[float], gold: str, values=None):
        gi = labels.index(gold)
        pi = max(range(len(probs)), key=lambda i: probs[i])
        row = {"family": family, "type": qtype, "correct": pi == gi, "conf": probs[pi], "brier": brier(probs, gi)}
        if qtype == "score":
            vals = values or list(range(len(labels)))
            exp = sum(p * v for p, v in zip(probs, vals))
            row["mae"] = abs(exp - vals[gi])
        self.rows.append(row)

    def summary(self) -> dict:
        n = len(self.rows)
        if n == 0:
            return {}
        accs = [r["correct"] for r in self.rows]
        confs = [r["conf"] for r in self.rows]
        e, table = ece(confs, accs)
        per_family = defaultdict(list)
        for r in self.rows:
            per_family[r["family"]].append(r)
        fam = {}
        for k, rs in per_family.items():
            d = {"n": len(rs), "accuracy": sum(r["correct"] for r in rs) / len(rs)}
            ms = [r["mae"] for r in rs if "mae" in r]
            if ms:
                d["mae"] = sum(ms) / len(ms)
            fam[k] = d
        return {
            "n_questions": n,
            "accuracy": sum(accs) / n,
            "brier": sum(r["brier"] for r in self.rows) / n,
            "ece": e,
            "ece_table": table,
            "latency_p50_ms": percentile(self.latencies, 0.5),
            "latency_p95_ms": percentile(self.latencies, 0.95),
            "invalid_rate": self.invalid / max(1, len(self.latencies)),
            "per_family": fam,
        }
