"""
GLiNER2.5-Decide アダプタ（隣接クラス: 分類ヘッド、ラベル集合を実行時に渡す）。

注意: GLiNER2 の per-label score はラベル集合上のカテゴリカル事後分布ではない。
分布へ直す正規化（normalize）を明示し、ベンチではその選択を記録する（JevBench が
「正規化の選択が較正値を左右する」として保留した点をこのラボで測る）。
"""
from __future__ import annotations

import math

from ..schema import Question, options_of
from .base import BaseAdapter


class Gliner2Adapter(BaseAdapter):
    probability_kind = "native"

    def _load(self):
        from gliner2 import AutoExtractor
        self.model = AutoExtractor.from_pretrained(self.cfg["hf_id"])
        self.normalize = self.cfg.get("normalize", "softmax_over_scores")

    def _scores(self, text: str, labels: list[str], prompt: str | None) -> list[float]:
        spec = {"labels": labels, "multi_label": True, "cls_threshold": 0.0}
        if prompt:
            spec["prompt"] = prompt
        # 生スコアを取りたいので return_scores 相当の API を使う。
        # gliner2 のバージョンで戻り値が変わるため、両方に対応する。
        res = self.model.classify_text(text, {"q": spec}, return_scores=True) \
            if "return_scores" in self.model.classify_text.__code__.co_varnames \
            else self.model.classify_text(text, {"q": spec})
        got = res["q"]
        if isinstance(got, dict):          # {label: score}
            return [float(got.get(l, 0.0)) for l in labels]
        if got and isinstance(got[0], dict):   # [{"label":..,"score":..}]
            m = {d["label"]: d["score"] for d in got}
            return [float(m.get(l, 0.0)) for l in labels]
        # スコア無し（ラベル名のみ）→ 選ばれたものに 1、他に 0（one-hot）
        return [1.0 if l in got else 0.0 for l in labels]

    def _to_dist(self, s: list[float]) -> list[float]:
        if self.normalize == "sum":
            t = sum(s) or 1.0
            return [x / t for x in s]
        # softmax_over_scores: logit(score) を softmax
        z = [math.log(max(x, 1e-6) / max(1 - x, 1e-6)) for x in s]
        m = max(z)
        e = [math.exp(v - m) for v in z]
        t = sum(e)
        return [v / t for v in e]

    def distributions(self, state, questions):
        out = []
        for q in questions:
            labels = options_of(q)
            out.append(self._to_dist(self._scores(state, labels, q.instructions)))
        return out
