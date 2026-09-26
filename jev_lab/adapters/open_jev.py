"""
open-jev-deberta-v3-large（Kotoba Labs）アダプタ。

HF リポに loader (typed_decisions/open_jev.py) が同梱されているので snapshot_download してそのまま使う。
decide(state, [{"type": "choice", "instructions", "options"}, {"type": "score", ...}, {"type": "noul", ...}])
という形が本ラボのスキーマとほぼ同じなので変換は薄い。
"""
from __future__ import annotations

import sys

from ..schema import Question, options_of
from .base import BaseAdapter


class OpenJevAdapter(BaseAdapter):
    probability_kind = "native"

    def _load(self):
        from huggingface_hub import snapshot_download
        path = snapshot_download(self.cfg["hf_id"])
        if path not in sys.path:
            sys.path.insert(0, path)
        from typed_decisions.open_jev import OpenJev  # リポ同梱の loader
        self.model = OpenJev.from_pretrained(path)

    def distributions(self, state, questions):
        qs = []
        for q in questions:
            if q.type == "choice":
                qs.append({"type": "choice", "instructions": q.instructions, "options": q.options})
            elif q.type == "score":
                qs.append({"type": "score", "instructions": q.instructions, "levels": q.levels})
            else:
                qs.append({"type": "noul", "instructions": q.instructions})
        res = self.model.decide(state, qs)
        out = []
        for q, r in zip(questions, res):
            labels = options_of(q)
            if q.type == "noul":
                p = float(r.get("noul", r.get("p_yes")))
                out.append([p, 1 - p])
            else:
                probs = r["probabilities"]
                # dict でも list でも受ける
                if isinstance(probs, dict):
                    out.append([float(probs[l]) for l in labels])
                else:
                    out.append([float(x) for x in probs])
        return out
