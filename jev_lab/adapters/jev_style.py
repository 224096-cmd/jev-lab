"""
chaoliangUNSW/Jev-Style-0.8B-Decision-v3（Qwen3.5-0.8B ベース、多言語・日本語対応、Apache-2.0）。
pip install "jev-style[torch]"。API: JevStyle.from_pretrained(id).decide(text, {name: noul("..."), ...})（要確認: choice/score ヘルパの名前）
"""
from __future__ import annotations
from ..schema import options_of
from .base import BaseAdapter

class JevStyleAdapter(BaseAdapter):
    probability_kind = "native"
    def _load(self):
        import jev_style
        self.js = jev_style.JevStyle.from_pretrained(self.cfg["hf_id"]); self.lib = jev_style
    def distributions(self, state, questions):
        out = []
        for q in questions:
            labels = options_of(q)
            if q.type == "noul":
                r = self.js.decide(state, {"q": self.lib.noul(q.instructions)})["q"]; p = float(getattr(r, "probability", getattr(r, "p", r))); out.append([p, 1 - p])
            else:
                helper = getattr(self.lib, "choice", None) or getattr(self.lib, "score", None)
                r = self.js.decide(state, {"q": helper(q.instructions, labels)})["q"]; d = getattr(r, "probabilities", None) or getattr(r, "distribution", None) or r
                out.append([float(d[l]) for l in labels] if isinstance(d, dict) else [1.0 / len(labels)] * len(labels))
        return out
