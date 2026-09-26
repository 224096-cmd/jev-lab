"""
lostargon/Tiny-Jev（Qwen3-0.6B ベースの System-One 判断モデル、trust_remote_code）。
モデルカードの API: model.choice(tok, state, question, {option: description}) / model.score(...) / model.noul(tok, state, statement)
戻り値の形はカード記載の範囲で解釈し、分布が取れない場合は one-hot 相当にする（要確認）。
"""
from __future__ import annotations
import torch
from ..schema import options_of
from .base import BaseAdapter

class TinyJevAdapter(BaseAdapter):
    probability_kind = "native"
    def _load(self):
        from transformers import AutoModel, AutoTokenizer
        dev = self.cfg.get("device", "auto"); self.device = "cuda" if dev == "auto" and torch.cuda.is_available() else ("cpu" if dev == "auto" else dev)
        self.tok = AutoTokenizer.from_pretrained(self.cfg["hf_id"]); self.model = AutoModel.from_pretrained(self.cfg["hf_id"], trust_remote_code=True).to(self.device).eval()
    def _probs_from(self, res, labels):
        if isinstance(res, dict):
            d = res.get("probabilities") or res.get("distribution") or res.get("probs") or res
            if isinstance(d, dict): return [float(d.get(l, 0.0)) for l in labels]
        if isinstance(res, (list, tuple)) and len(res) == len(labels): return [float(x) for x in res]
        return [1.0 / len(labels)] * len(labels)
    @torch.no_grad()
    def distributions(self, state, questions):
        out = []
        for q in questions:
            labels = options_of(q)
            if q.type == "noul":
                r = self.model.noul(self.tok, state, q.instructions); p = float(r["probability"] if isinstance(r, dict) and "probability" in r else (r if isinstance(r, (int, float)) else 0.5)); out.append([p, 1 - p])
            elif q.type == "score":
                r = self.model.score(self.tok, state, q.instructions, labels); out.append(self._probs_from(r, labels))
            else:
                r = self.model.choice(self.tok, state, q.instructions, {l: (q.descriptions or {}).get(l, l) for l in labels}); out.append(self._probs_from(r, labels))
        return out
