"""
HF の Jev 級 cross-encoder（1 logit の AutoModelForSequenceClassification）。
例: argos1111/modernbert-ja-310m-jev — 「質問: …\n状況: …」×候補 のペアを読み、候補間 softmax。Noul は ["true","false"]。
"""
from __future__ import annotations
import torch
from ..schema import options_of
from .base import BaseAdapter

class CrossEncHfAdapter(BaseAdapter):
    probability_kind = "native"
    def _load(self):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        dev = self.cfg.get("device", "auto"); self.device = "cuda" if dev == "auto" and torch.cuda.is_available() else ("cpu" if dev == "auto" else dev)
        self.tok = AutoTokenizer.from_pretrained(self.cfg["hf_id"]); self.model = AutoModelForSequenceClassification.from_pretrained(self.cfg["hf_id"]).to(self.device).eval()
        self.prompt = self.cfg.get("prompt", "質問: {question}\n状況: {state}"); self.noul = self.cfg.get("noul_options", ["true", "false"]); self.max_length = self.cfg.get("max_length", 512)
    @torch.no_grad()
    def distributions(self, state, questions):
        out = []
        for q in questions:
            cands = self.noul if q.type == "noul" else options_of(q)
            enc = self.tok([self.prompt.format(question=q.instructions, state=state)] * len(cands), list(cands), padding=True, truncation="only_first", max_length=self.max_length, return_tensors="pt").to(self.device)
            z = self.model(**enc).logits[:, 0] / float(self.cfg.get("temperature", 1.0))
            out.append(torch.softmax(z, -1).cpu().tolist())
        return out
