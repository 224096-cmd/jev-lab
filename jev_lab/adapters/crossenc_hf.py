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
        kw = {"subfolder": self.cfg["subfolder"]} if self.cfg.get("subfolder") else {}
        self.tok = AutoTokenizer.from_pretrained(self.cfg["hf_id"], **kw); self.model = AutoModelForSequenceClassification.from_pretrained(self.cfg["hf_id"], **kw).to(self.device).eval()
        self.prompt = self.cfg.get("prompt", "質問: {question}\n状況: {state}"); self.noul = self.cfg.get("noul_options", ["true", "false"]); self.max_length = self.cfg.get("max_length", 512)
        self.decision_first = self.cfg.get("pair_order", "state_first") == "decision_first"; self.sigmoid = self.cfg.get("score", "logit") == "sigmoid"
    @torch.no_grad()
    def distributions(self, state, questions):
        out = []
        for q in questions:
            cands = self.noul if q.type == "noul" else options_of(q)
            render = lambda c: self.prompt.replace("{type}", q.type).replace("{question}", q.instructions).replace("{option}", c).replace("{options}", " ; ".join(cands)).replace("{state}", state)
            if self.decision_first: enc = self.tok([render(c) for c in cands], [state] * len(cands), padding=True, truncation="only_second", max_length=self.max_length, return_tensors="pt").to(self.device)
            else: enc = self.tok([render("")] * len(cands), list(cands), padding=True, truncation="only_first", max_length=self.max_length, return_tensors="pt").to(self.device)
            z = self.model(**enc).logits[:, 0] / float(self.cfg.get("temperature", 1.0))
            if self.sigmoid: p = torch.sigmoid(z); out.append((p / p.sum()).cpu().tolist())
            else: out.append(torch.softmax(z, -1).cpu().tolist())
        return out
