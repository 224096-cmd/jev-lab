"""
ゼロショット NLI ベースライン（学習なし）。

各選択肢を仮説文に埋め込み、含意確率を選択肢間で softmax する。
mDeBERTa-v3 XNLI を使えば日本語でも動く。Jev 級モデルとの差 = 「判断ヘッドを学習した価値」。
"""
from __future__ import annotations

import torch

from ..schema import Question, options_of
from .base import BaseAdapter


class NliZeroShotAdapter(BaseAdapter):
    probability_kind = "native"

    def _load(self):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        dev = self.cfg.get("device", "auto")
        self.device = "cuda" if dev == "auto" and torch.cuda.is_available() else ("cpu" if dev == "auto" else dev)
        self.tok = AutoTokenizer.from_pretrained(self.cfg["hf_id"])
        self.model = AutoModelForSequenceClassification.from_pretrained(self.cfg["hf_id"]).to(self.device).eval()
        id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        self.ent = [i for i, l in id2label.items() if l.startswith("entail")][0]
        self.template = self.cfg.get("hypothesis_template", "この文の答えは「{}」である。")

    @torch.no_grad()
    def distributions(self, state, questions):
        out = []
        for q in questions:
            labels = options_of(q)
            hyps = [f"{q.instructions} {self.template.format(l)}" for l in labels]
            enc = self.tok([state] * len(hyps), hyps, return_tensors="pt", padding=True,
                           truncation="only_first", max_length=512).to(self.device)
            logits = self.model(**enc).logits[:, self.ent]
            out.append(torch.softmax(logits, -1).cpu().tolist())
        return out
