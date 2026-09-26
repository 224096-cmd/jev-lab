"""
LLM に確率分布を JSON で「書かせる」方式（verbalized）。

JevBench の openai_compat アダプタに相当。native（モデル自身の分布）とは別物として扱い、
ベンチでは probability_kind="verbalized" と明示する。総和が 1±2% に入らなければ invalid。
"""
from __future__ import annotations

import json
import re

import torch

from ..schema import Question, options_of
from .base import BaseAdapter

SYSTEM = ("あなたは判断専用モデルです。与えられた状況と各質問について、選択肢ごとの確率を "
          "JSON のみで出力します。説明や前置きは書きません。各質問の確率の合計は 1.0 にします。")


class LlmVerbalizedAdapter(BaseAdapter):
    probability_kind = "verbalized"

    def _load(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        dev = self.cfg.get("device", "auto")
        self.device = "cuda" if dev == "auto" and torch.cuda.is_available() else ("cpu" if dev == "auto" else dev)
        self.tok = AutoTokenizer.from_pretrained(self.cfg["hf_id"])
        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg["hf_id"], torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device).eval()

    def _prompt(self, state: str, questions: list[Question]) -> str:
        qs = []
        for q in questions:
            qs.append({"id": q.id, "instructions": q.instructions, "options": options_of(q)})
        schema = {q.id: {o: 0.0 for o in options_of(q)} for q in questions}
        return (f"状況:\n{state}\n\n質問:\n{json.dumps(qs, ensure_ascii=False)}\n\n"
                f"次の形式の JSON だけを出力:\n{json.dumps(schema, ensure_ascii=False)}")

    @torch.no_grad()
    def distributions(self, state, questions):
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": self._prompt(state, questions)}]
        kw = {}
        if "enable_thinking" in self.cfg:
            kw["enable_thinking"] = self.cfg["enable_thinking"]
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
        enc = self.tok(text, return_tensors="pt").to(self.device)
        gen = self.model.generate(**enc, max_new_tokens=self.cfg.get("max_new_tokens", 256), do_sample=False)
        out = self.tok.decode(gen[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        m = re.search(r"\{.*\}", out, re.S)
        data = json.loads(m.group(0)) if m else {}
        res = []
        for q in questions:
            labels = options_of(q)
            d = data.get(q.id, {})
            probs = [float(d.get(l, 0.0)) for l in labels]
            if sum(probs) <= 0:            # 出力失敗 → 一様（invalid として記録される）
                probs = [1.0 / len(labels)] * len(labels)
            res.append(probs)
        return res
