"""
Laya（Convai Innovations, Apache-2.0）— 最も広く使われている小型 Jev 級モデル（HF ❤ 3.8k）。
  convaiinnovations/laya               ModernBERT-large 421M 英語
  convaiinnovations/laya-multilingual  mmBERT-base 322M 100+ 言語（日本語を含む）
  convaiinnovations/laya-typed-decisions  typed-decisions 4 業務に特化

入力形式（laya.common.build_sequence）:
  [CLS] "{type} question: {instructions}" [SEP] [MASK] opt0 [MASK] opt1 … [SEP] state [SEP]
  → [MASK] 位置のベクトルを scorer(LayerNorm→Linear→GELU→Linear) で 1 logit ずつ → softmax
  noul は [false, true] の順で "false: no, the statement does not hold" / "true: yes, the statement holds"

`pip install laya` の Agent をそのまま使う（PC 用）。ブラウザ用は export/export_laya.py。
"""
from __future__ import annotations
from ..schema import options_of
from .base import BaseAdapter


class LayaAdapter(BaseAdapter):
    probability_kind = "native"

    def _load(self):
        import os
        os.environ.setdefault("USE_TF", "0")
        from laya.agent import Agent
        self.agent = Agent(self.cfg["hf_id"], subfolder=self.cfg.get("subfolder"))
        self.max_len = self.cfg.get("max_len")

    def distributions(self, state, questions):
        qs = {}
        for q in questions:
            if q.type == "choice":
                qs[q.id] = {"type": "choice", "instructions": q.instructions, "criteria": list(q.options)}
            elif q.type == "score":
                qs[q.id] = {"type": "score", "instructions": q.instructions, "criteria": list(q.levels)}
            else:
                qs[q.id] = {"type": "noul", "instructions": q.instructions}
        kw = {"max_len": self.max_len} if self.max_len else {}
        res = self.agent.predict(state, qs, **kw)
        out = []
        for q in questions:
            a = res["answers"][q.id]
            if q.type == "choice":
                out.append([float(a["probabilities"][o]) for o in q.options])
            elif q.type == "score":
                out.append([float(a["probabilities"][str(i)]) for i in range(len(q.levels))])
            else:
                p_true = float(a["noul"]); out.append([p_true, 1.0 - p_true])   # NOUL_OPTIONS = [yes, no]
        return out
