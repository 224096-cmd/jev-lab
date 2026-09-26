"""
Jevlet v6（NAME0x0/Jevlet、MIT）— bge-small-en-v1.5（33.5M）の上に「共有 state ＋ 質問ごとの分岐」を 1 系列に詰めた小型 Jev。
自作 jev_ja_30m と同じ規模の既存モデル（英語）。Pointer head、種類別温度。pip install git+https://github.com/NAME0x0/Jevlet  (transformers も必要)
系列: [CLS][STATE] state | [QUESTION] q [OPTION] o [END_OPTION] … [DECIDE] | … 、分岐は state と自分だけを見る（block_bidir）、位置は state の末尾から再開
"""
from __future__ import annotations
import json, os
import torch
from ..schema import options_of
from .base import BaseAdapter


def load_jevlet(hf_id="NAME0x0/Jevlet"):
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from jevlet.pretrained import PretrainedConfig, PretrainedJevlet
    cfg = json.load(open(hf_hub_download(hf_id, "config.json")))
    model = PretrainedJevlet(PretrainedConfig.from_dict(cfg["model_config"]), load_weights=False)
    model.load_state_dict(load_file(hf_hub_download(hf_id, "model.safetensors")))
    return model.eval(), cfg


class JevletAdapter(BaseAdapter):
    probability_kind = "native"
    def _load(self):
        from jevlet.data import DecisionExample, Question
        self.model, self.cfg_ckpt = load_jevlet(self.cfg.get("hf_id", "NAME0x0/Jevlet"))
        self.collator = self.model.make_collator(); self.temps = self.cfg_ckpt.get("temperatures", {}); self.Q, self.E = Question, DecisionExample
    @torch.no_grad()
    def distributions(self, state, questions):
        packed = []
        for q in questions:
            if q.type == "noul": packed.append(self.Q(q.instructions, ["True", "False"], 0, "noul"))
            elif q.type == "score": packed.append(self.Q(q.instructions, [str(l) for l in q.levels], 0, "score"))
            else: packed.append(self.Q(q.instructions, list(q.options), 0, "choice"))
        out = self.model(self.collator([self.E("x", state, packed, "inference", "local", "inference")]))
        res = []
        for q, z in zip(packed, out.logits):
            T = float(self.temps.get(q.kind, self.temps.get("default", 1.0))); res.append(torch.softmax(z.float() / T, -1).tolist())
        return res
