"""
JEV-JA: 日本語エンコーダ + 型付き判断ヘッド（本プロジェクトの成果物）

構造（Kotoba typed-decisions の encoder 方式を日本語で再現・拡張）:

  入力  [CLS] [STATE] s [Q] instr_1 [OPT] o_11 [OPT] o_12 ... [Q] instr_2 [OPT] o_21 ... [SEP]
  H     = Enc(入力)                                  (L x d)
  u_j   = mean(H[instr_j のトークン])                 (question 表現)
  v_jk  = mean(H[o_jk のトークン])                    (option 表現)
  z_jk  = g(u_j, v_jk) = w^T tanh(W [u_j ; v_jk ; u_j*v_jk]) + b
  p_jk  = softmax_k(z_jk / T)

  損失   L = CE + λ Brier      温度 T は検証データで事後推定

pool="marker" にすると v_jk を [OPT] マーカー位置の hidden state から読む（比較・教材用。
Kotoba の記録では学習しない）。
"""
from __future__ import annotations

import json
import os
from typing import Any

import torch
import torch.nn as nn

from ..schema import Question, options_of
from .base import BaseAdapter

SPECIAL = ["[STATE]", "[Q]", "[OPT]"]


class DecisionHead(nn.Module):
    """v2: u, v をそれぞれ LayerNorm してから [u;v;u⊙v] → MLP。
    ModernBERT-Ja の隠れ状態はノルムが大きく（|u|≈85）、正規化なしでは tanh が飽和して
    全選択肢に同じ logit を出す（学習不能）ことを実測したための修正。"""
    def __init__(self, d: int, hidden: int = 512, norm: bool = True):
        super().__init__()
        self.norm = norm
        self.ln_u = nn.LayerNorm(d) if norm else nn.Identity()
        self.ln_v = nn.LayerNorm(d) if norm else nn.Identity()
        self.mlp = nn.Sequential(nn.Linear(3 * d, hidden), nn.Tanh(), nn.Dropout(0.1), nn.Linear(hidden, 1))

    def forward(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # u: (d,)  v: (K, d)  -> logits (K,)
        un = self.ln_u(u).unsqueeze(0).expand_as(v)
        vn = self.ln_v(v)
        return self.mlp(torch.cat([un, vn, un * vn], dim=-1)).squeeze(-1)


class JevJaModel(nn.Module):
    """エンコーダ + ヘッド。学習・推論の両方で使う。"""

    def __init__(self, backbone: str, pool: str = "span", max_length: int = 1024):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(backbone)
        self.tok.add_special_tokens({"additional_special_tokens": SPECIAL})
        self.enc = AutoModel.from_pretrained(backbone)
        self.enc.resize_token_embeddings(len(self.tok))
        self.head = DecisionHead(self.enc.config.hidden_size)
        self.pool = pool
        self.max_length = max_length
        self.temperature = 1.0
        self.ids = {s: self.tok.convert_tokens_to_ids(s) for s in SPECIAL}

    # ---- 入力の組み立て。各スパンのトークン範囲を記録する ----------------
    def encode_example(self, state: str, questions: list[Question]):
        pieces: list[tuple[str, str]] = [("state", state)]
        spans: list[tuple[str, int, int]] = []  # (tag, start, end) tag = q{j} / o{j}_{k}
        for j, q in enumerate(questions):
            pieces.append(("Q", q.instructions))
            pieces.append(("qtag", f"q{j}"))
            for k, o in enumerate(options_of(q)):
                pieces.append(("OPT", o))
                pieces.append(("otag", f"o{j}_{k}"))

        ids = [self.tok.cls_token_id, self.ids["[STATE]"]]
        ids += self.tok(state, add_special_tokens=False)["input_ids"]
        marker_pos: dict[str, int] = {}
        for j, q in enumerate(questions):
            ids.append(self.ids["[Q]"])
            s = len(ids)
            ids += self.tok(q.instructions, add_special_tokens=False)["input_ids"]
            spans.append((f"q{j}", s, len(ids)))
            for k, o in enumerate(options_of(q)):
                marker_pos[f"o{j}_{k}"] = len(ids)
                ids.append(self.ids["[OPT]"])
                s = len(ids)
                ids += self.tok(o, add_special_tokens=False)["input_ids"]
                spans.append((f"o{j}_{k}", s, len(ids)))
        ids.append(self.tok.sep_token_id)
        if len(ids) > self.max_length:
            raise ValueError(f"入力長 {len(ids)} が max_length {self.max_length} を超えました（state を短くするか max_length を上げる）")
        return torch.tensor(ids), spans, marker_pos

    def forward_logits(self, state: str, questions: list[Question]) -> list[torch.Tensor]:
        ids, spans, marker_pos = self.encode_example(state, questions)
        dev = next(self.parameters()).device
        H = self.enc(input_ids=ids.unsqueeze(0).to(dev)).last_hidden_state[0]  # (L, d)
        span_vec = {tag: H[s:e].mean(0) for tag, s, e in spans}
        out = []
        for j, q in enumerate(questions):
            u = span_vec[f"q{j}"]
            if self.pool == "span":
                V = torch.stack([span_vec[f"o{j}_{k}"] for k in range(len(options_of(q)))])
            else:  # marker
                V = torch.stack([H[marker_pos[f"o{j}_{k}"]] for k in range(len(options_of(q)))])
            out.append(self.head(u, V))
        return out

    # ---- 保存 / 読込（backbone は HF 形式、head は state_dict）----------------
    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        self.enc.save_pretrained(os.path.join(path, "backbone"))
        self.tok.save_pretrained(os.path.join(path, "backbone"))
        torch.save(self.head.state_dict(), os.path.join(path, "head.pt"))
        json.dump({"pool": self.pool, "temperature": self.temperature, "max_length": self.max_length, "head_norm": self.head.norm},
                  open(os.path.join(path, "jev_ja_config.json"), "w"), ensure_ascii=False, indent=2)

    @classmethod
    def load_trained(cls, path: str):
        cfg = json.load(open(os.path.join(path, "jev_ja_config.json")))
        m = cls(os.path.join(path, "backbone"), cfg["pool"], cfg["max_length"])
        if not cfg.get("head_norm", False):   # v1 チェックポイント（正規化なし）
            m.head = DecisionHead(m.enc.config.hidden_size, norm=False)
        m.head.load_state_dict(torch.load(os.path.join(path, "head.pt"), map_location="cpu"))
        m.temperature = cfg.get("temperature", 1.0)
        return m


class JevJaAdapter(BaseAdapter):
    probability_kind = "native"

    def _load(self):
        dev = self.cfg.get("device", "auto")
        self.device = "cuda" if dev == "auto" and torch.cuda.is_available() else ("cpu" if dev == "auto" else dev)
        hp = self.cfg.get("head_path")
        if hp and os.path.exists(os.path.join(hp, "head.pt")):
            self.model = JevJaModel.load_trained(hp)
            self.trained = True
        else:
            self.model = JevJaModel(self.cfg["backbone"], self.cfg.get("pool", "span"), self.cfg.get("max_length", 1024))
            self.trained = False
        self.model.to(self.device).eval()

    @torch.no_grad()
    def distributions(self, state, questions):
        if self.cfg.get("method", "head") == "cos":
            return self._cos_distributions(state, questions)
        logits = self.model.forward_logits(state, questions)
        return [torch.softmax(z / self.model.temperature, -1).cpu().tolist() for z in logits]

    @torch.no_grad()
    def _cos_distributions(self, state, questions):
        """ゼロショット: z = scale * cos(u, v)。ヘッドを使わない（Play の method=cos と同じ）。"""
        m = self.model
        ids, spans, marker = m.encode_example(state, questions)
        H = m.enc(input_ids=ids.unsqueeze(0).to(self.device)).last_hidden_state[0]
        vec = {tag: H[s:e].mean(0) for tag, s, e in spans}
        scale = float(self.cfg.get("cos_scale", 20.0))
        out = []
        for j, q in enumerate(questions):
            u = vec[f"q{j}"]
            V = torch.stack([vec[f"o{j}_{k}"] for k in range(len(options_of(q)))])
            z = scale * torch.nn.functional.cosine_similarity(u.unsqueeze(0), V, dim=-1)
            out.append(torch.softmax(z, -1).cpu().tolist())
        return out

    def info(self) -> dict[str, Any]:
        d = super().info()
        d["trained"] = getattr(self, "trained", None)
        return d
