"""
ローカル RAG: ruri-v3-30m（Apache 2.0）埋め込み + 文字 bigram BM25 の RRF 融合。

  index = RagIndex.build(["文1", "文2", ...])
  index.save("rag/index_tsu"); index = RagIndex.load("rag/index_tsu")
  index.search("津波 避難所", k=3) -> [(text, score), ...]

P2 では同じ索引を Int8 で IndexedDB へ同期する（build_export で JSON 出力）。
"""
from __future__ import annotations

import json
import math
import os
import re
from collections import Counter

import numpy as np

EMB_MODEL = "cl-nagoya/ruri-v3-30m"


def _bigrams(s: str) -> list[str]:
    s = re.sub(r"\s+", "", s)
    return [s[i:i + 2] for i in range(len(s) - 1)] if len(s) > 1 else [s]


class BM25:
    def __init__(self, docs: list[str], k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.toks = [_bigrams(d) for d in docs]
        self.dl = [len(t) for t in self.toks]
        self.avgdl = sum(self.dl) / max(1, len(self.dl))
        df = Counter()
        for t in self.toks:
            df.update(set(t))
        N = len(docs)
        self.idf = {w: math.log(1 + (N - n + 0.5) / (n + 0.5)) for w, n in df.items()}
        self.tf = [Counter(t) for t in self.toks]

    def scores(self, q: str) -> np.ndarray:
        qt = _bigrams(q)
        out = np.zeros(len(self.toks))
        for i, tf in enumerate(self.tf):
            s = 0.0
            for w in qt:
                if w in tf:
                    f = tf[w]
                    s += self.idf.get(w, 0) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avgdl))
            out[i] = s
        return out


class RagIndex:
    def __init__(self, docs: list[str], emb: np.ndarray, meta: list[dict] | None = None):
        self.docs, self.emb, self.meta = docs, emb, meta or [{} for _ in docs]
        self.bm25 = BM25(docs)
        self._model = None

    @classmethod
    def build(cls, docs: list[str], meta=None, model_name: str = EMB_MODEL):
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(model_name)
        emb = m.encode(docs, normalize_embeddings=True, batch_size=32, show_progress_bar=True).astype(np.float32)
        idx = cls(docs, emb, meta)
        idx._model = m
        return idx

    def _embed_query(self, q: str) -> np.ndarray:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(EMB_MODEL)
        return self._model.encode(["検索クエリ: " + q], normalize_embeddings=True)[0].astype(np.float32)

    def search(self, q: str, k: int = 5, rrf_k: int = 60):
        dense = self.emb @ self._embed_query(q)
        sparse = self.bm25.scores(q)
        r_d = np.argsort(-dense); r_s = np.argsort(-sparse)
        rank_d = np.empty_like(r_d); rank_d[r_d] = np.arange(len(r_d))
        rank_s = np.empty_like(r_s); rank_s[r_s] = np.arange(len(r_s))
        rrf = 1 / (rrf_k + rank_d) + 1 / (rrf_k + rank_s)
        top = np.argsort(-rrf)[:k]
        return [(self.docs[i], float(rrf[i]), self.meta[i]) for i in top]

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        np.save(os.path.join(path, "emb.npy"), self.emb)
        json.dump({"docs": self.docs, "meta": self.meta}, open(os.path.join(path, "docs.json"), "w"), ensure_ascii=False)

    @classmethod
    def load(cls, path: str):
        d = json.load(open(os.path.join(path, "docs.json"), encoding="utf-8"))
        return cls(d["docs"], np.load(os.path.join(path, "emb.npy")), d["meta"])

    def export_int8(self, path: str):
        """P2 の端末同期用: 埋め込みを Int8 量子化して JSON 化（256 次元 × N）。"""
        scale = float(np.abs(self.emb).max())
        q = np.round(self.emb / scale * 127).astype(np.int8)
        json.dump({"scale": scale, "dim": int(self.emb.shape[1]), "docs": self.docs, "meta": self.meta,
                   "emb_int8": q.flatten().tolist()}, open(path, "w"), ensure_ascii=False)
