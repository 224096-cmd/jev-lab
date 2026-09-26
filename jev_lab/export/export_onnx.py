"""
JEV-JA → ONNX（GitHub Pages 用）一括書き出し

  python -m jev_lab.export.export_onnx --model jev_ja_30m            # models.yaml のキー
  python -m jev_lab.export.export_onnx --model jev_ja_30m --no-int8  # 量子化なし（デバッグ）

出力: docs/models/<name>/
  encoder.onnx     input_ids (1,L) → last_hidden_state (1,L,d)   ※ int8 動的量子化
  head.onnx        u (d,), v (K,d) → logits (K,)
  tokenizer.json / tokenizer_config.json / special_tokens_map.json
  config.json      {pool, temperature, max_length, hidden, special_ids, trained, backbone}
docs/models/index.json にエントリを追加（play.html がこれを読んでモデル一覧を出す）
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import torch

from ..adapters import get_adapter

DOCS_MODELS = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "models")


class EncWrap(torch.nn.Module):
    def __init__(self, enc):
        super().__init__(); self.enc = enc
    def forward(self, input_ids):
        return self.enc(input_ids=input_ids, attention_mask=torch.ones_like(input_ids)).last_hidden_state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--no-int8", action="store_true")
    ap.add_argument("--out", default=DOCS_MODELS)
    a = ap.parse_args()

    ad = get_adapter(a.model)
    ad.load()
    m = ad.model.cpu().eval()
    out = os.path.join(a.out, a.model)
    os.makedirs(out, exist_ok=True)

    # --- encoder ---------------------------------------------------------
    enc_path = os.path.join(out, "encoder.onnx")
    dummy = torch.tensor([[m.tok.cls_token_id] + [5] * 30 + [m.tok.sep_token_id]])
    torch.onnx.export(EncWrap(m.enc), (dummy,), enc_path, input_names=["input_ids"], output_names=["last_hidden_state"],
                      dynamic_axes={"input_ids": {1: "L"}, "last_hidden_state": {1: "L"}}, opset_version=17, dynamo=False)
    if not a.no_int8:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        tmp = enc_path + ".fp32"
        os.replace(enc_path, tmp)
        quantize_dynamic(tmp, enc_path, weight_type=QuantType.QInt8)
        os.remove(tmp)

    # --- head ------------------------------------------------------------
    d = m.enc.config.hidden_size
    head_path = os.path.join(out, "head.onnx")
    torch.onnx.export(m.head, (torch.zeros(d), torch.zeros(3, d)), head_path, input_names=["u", "v"], output_names=["logits"],
                      dynamic_axes={"v": {0: "K"}, "logits": {0: "K"}}, opset_version=17, dynamo=False)

    # --- tokenizer / config ---------------------------------------------
    m.tok.save_pretrained(out)
    # Transformers.js が読める tokenizer_config にする（transformers v5 は "TokenizersBackend" と書く）
    tc_path = os.path.join(out, "tokenizer_config.json")
    tc = json.load(open(tc_path, encoding="utf-8"))
    tc["tokenizer_class"] = "PreTrainedTokenizerFast"
    json.dump(tc, open(tc_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    cfg = {"name": a.model, "backbone": ad.cfg.get("backbone"), "pool": m.pool, "temperature": m.temperature,
           "max_length": m.max_length, "hidden": d, "trained": ad.trained, "int8": not a.no_int8,
           "special_ids": {"cls": m.tok.cls_token_id, "sep": m.tok.sep_token_id, "unk": m.tok.unk_token_id,
                           **{k.strip("[]").lower(): v for k, v in m.ids.items()}},
           # Transformers.js の Unigram は byte fallback 未実装なので、<0x00>..<0xFF> の id 表を渡し JS 側で再現する
           "byte_ids": [m.tok.convert_tokens_to_ids(f"<0x{b:02X}>") for b in range(256)],
           "size_mb": round(sum(os.path.getsize(os.path.join(out, f)) for f in os.listdir(out)) / 1e6, 1)}
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), ensure_ascii=False, indent=2)

    # --- index.json ------------------------------------------------------
    idx_path = os.path.join(a.out, "index.json")
    idx = json.load(open(idx_path)) if os.path.exists(idx_path) else {"models": []}
    idx["models"] = [e for e in idx["models"] if e["name"] != a.model] + [
        {"name": a.model, "size_mb": cfg["size_mb"], "trained": cfg["trained"], "backbone": cfg["backbone"]}]
    json.dump(idx, open(idx_path, "w"), ensure_ascii=False, indent=2)

    # --- 検証: ONNX と PyTorch の出力一致 ------------------------------------
    import onnxruntime as ort, numpy as np
    s = ort.InferenceSession(enc_path)
    h_onnx = s.run(None, {"input_ids": dummy.numpy()})[0]
    with torch.no_grad():
        h_pt = m.enc(input_ids=dummy).last_hidden_state.numpy()
    diff = float(np.abs(h_onnx - h_pt).max())
    print(f"exported → {out}  ({cfg['size_mb']} MB)  max|onnx-pt| = {diff:.4f}  {'(int8)' if not a.no_int8 else ''}")


if __name__ == "__main__":
    main()
