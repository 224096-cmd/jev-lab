"""
HF の Jev 級 cross-encoder（AutoModelForSequenceClassification、1 logit）をブラウザ用 ONNX に変換する。
例: argos1111/modernbert-ja-310m-jev（日本語 Jev、format "質問: …\n状況: …" / 候補）

  python -m jev_lab.export.export_crossenc --hf argos1111/modernbert-ja-310m-jev --name jev_ja_310m_argos --chunk-mb 90

出力 docs/models/<name>/: model.onnx（int8）＋ 外部データを chunk-mb 以下に分割（GitHub Pages の 100MB/ファイル制限対策）、tokenizer、config.json
"""
from __future__ import annotations
import argparse, json, os
import numpy as np, torch, onnx
from onnx.external_data_helper import set_external_data
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DOCS = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "models")

class Wrap(torch.nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, input_ids, attention_mask): return self.m(input_ids=input_ids, attention_mask=attention_mask).logits

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--hf", required=True); ap.add_argument("--name", required=True); ap.add_argument("--chunk-mb", type=int, default=90); ap.add_argument("--no-int8", action="store_true")
    ap.add_argument("--prompt", default="質問: {question}\n状況: {state}"); ap.add_argument("--noul", default="true,false"); ap.add_argument("--max-length", type=int, default=512)
    a = ap.parse_args(); out = os.path.join(DOCS, a.name); os.makedirs(out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.hf); m = AutoModelForSequenceClassification.from_pretrained(a.hf, attn_implementation="eager").eval()
    enc = tok("質問: テスト\n状況: 二重請求です。", "billing — 請求", return_tensors="pt")
    tmp = os.path.join(out, "model_fp32.onnx")
    torch.onnx.export(Wrap(m), (enc["input_ids"], enc["attention_mask"]), tmp, input_names=["input_ids", "attention_mask"], output_names=["logits"], dynamic_axes={"input_ids": {0: "B", 1: "L"}, "attention_mask": {0: "B", 1: "L"}, "logits": {0: "B"}}, opset_version=17, dynamo=False)
    path = os.path.join(out, "model.onnx")
    # onnxruntime 1.30 系の量子化器は opset_import に ai.onnx（domain ""）が無いと落ちる → 正規化
    mo = onnx.load(tmp, load_external_data=True)
    if not any(op.domain in ("", "ai.onnx") for op in mo.opset_import):
        mo.opset_import.append(onnx.helper.make_opsetid("", 17))
    for op in mo.opset_import:
        if op.domain == "ai.onnx": op.domain = ""
    onnx.save_model(mo, tmp)
    if a.no_int8: os.replace(tmp, path)
    else:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        try:
            quantize_dynamic(tmp, path, weight_type=QuantType.QInt8)
        except Exception as e:
            # 新しい onnxruntime での失敗時: 前処理（shape 推論）を挟んで再試行
            print("quantize_dynamic 失敗:", e, "→ preprocess して再試行")
            from onnxruntime.quantization.shape_inference import quant_pre_process
            pre = tmp + ".pre.onnx"; quant_pre_process(tmp, pre, skip_symbolic_shape=True); quantize_dynamic(pre, path, weight_type=QuantType.QInt8); os.remove(pre)
        os.remove(tmp)
    # 外部データを chunk に分割
    model = onnx.load(path, load_external_data=True); limit = a.chunk_mb * 1024 * 1024; files = []; cur, size = 0, 0
    fh = open(os.path.join(out, f"model.onnx_data_{cur}"), "wb"); files.append(f"model.onnx_data_{cur}")
    for t in model.graph.initializer:
        n = len(t.raw_data) if t.raw_data else 0
        if n < 1024: continue
        if size + n > limit:
            fh.close(); cur += 1; size = 0; fh = open(os.path.join(out, f"model.onnx_data_{cur}"), "wb"); files.append(f"model.onnx_data_{cur}")
        raw = t.raw_data; fh.write(raw)
        set_external_data(t, location=f"model.onnx_data_{cur}", offset=size, length=n); t.ClearField("raw_data"); size += n
    fh.close()
    onnx.save_model(model, path)
    # 検証
    import onnxruntime as ort
    s = ort.InferenceSession(path); z = s.run(None, {"input_ids": enc["input_ids"].numpy(), "attention_mask": enc["attention_mask"].numpy()})[0]
    with torch.no_grad(): zp = m(**enc).logits.numpy()
    tok.save_pretrained(out)
    tc = json.load(open(os.path.join(out, "tokenizer_config.json"))); tc["tokenizer_class"] = "PreTrainedTokenizerFast"; json.dump(tc, open(os.path.join(out, "tokenizer_config.json"), "w"), ensure_ascii=False, indent=1)
    ext = sorted(set(files)); sizes = {f: os.path.getsize(os.path.join(out, f)) for f in ext}
    cfg = {"name": a.name, "kind": "crossenc", "hf_id": a.hf, "prompt": a.prompt, "noul_options": a.noul.split(","), "max_length": a.max_length, "int8": not a.no_int8, "external_data": ext, "size_mb": round((os.path.getsize(path) + sum(sizes.values())) / 1e6, 1), "onnx_vs_pt_maxdiff": float(np.abs(z - zp).max()), "cls_id": tok.cls_token_id, "sep_id": tok.sep_token_id, "eos_id": tok.eos_token_id, "bos_id": tok.bos_token_id}
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), ensure_ascii=False, indent=2)
    print("exported", out, cfg["size_mb"], "MB; chunks", {k: round(v / 1e6) for k, v in sizes.items()}, "; maxdiff", cfg["onnx_vs_pt_maxdiff"])

if __name__ == "__main__":
    main()
