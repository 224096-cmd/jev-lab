"""
HF の Jev 級 cross-encoder（AutoModelForSequenceClassification、1 logit）をブラウザ用 ONNX に変換する。

  # argos（日本語、「質問: …\\n状況: …」＋候補、状況が先）
  python -m jev_lab.export.export_crossenc --hf argos1111/modernbert-ja-310m-jev --name jev_ja_310m_argos
  # open-jev-base（mmBERT-small、決定文が先・本文が後、sigmoid を K 候補で正規化）＋語彙の間引き＋int8 → 約 60 MB
  python -m jev_lab.export.export_crossenc --hf sshalimov04/open-jev-base --subfolder student --name openjev_base_60m \\
      --pair-order decision_first --prompt "A: {type} | Q: {question} | option: {option} | options: {options}" --score sigmoid \\
      --prune-vocab data/jevbench_ja/train.jsonl data/jevbench_ja/val.jsonl

出力 docs/models/<name>/: model.onnx（int8 または 4bit）＋ 外部データ（≤ chunk-mb）、tokenizer、config.json（kind: crossenc）
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys
import numpy as np, torch, onnx
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from .onnx_utils import normalize_opset, prune_embeddings, quantize_embeddings, quantize_matmul_4bit, chunk_external, vocab_from_corpus, common_ids

DOCS = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "models")


class Wrap(torch.nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, input_ids, attention_mask): return self.m(input_ids=input_ids, attention_mask=attention_mask).logits


def render(prompt, q_type, question, option, options, state):
    return prompt.replace("{type}", q_type).replace("{question}", question).replace("{option}", option).replace("{options}", " ; ".join(options)).replace("{state}", state)


def quant_stage(out, a):
    tmp = os.path.join(out, "model_fp32.onnx"); path = os.path.join(out, "model.onnx")
    mo = onnx.load(tmp, load_external_data=True); normalize_opset(mo)
    if a.prune_vocab is not None:
        keep = json.load(open(os.path.join(out, "_keep.json"))); prune_embeddings(mo, keep["ids"], keep["unk"]); os.remove(os.path.join(out, "_keep.json"))
    if not a.no_int8: quantize_embeddings(mo)
    onnx.save_model(mo, tmp, save_as_external_data=True, all_tensors_to_one_file=True, location="model_fp32.onnx_data"); del mo
    if a.no_int8: os.replace(tmp, path)
    elif a.int4:
        quantize_matmul_4bit(tmp, path); os.remove(tmp)
    else:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        try: quantize_dynamic(tmp, path, weight_type=QuantType.QInt8)
        except Exception as e:
            print("quantize_dynamic 失敗:", e, "→ preprocess して再試行"); from onnxruntime.quantization.shape_inference import quant_pre_process
            pre = tmp + ".pre.onnx"; quant_pre_process(tmp, pre, skip_symbolic_shape=True); quantize_dynamic(pre, path, weight_type=QuantType.QInt8); os.remove(pre)
        os.remove(tmp)
    for f in os.listdir(out):
        if f.startswith("model_fp32.onnx_data") or f.endswith(".onnx.data"): pass
    if os.path.exists(tmp + "_data"): os.remove(tmp + "_data")
    files = chunk_external(path, out, a.chunk_mb)
    for f in (path + ".data", path + "_data"):
        if os.path.exists(f): os.remove(f)
    json.dump(files, open(os.path.join(out, "_files.json"), "w"))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--hf", required=True); ap.add_argument("--subfolder", default=None); ap.add_argument("--name", required=True); ap.add_argument("--chunk-mb", type=int, default=90)
    ap.add_argument("--no-int8", action="store_true"); ap.add_argument("--int4", action="store_true", help="MatMul を 4bit（MatMulNBits）にする")
    ap.add_argument("--prompt", default="質問: {question}\n状況: {state}"); ap.add_argument("--noul", default="true,false"); ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--pair-order", default="state_first", choices=["state_first", "decision_first"], help="state_first: (prompt(state), option) / decision_first: (prompt(decision), state)")
    ap.add_argument("--score", default="logit", choices=["logit", "sigmoid"], help="sigmoid: 各候補の sigmoid 確率を K 候補で正規化（open-jev-base）")
    ap.add_argument("--prune-vocab", nargs="*", default=None, help="語彙の間引きに使うコーパス JSONL（state/context/questions を読む）"); ap.add_argument("--stage", default="all", choices=["all", "quant"])
    a = ap.parse_args(); out = os.path.join(DOCS, a.name); os.makedirs(out, exist_ok=True)
    if a.stage == "quant": return quant_stage(out, a)
    kw = {"subfolder": a.subfolder} if a.subfolder else {}
    tok = AutoTokenizer.from_pretrained(a.hf, **kw); m = AutoModelForSequenceClassification.from_pretrained(a.hf, attn_implementation="eager", **kw).eval()
    # 例
    if a.pair_order == "state_first": enc = tok(render(a.prompt, "choice", "この問い合わせを担当すべき部署", "", [], "先週注文した商品が二重に請求されています。"), "billing — 請求", return_tensors="pt")
    else: enc = tok(render(a.prompt, "choice", "Which department should handle this?", "billing", ["billing", "shipping", "tech"], ""), "I was charged twice for the same order.", return_tensors="pt")
    tmp = os.path.join(out, "model_fp32.onnx")
    torch.onnx.export(Wrap(m), (enc["input_ids"], enc["attention_mask"]), tmp, input_names=["input_ids", "attention_mask"], output_names=["logits"], dynamic_axes={"input_ids": {0: "B", 1: "L"}, "attention_mask": {0: "B", 1: "L"}, "logits": {0: "B"}}, opset_version=17, dynamo=False)
    with torch.no_grad(): zp = m(**enc).logits.numpy()
    if a.prune_vocab is not None:
        keep, info = vocab_from_corpus(tok, a.prune_vocab, always=common_ids(tok)); keep |= set(enc["input_ids"][0].tolist()); print("語彙:", info)
        json.dump({"ids": sorted(keep), "unk": tok.unk_token_id if tok.unk_token_id is not None else tok.pad_token_id}, open(os.path.join(out, "_keep.json"), "w"))
    del m; import gc; gc.collect()
    r = subprocess.run([sys.executable, "-m", "jev_lab.export.export_crossenc", "--hf", a.hf, "--name", a.name, "--stage", "quant", "--chunk-mb", str(a.chunk_mb)] + (["--no-int8"] if a.no_int8 else []) + (["--int4"] if a.int4 else []) + (["--prune-vocab"] if a.prune_vocab is not None else []))
    if r.returncode != 0: raise SystemExit("量子化ステージが失敗しました（メモリ不足の可能性）")
    path = os.path.join(out, "model.onnx"); files = json.load(open(os.path.join(out, "_files.json"))); os.remove(os.path.join(out, "_files.json"))
    import onnxruntime as ort
    s = ort.InferenceSession(path); z = s.run(None, {"input_ids": enc["input_ids"].numpy(), "attention_mask": enc["attention_mask"].numpy()})[0]
    tok.save_pretrained(out)
    tc = json.load(open(os.path.join(out, "tokenizer_config.json"), encoding="utf-8")); tc["tokenizer_class"] = "PreTrainedTokenizerFast"; tc.pop("extra_special_tokens", None); json.dump(tc, open(os.path.join(out, "tokenizer_config.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    sizes = {f: os.path.getsize(os.path.join(out, f)) for f in files}
    cfg = {"name": a.name, "kind": "crossenc", "hf_id": a.hf + ("/" + a.subfolder if a.subfolder else ""), "prompt": a.prompt, "pair_order": a.pair_order, "score": a.score, "noul_options": a.noul.split(","), "max_length": a.max_length, "int8": not a.no_int8, "int4": a.int4, "pruned_vocab": a.prune_vocab is not None, "external_data": files,
           "size_mb": round((os.path.getsize(path) + sum(sizes.values()) + os.path.getsize(os.path.join(out, "tokenizer.json"))) / 1e6, 1), "onnx_vs_pt_maxdiff": float(np.abs(z - zp).max()), "cls_id": tok.cls_token_id, "sep_id": tok.sep_token_id, "eos_id": tok.eos_token_id, "bos_id": tok.bos_token_id, "pad_id": tok.pad_token_id}
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), ensure_ascii=False, indent=2)
    ip = os.path.join(DOCS, "index.json"); idx = json.load(open(ip, encoding="utf-8")); idx["models"] = [e for e in idx["models"] if e["name"] != a.name] + [{"name": a.name, "kind": "crossenc", "size_mb": cfg["size_mb"], "trained": True, "backbone": cfg["hf_id"]}]
    json.dump(idx, open(ip, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("exported", out, cfg["size_mb"], "MB; chunks", {k: round(v / 1e6) for k, v in sizes.items()}, "; logit maxdiff", cfg["onnx_vs_pt_maxdiff"], "pt", zp.ravel()[:3], "onnx", z.ravel()[:3])


if __name__ == "__main__":
    main()
