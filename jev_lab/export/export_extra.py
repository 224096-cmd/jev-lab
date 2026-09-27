"""
既に ONNX で配布されている Jev 級モデルを、100 MB 以下のブラウザ向けに詰め直す（v3.0）

  # OpenJev Verdict（heman10x/rlcd-modernbert-151m、GLiClass 形、英語）
  python -m jev_lab.export.export_extra verdict --onnx dl/rlcd_model.onnx --tokenizer dl/rlcd_tokenizer.json \
      --calibrator dl/rlcd_calibrator.json --name verdict_151m
  # verdict-small（Manav2op/verdict-small、multilingual-e5-small の bi-encoder、多言語）
  python -m jev_lab.export.export_extra biencoder --onnx dl/vs/onnx/model_quantized.onnx --tokenizer dl/vs/tokenizer.json \
      --name verdict_small_e5 --prune-vocab docs/bench/jevbench_ja_small.jsonl --extra-text some.txt

出力: docs/models/<name>/{model.onnx, model.onnx_data_*, tokenizer.json, tokenizer_config.json, config.json}
"""
from __future__ import annotations
import argparse, json, os, shutil, sys, glob
import numpy as np, onnx
from onnx.external_data_helper import convert_model_to_external_data
from .onnx_utils import normalize_opset, prune_embeddings, quantize_embeddings, quantize_matmul_4bit, chunk_external, vocab_from_corpus, common_ids

def size_mb(d): return round(sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d) if f.startswith("model.onnx")) / 1e6, 1)

def finish(out, model, chunk_mb, cfg):
    tmp = os.path.join(out, "model.onnx"); onnx.save_model(model, tmp, save_as_external_data=True, all_tensors_to_one_file=True, location="model.onnx.data", size_threshold=1024)
    files = chunk_external(tmp, out, chunk_mb); os.remove(os.path.join(out, "model.onnx.data")) if os.path.exists(os.path.join(out, "model.onnx.data")) else None
    cfg["external_data"] = files; cfg["size_mb"] = size_mb(out); json.dump(cfg, open(os.path.join(out, "config.json"), "w"), ensure_ascii=False, indent=1); print("完了:", out, cfg["size_mb"], "MB", files)

def cmd_verdict(a):
    out = os.path.join("docs", "models", a.name); os.makedirs(out, exist_ok=True)
    m = onnx.load(a.onnx, load_external_data=True); normalize_opset(m)
    if a.prune_vocab is not None:
        from transformers import PreTrainedTokenizerFast
        tok = PreTrainedTokenizerFast(tokenizer_file=a.tokenizer); extra = []
        for f in a.extra_text or []: extra += [l.strip() for l in open(f, encoding="utf-8") if l.strip()]
        keep, info = vocab_from_corpus(tok, a.prune_vocab, extra_texts=extra, min_count=1, always=common_ids(tok)); keep |= {50368, 50369, 50281, 50282, 50283, 50284}; print("語彙:", info)
        prune_embeddings(m, keep, 50280, min_rows=20000)
    print("埋め込みを int8 に…"); quantize_embeddings(m, min_rows=5000)
    tmp = os.path.join(out, "_q8.onnx"); onnx.save_model(m, tmp, save_as_external_data=True, all_tensors_to_one_file=True, location="_q8.onnx.data", size_threshold=1024); del m
    print("MatMul を 4 bit に…"); q4 = os.path.join(out, "_q4.onnx"); quantize_matmul_4bit(tmp, q4)
    m = onnx.load(q4, load_external_data=True)
    for f in glob.glob(os.path.join(out, "_q*")): os.remove(f)
    shutil.copy(a.tokenizer, os.path.join(out, "tokenizer.json")); json.dump({"tokenizer_class": "PreTrainedTokenizerFast", "model_max_length": 8192, "cls_token": "[CLS]", "sep_token": "[SEP]", "pad_token": "[PAD]", "mask_token": "[MASK]", "unk_token": "[UNK]"}, open(os.path.join(out, "tokenizer_config.json"), "w"))
    cal = json.load(open(a.calibrator)) if a.calibrator else {}
    cfg = {"name": a.name, "kind": "gliclass", "hf_id": "heman10x/rlcd-modernbert-151m", "backbone": "knowledgator/gliclass-modern-base-v2.0 (ModernBERT-base)", "language": "en", "license": "Apache-2.0", "label_id": 50368, "sep_id": 50369, "max_candidates": 25, "max_len": 1024, "temperature": cal.get("temperature", 1.0), "per_k": cal.get("per_k", {}), "int8": True, "int4": True, "trained": True, "pruned_vocab": a.prune_vocab is not None,
           "note": "OpenJev Verdict（RLCD、151M）。<<LABEL>> 各選択肢 <<SEP>> 質問＋状況 を 1 系列にし、選択肢ごとの logit を 1 forward で読む。「insufficient evidence」の棄権枠つき、較正済み（ECE 3%）。埋め込み int8 ＋ 本体 4 bit"}
    finish(out, m, a.chunk_mb, cfg)

def cmd_biencoder(a):
    out = os.path.join("docs", "models", a.name); os.makedirs(out, exist_ok=True)
    from transformers import PreTrainedTokenizerFast
    tok = PreTrainedTokenizerFast(tokenizer_file=a.tokenizer)
    m = onnx.load(a.onnx, load_external_data=True); normalize_opset(m)
    if a.prune_vocab is not None:
        extra = []
        for f in a.extra_text or []: extra += [l.strip() for l in open(f, encoding="utf-8") if l.strip()]
        keep, info = vocab_from_corpus(tok, a.prune_vocab, extra_texts=extra, min_count=a.min_count, always=common_ids(tok)); print("語彙:", info)
        # 日本語（かな・漢字）のトークンは全部、英語は Unigram スコア上位 N を残す（コーパスに無い語も unk にしない）
        import re; tj = json.load(open(a.tokenizer, encoding="utf-8")); vocab = tj["model"].get("vocab", [])
        if isinstance(vocab, list):
            ja = re.compile(r"^[▁]?[\u3040-\u30ff\u4e00-\u9fff\u3000-\u303f\uff01-\uff60ー]+$"); en = re.compile(r"^[▁]?[A-Za-z0-9'\-.,%$]+$")
            keep |= {i for i, (w, _) in enumerate(vocab) if ja.match(w)}
            top_en = sorted(((sc, i) for i, (w, sc) in enumerate(vocab) if en.match(w)), reverse=True)[:a.keep_en]; keep |= {i for _, i in top_en}
        keep |= set(range(0, 16))  # <s> <pad> </s> <unk> など（tokenizer_file だけでは all_special_ids が空）
        print("最終的に残す行:", len(keep))
        unk = tok.unk_token_id if tok.unk_token_id is not None else 3
        prune_embeddings(m, keep, unk, min_rows=20000)
    shutil.copy(a.tokenizer, os.path.join(out, "tokenizer.json")); json.dump({"tokenizer_class": "XLMRobertaTokenizerFast", "model_max_length": 512, "cls_token": "<s>", "sep_token": "</s>", "pad_token": "<pad>", "mask_token": "<mask>", "unk_token": "<unk>"}, open(os.path.join(out, "tokenizer_config.json"), "w"))
    cfg = {"name": a.name, "kind": "biencoder", "hf_id": "Manav2op/verdict-small", "backbone": "intfloat/multilingual-e5-small", "language": "multilingual", "license": "Apache-2.0", "query_prefix": "query: ", "passage_prefix": "passage: ", "scale": 20.0, "max_len": 512, "int8": True, "pruned_vocab": a.prune_vocab is not None, "trained": True,
           "note": "verdict-small（Verdict、e5-small 118M を型付き判断で微調整、多言語）。状況と各選択肢を別々に埋め込み cos×20 を logit にする bi-encoder 形。語彙間引き＋int8"}
    finish(out, m, a.chunk_mb, cfg)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("verdict"); p1.add_argument("--onnx", required=True); p1.add_argument("--tokenizer", required=True); p1.add_argument("--calibrator"); p1.add_argument("--name", default="verdict_151m"); p1.add_argument("--chunk-mb", type=int, default=90); p1.add_argument("--prune-vocab", nargs="*", default=None); p1.add_argument("--extra-text", nargs="*")
    p2 = sub.add_parser("biencoder"); p2.add_argument("--onnx", required=True); p2.add_argument("--tokenizer", required=True); p2.add_argument("--name", default="verdict_small_e5"); p2.add_argument("--prune-vocab", nargs="*", default=None); p2.add_argument("--extra-text", nargs="*"); p2.add_argument("--min-count", type=int, default=1); p2.add_argument("--keep-en", type=int, default=25000); p2.add_argument("--chunk-mb", type=int, default=90)
    a = ap.parse_args(); {"verdict": cmd_verdict, "biencoder": cmd_biencoder}[a.cmd](a)
