"""
Laya（convaiinnovations/laya 系）をブラウザ用 ONNX（int8・分割）に変換する。

  python -m jev_lab.export.export_laya --hf convaiinnovations/laya-multilingual --name laya_multi_322m
  python -m jev_lab.export.export_laya --hf convaiinnovations/laya --subfolder multilingual --name laya_multi_322m   # 同じもの（束ねリポ）

出力 docs/models/<name>/: model.onnx（int8）+ model.onnx_data_N（≤ chunk-mb）、tokenizer、config.json（kind: laya）
入力: input_ids, attention_mask, marker_pos, marker_mask, qtype → 出力 logits（[MASK] 位置ごとの 1 logit）
必要: pip install laya（空き容量 2 GB 以上：fp32 の一時ファイル 1.3 GB を書く）
"""
from __future__ import annotations
import argparse, json, os
import numpy as np, torch, onnx
from onnx.external_data_helper import set_external_data

DOCS = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "models")


class ManualLayer(torch.nn.Module):
    """nn.TransformerEncoderLayer（norm_first, batch_first）を同じ重みで手書きにしたもの。
       純正の MultiheadAttention は ONNX 化すると系列長が定数に焼き付くため、shape を動的に扱う形で書き直す"""
    def __init__(self, L):
        super().__init__(); a = L.self_attn; self.H = a.num_heads; self.Wqkv = a.in_proj_weight; self.bqkv = a.in_proj_bias; self.out = a.out_proj
        self.ln1, self.ln2, self.l1, self.l2 = L.norm1, L.norm2, L.linear1, L.linear2; self.act = L.activation
    def forward(self, x, src_key_padding_mask=None):
        B, T, D = x.size(0), x.size(1), x.size(2); H = self.H; Dh = D // H
        h = self.ln1(x); qkv = torch.nn.functional.linear(h, self.Wqkv, self.bqkv); q, k, v = qkv.split(D, dim=-1)
        q = q.view(B, T, H, Dh).transpose(1, 2); k = k.view(B, T, H, Dh).transpose(1, 2); v = v.view(B, T, H, Dh).transpose(1, 2)
        s = torch.matmul(q, k.transpose(-1, -2)) / (Dh ** 0.5)
        if src_key_padding_mask is not None: s = s.masked_fill(src_key_padding_mask[:, None, None, :], -1e4)
        o = torch.matmul(torch.softmax(s, -1), v).transpose(1, 2).reshape(B, T, D); x = x + self.out(o)
        return x + self.l2(self.act(self.l1(self.ln2(x))))


class Wrap(torch.nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        logits, _act = self.m(input_ids, attention_mask, marker_pos, marker_mask, qtype)
        return logits


def split_large_gathers(model, limit):
    """1 テンソルが limit を超える Gather の重み（語彙埋め込み）を行方向に分割し、
       Where(ids < hi_0, Gather(W_0, clip(ids)), Where(ids < hi_1, Gather(W_1, clip(ids - lo_1)), …)) に書き換える
       （外部データは 1 テンソルを複数ファイルに跨がせられないため。GitHub の 100 MB/ファイル制限対策）"""
    from onnx import numpy_helper, helper, TensorProto
    inits = {t.name: t for t in model.graph.initializer}
    for node in list(model.graph.node):
        if node.op_type != "Gather" or node.input[0] not in inits: continue
        t = inits[node.input[0]]; n = len(t.raw_data) if t.raw_data else 0
        if n <= limit or len(t.dims) != 2: continue
        W = numpy_helper.to_array(t); parts = int(np.ceil(n / limit)); rows = int(np.ceil(W.shape[0] / parts)); ids = node.input[1]; y = node.output[0]
        model.graph.initializer.remove(t); pos = list(model.graph.node).index(node); model.graph.node.remove(node); new = []
        for vi in list(model.graph.value_info):
            if vi.name == y: model.graph.value_info.remove(vi)      # 出力型が int8 → float に変わるので古い型情報を消す
        outs = []
        for i in range(parts):
            lo, hi = i * rows, min(W.shape[0], (i + 1) * rows); nm = f"{t.name}_p{i}"
            model.graph.initializer.append(numpy_helper.from_array(np.ascontiguousarray(W[lo:hi]), nm))
            model.graph.initializer.append(numpy_helper.from_array(np.array(lo, dtype=np.int64), nm + "_lo")); model.graph.initializer.append(numpy_helper.from_array(np.array(hi - lo - 1, dtype=np.int64), nm + "_max")); model.graph.initializer.append(numpy_helper.from_array(np.array(0, dtype=np.int64), nm + "_zero")); model.graph.initializer.append(numpy_helper.from_array(np.array(hi, dtype=np.int64), nm + "_hi")); model.graph.initializer.append(numpy_helper.from_array(np.array([-1], dtype=np.int64), nm + "_ax"))
            new += [helper.make_node("Sub", [ids, nm + "_lo"], [nm + "_rel"]), helper.make_node("Clip", [nm + "_rel", nm + "_zero", nm + "_max"], [nm + "_idx"]), helper.make_node("Gather", [nm, nm + "_idx"], [nm + "_g8"], axis=0), helper.make_node("Cast", [nm + "_g8"], [nm + "_g"], to=TensorProto.FLOAT), helper.make_node("Less", [ids, nm + "_hi"], [nm + "_lt0"]), helper.make_node("Unsqueeze", [nm + "_lt0", nm + "_ax"], [nm + "_lt"])]   # Where は int8 非対応なので float に上げてから。条件は [B,L,1] に
            outs.append((nm + "_lt", nm + "_g"))
        # 後ろから Where で畳む
        cur = outs[-1][1]
        for i in range(parts - 2, -1, -1):
            o = y if i == 0 else f"{t.name}_w{i}"; new.append(helper.make_node("Where", [outs[i][0], outs[i][1], cur], [o])); cur = o
        for k, nn_ in enumerate(new): model.graph.node.insert(pos + k, nn_)
        print("大きな Gather を分割:", t.name, list(t.dims), "→", parts, "parts"); del W
    return model


def chunk_external(path, out, chunk_mb):
    model = onnx.load(path, load_external_data=True); limit = chunk_mb * 1024 * 1024; files = []; cur, size = 0, 0
    split_large_gathers(model, limit)
    fh = open(os.path.join(out, f"model.onnx_data_{cur}"), "wb"); files.append(f"model.onnx_data_{cur}")
    for t in model.graph.initializer:
        n = len(t.raw_data) if t.raw_data else 0
        if n < 1024: continue
        if size + n > limit:
            fh.close(); cur += 1; size = 0; fh = open(os.path.join(out, f"model.onnx_data_{cur}"), "wb"); files.append(f"model.onnx_data_{cur}")
        fh.write(t.raw_data); set_external_data(t, location=f"model.onnx_data_{cur}", offset=size, length=n); t.ClearField("raw_data"); size += n
    fh.close(); onnx.save_model(model, path); return files


def quantize_embeddings(mo, min_rows=20000):
    """大きな Gather の重み（語彙埋め込み）を行ごとのスケール付き int8 に置き換える:
       Gather(W_fp32, ids) → Cast(Gather(W_int8, ids)) * Gather(scale, ids)   （onnxruntime の量子化器と同等の精度で、メモリを 1/4）"""
    from onnx import numpy_helper, helper, TensorProto
    inits = {t.name: t for t in mo.graph.initializer}
    for node in list(mo.graph.node):
        if node.op_type != "Gather" or node.input[0] not in inits: continue
        t = inits[node.input[0]]
        if len(t.dims) != 2 or t.dims[0] < min_rows: continue
        W = numpy_helper.to_array(t).astype(np.float32); scale = (np.abs(W).max(axis=1, keepdims=True) / 127.0).astype(np.float32); scale[scale == 0] = 1.0
        Wq = np.clip(np.round(W / scale), -127, 127).astype(np.int8); del W
        mo.graph.initializer.remove(t)
        mo.graph.initializer.append(numpy_helper.from_array(Wq, t.name)); mo.graph.initializer.append(numpy_helper.from_array(scale, t.name + "_scale"))
        y = node.output[0]; node.output[0] = y + "_q8"
        mo.graph.node.extend([helper.make_node("Cast", [y + "_q8"], [y + "_f"], to=TensorProto.FLOAT), helper.make_node("Gather", [t.name + "_scale", node.input[1]], [y + "_s"], axis=0), helper.make_node("Mul", [y + "_f", y + "_s"], [y])])
        # Gather の後ろに挿入した順序を守るため、node 群を並べ直す（トポロジカル順）
        idx = list(mo.graph.node).index(node); tail = [mo.graph.node.pop() for _ in range(3)][::-1]
        for k, n in enumerate(tail): mo.graph.node.insert(idx + 1 + k, n)
        print("embedding を int8 化:", t.name, list(t.dims))


def quant_stage(out, no_int8, chunk_mb, int4=False, prune=False):
    """fp32 ONNX → （語彙の間引き）→ 埋め込み int8 → MatMul int8 / 4bit → 分割。torch を読まない別プロセスで実行してメモリを節約する"""
    from .onnx_utils import normalize_opset, prune_embeddings, quantize_embeddings, quantize_matmul_4bit, chunk_external
    tmp = os.path.join(out, "model_fp32.onnx"); path = os.path.join(out, "model.onnx")
    mo = onnx.load(tmp, load_external_data=True); normalize_opset(mo)
    if prune: keep = json.load(open(os.path.join(out, "_keep.json"))); prune_embeddings(mo, keep["ids"], keep["unk"]); os.remove(os.path.join(out, "_keep.json"))
    if not no_int8: quantize_embeddings(mo)
    onnx.save_model(mo, tmp, save_as_external_data=True, all_tensors_to_one_file=True, location="model_fp32.onnx_data"); del mo; import gc; gc.collect()
    if no_int8: os.replace(tmp, path)
    elif int4:
        quantize_matmul_4bit(tmp, path)
        if not os.environ.get("KEEP_FP32"): os.remove(tmp)
    else:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        try: quantize_dynamic(tmp, path, weight_type=QuantType.QInt8)
        except Exception as e:
            print("quantize_dynamic 失敗:", e, "→ preprocess して再試行")
            from onnxruntime.quantization.shape_inference import quant_pre_process
            pre = tmp + ".pre.onnx"; quant_pre_process(tmp, pre, skip_symbolic_shape=True); quantize_dynamic(pre, path, weight_type=QuantType.QInt8); os.remove(pre)
        os.remove(tmp)
    if os.path.exists(tmp + "_data") and not os.environ.get("KEEP_FP32"): os.remove(tmp + "_data")
    files = chunk_external(path, out, chunk_mb)
    for f in (path + ".data", path + "_data"):
        if os.path.exists(f): os.remove(f)
    json.dump(files, open(os.path.join(out, "_files.json"), "w"))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--hf", default="convaiinnovations/laya-multilingual"); ap.add_argument("--subfolder", default=None)
    ap.add_argument("--name", required=True); ap.add_argument("--chunk-mb", type=int, default=90); ap.add_argument("--no-int8", action="store_true"); ap.add_argument("--int4", action="store_true"); ap.add_argument("--prune-vocab", nargs="*", default=None); ap.add_argument("--stage", default="all", choices=["all", "quant"])
    a = ap.parse_args(); out = os.path.join(DOCS, a.name); os.makedirs(out, exist_ok=True)
    if a.stage == "quant": return quant_stage(out, a.no_int8, a.chunk_mb, a.int4, a.prune_vocab is not None)
    os.environ.setdefault("USE_TF", "0")
    from laya.agent import Agent
    from laya.common import build_sequence, collate_items, QTYPES
    ag = Agent(a.hf, subfolder=a.subfolder)              # 重み・tokenizer・設定を公式の方法で読む
    m = ag.model.cpu().float().eval(); tok = ag.tok; cfg = ag.cfg
    # ONNX に落とすため attention を eager に、TransformerEncoder の fastpath を無効に
    m.encoder.config._attn_implementation = "eager"
    for layer in m.encoder.layers:
        if hasattr(layer.attn, "config"): layer.attn.config._attn_implementation = "eager"
    try: torch.backends.mha.set_fastpath_enabled(False)
    except Exception: pass
    if m.head is not None:
        for i, L in enumerate(m.head.layers): m.head.layers[i] = ManualLayer(L)

    # 例（日本語）で入力を作る
    q = {"t": "choice", "ins": "この問い合わせの担当部署", "crit": {"請求": None, "配送": None, "技術": None}}
    seq, markers = build_sequence(tok, "先週注文した商品が二重に請求されています。返金をお願いします。", q, cfg.get("max_len", 1024), cfg.get("head_max_len", 256))
    b = collate_items([[{"ids": seq, "markers": markers, "qtype": QTYPES["choice"]}]], tok.pad_token_id)
    args = (b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"])
    with torch.no_grad(): z_pt = m(*args)[0].numpy()

    tmp = os.path.join(out, "model_fp32.onnx")
    torch.onnx.export(Wrap(m), args, tmp, input_names=["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"], output_names=["logits"],
                      dynamic_axes={"input_ids": {0: "B", 1: "L"}, "attention_mask": {0: "B", 1: "L"}, "marker_pos": {0: "B", 1: "K"}, "marker_mask": {0: "B", 1: "K"}, "qtype": {0: "B"}, "logits": {0: "B", 1: "K"}},
                      opset_version=17, dynamo=False)
    # 量子化は別プロセス（torch を持たない）で行い、メモリ不足で落ちるのを避ける
    if a.prune_vocab is not None:
        from .onnx_utils import vocab_from_corpus, common_ids
        keep, info = vocab_from_corpus(tok, a.prune_vocab, always=common_ids(tok)); keep |= set(b["input_ids"][0].tolist()); print("語彙:", info)
        json.dump({"ids": sorted(keep), "unk": tok.unk_token_id if tok.unk_token_id is not None else tok.pad_token_id}, open(os.path.join(out, "_keep.json"), "w"))
    np.save(os.path.join(out, "_zpt.npy"), z_pt); del m, ag; import gc; gc.collect()
    import subprocess, sys
    r = subprocess.run([sys.executable, "-m", "jev_lab.export.export_laya", "--name", a.name, "--stage", "quant", "--chunk-mb", str(a.chunk_mb)] + (["--no-int8"] if a.no_int8 else []) + (["--int4"] if a.int4 else []) + (["--prune-vocab"] if a.prune_vocab is not None else []))
    if r.returncode != 0: raise SystemExit("量子化ステージが失敗しました（メモリ不足の可能性。空きメモリ 6 GB 以上で再実行）")
    path = os.path.join(out, "model.onnx"); files = json.load(open(os.path.join(out, "_files.json"))); os.remove(os.path.join(out, "_files.json"))

    import onnxruntime as ort
    s = ort.InferenceSession(path)
    feed = {"input_ids": b["input_ids"].numpy().astype(np.int64), "attention_mask": b["attention_mask"].numpy().astype(np.int64), "marker_pos": b["marker_pos"].numpy().astype(np.int64), "marker_mask": b["marker_mask"].numpy().astype(bool), "qtype": b["qtype"].numpy().astype(np.int64)}
    z = s.run(None, feed)[0][0]
    sm = lambda v: np.exp(v - v.max()) / np.exp(v - v.max()).sum()
    print("pt:", sm(z_pt).round(4), " onnx:", sm(z).round(4))

    tok.save_pretrained(out)
    tc = json.load(open(os.path.join(out, "tokenizer_config.json"), encoding="utf-8")); tc["tokenizer_class"] = "PreTrainedTokenizerFast"
    tc.pop("extra_special_tokens", None); json.dump(tc, open(os.path.join(out, "tokenizer_config.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    sizes = {f: os.path.getsize(os.path.join(out, f)) for f in files}
    c = {"name": a.name, "kind": "laya", "hf_id": a.hf + (("/" + a.subfolder) if a.subfolder else ""), "backbone": cfg.get("encoder"), "max_len": cfg.get("max_len", 1024), "head_max_len": cfg.get("head_max_len", 256),
         "temperature": cfg.get("temperature", [1, 1, 1]), "temperature_by_options": cfg.get("temperature_by_options", {}), "int8": not a.no_int8, "int4": a.int4, "pruned_vocab": a.prune_vocab is not None, "external_data": files,
         "size_mb": round((os.path.getsize(path) + sum(sizes.values())) / 1e6, 1), "onnx_vs_pt_maxdiff": float(np.abs(sm(z) - sm(z_pt)).max()),
         "cls_id": tok.cls_token_id, "sep_id": tok.sep_token_id, "mask_id": tok.mask_token_id, "pad_id": tok.pad_token_id, "language": "multilingual", "license": "Apache-2.0",
         "note": "Laya（Convai Innovations）。[MASK] 位置の 1 logit を選択肢ごとに読む Jev 級モデル。100+ 言語、context 1024。PC ブラウザ向け"}
    json.dump(c, open(os.path.join(out, "config.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    idx_path = os.path.join(DOCS, "index.json"); idx = json.load(open(idx_path, encoding="utf-8")) if os.path.exists(idx_path) else {"models": []}
    idx["models"] = [e for e in idx["models"] if e["name"] != a.name] + [{"name": a.name, "kind": "laya", "size_mb": c["size_mb"], "trained": True, "backbone": c["hf_id"], "language": "multilingual", "license": "Apache-2.0", "note": c["note"]}]
    json.dump(idx, open(idx_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("exported", out, c["size_mb"], "MB; chunks", {k: round(v / 1e6) for k, v in sizes.items()}, "; maxdiff(prob)", c["onnx_vs_pt_maxdiff"])


if __name__ == "__main__":
    main()
