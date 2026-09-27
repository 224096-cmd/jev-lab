"""
ブラウザ向け ONNX の共通処理（100 MB 以下に収めるための道具）

- normalize_opset(model)                 : onnxruntime 1.30 系の量子化器が要求する ai.onnx の opset 宣言を補う
- prune_embeddings(model, keep_ids, unk) : 語彙埋め込み（Gather の重み）を「使う id だけ」に間引く。id → 行 の対応表（int32）を Gather で引く形に書き換えるので tokenizer は無変更
- quantize_embeddings(model)             : 大きな Gather の重みを行スケール付き int8 に（Cast + Mul）
- quantize_matmul_4bit(path_in, path_out): MatMul を 4 bit（MatMulNBits、block 32）に。onnxruntime-web の WASM/WebGPU で動く
- split_large_gathers(model, limit)      : 1 テンソルが limit を超える Gather を行方向に分割（GitHub の 100 MB/ファイル制限）
- chunk_external(path, out, chunk_mb)    : 外部データを chunk_mb 以下のファイルに分ける
- vocab_from_corpus(tok, files, ...)     : 語彙の間引きに使う id 集合を、コーパス（JSONL の state/option 文字列）から作る
"""
from __future__ import annotations
import json, os
import numpy as np, onnx
from onnx import numpy_helper, helper, TensorProto
from onnx.external_data_helper import set_external_data


def normalize_opset(model):
    if not any(op.domain in ("", "ai.onnx") for op in model.opset_import): model.opset_import.append(onnx.helper.make_opsetid("", 17))
    for op in model.opset_import:
        if op.domain == "ai.onnx": op.domain = ""
    return model


def _gather_nodes(model, min_rows):
    inits = {t.name: t for t in model.graph.initializer}
    for node in list(model.graph.node):
        if node.op_type == "Gather" and node.input[0] in inits and len(inits[node.input[0]].dims) == 2 and inits[node.input[0]].dims[0] >= min_rows:
            yield node, inits[node.input[0]]


def prune_embeddings(model, keep_ids, unk_id, min_rows=20000):
    """W[V,d] → W'[K,d]（keep_ids の行）。Gather(W, ids) → Gather(W', Gather(map, ids))、map[V] は id → 行（無い id は unk の行）"""
    keep = sorted(set(int(i) for i in keep_ids) | {int(unk_id)})
    for node, t in _gather_nodes(model, min_rows):
        W = numpy_helper.to_array(t); V = W.shape[0]; rows = [i for i in keep if i < V]; pos = {i: r for r, i in enumerate(rows)}
        m = np.full(V, pos[int(unk_id)], dtype=np.int32); m[rows] = np.arange(len(rows), dtype=np.int32)
        model.graph.initializer.remove(t); model.graph.initializer.append(numpy_helper.from_array(np.ascontiguousarray(W[rows]), t.name)); model.graph.initializer.append(numpy_helper.from_array(m, t.name + "_map"))
        ids = node.input[1]; node.input[1] = t.name + "_rowid"
        pos_ = list(model.graph.node).index(node); model.graph.node.insert(pos_, helper.make_node("Gather", [t.name + "_map", ids], [t.name + "_rowid"], axis=0))
        print(f"embedding を間引き: {t.name} {V} → {len(rows)} 行（{(1 - len(rows) / V) * 100:.0f}% 削減）"); del W
    return model


def quantize_embeddings(model, min_rows=20000):
    for node, t in _gather_nodes(model, min_rows):
        W = numpy_helper.to_array(t).astype(np.float32); scale = (np.abs(W).max(axis=1, keepdims=True) / 127.0).astype(np.float32); scale[scale == 0] = 1.0
        Wq = np.clip(np.round(W / scale), -127, 127).astype(np.int8); del W
        model.graph.initializer.remove(t); model.graph.initializer.append(numpy_helper.from_array(Wq, t.name)); model.graph.initializer.append(numpy_helper.from_array(scale, t.name + "_scale"))
        y = node.output[0]; node.output[0] = y + "_q8"
        for vi in list(model.graph.value_info):
            if vi.name == y: model.graph.value_info.remove(vi)
        new = [helper.make_node("Cast", [y + "_q8"], [y + "_f"], to=TensorProto.FLOAT), helper.make_node("Gather", [t.name + "_scale", node.input[1]], [y + "_s"], axis=0), helper.make_node("Mul", [y + "_f", y + "_s"], [y])]
        idx = list(model.graph.node).index(node)
        for k, n in enumerate(new): model.graph.node.insert(idx + 1 + k, n)
        print("embedding を int8 化:", t.name, list(t.dims))
    return model


def quantize_matmul_4bit(path_in, path_out, block_size=None, symmetric=None):
    """環境変数 Q4_BLOCK（既定 64）/ Q4_SYM（既定 0 = 非対称）で調整できる"""
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer, DefaultWeightOnlyQuantConfig
    block_size = block_size or int(os.environ.get("Q4_BLOCK", 64)); sym = (os.environ.get("Q4_SYM", "0") == "1") if symmetric is None else symmetric
    m = onnx.load(path_in, load_external_data=True)
    q = MatMulNBitsQuantizer(m, block_size=block_size, is_symmetric=sym, accuracy_level=4, algo_config=DefaultWeightOnlyQuantConfig(block_size=block_size, is_symmetric=sym, accuracy_level=4))
    q.process(); q.model.save_model_to_file(path_out, use_external_data_format=True)
    return path_out   # 注意: <path_out>.data が併せて書かれる（chunk_external が置き換えた後に消す）


def split_large_gathers(model, limit):
    inits = {t.name: t for t in model.graph.initializer}
    for node in list(model.graph.node):
        if node.op_type != "Gather" or node.input[0] not in inits: continue
        t = inits[node.input[0]]; n = len(t.raw_data) if t.raw_data else 0
        if n <= limit or len(t.dims) != 2: continue
        W = numpy_helper.to_array(t); parts = int(np.ceil(n / limit)); rows = int(np.ceil(W.shape[0] / parts)); ids = node.input[1]; y = node.output[0]
        model.graph.initializer.remove(t); pos = list(model.graph.node).index(node); model.graph.node.remove(node); new = []
        for vi in list(model.graph.value_info):
            if vi.name == y: model.graph.value_info.remove(vi)
        outs = []
        for i in range(parts):
            lo, hi = i * rows, min(W.shape[0], (i + 1) * rows); nm = f"{t.name}_p{i}"
            model.graph.initializer.append(numpy_helper.from_array(np.ascontiguousarray(W[lo:hi]), nm))
            for suffix, val in (("_lo", lo), ("_max", hi - lo - 1), ("_zero", 0), ("_hi", hi)): model.graph.initializer.append(numpy_helper.from_array(np.array(val, dtype=np.int64), nm + suffix))
            model.graph.initializer.append(numpy_helper.from_array(np.array([-1], dtype=np.int64), nm + "_ax"))
            new += [helper.make_node("Sub", [ids, nm + "_lo"], [nm + "_rel"]), helper.make_node("Clip", [nm + "_rel", nm + "_zero", nm + "_max"], [nm + "_idx"]), helper.make_node("Gather", [nm, nm + "_idx"], [nm + "_g8"], axis=0), helper.make_node("Cast", [nm + "_g8"], [nm + "_g"], to=TensorProto.FLOAT), helper.make_node("Less", [ids, nm + "_hi"], [nm + "_lt0"]), helper.make_node("Unsqueeze", [nm + "_lt0", nm + "_ax"], [nm + "_lt"])]
            outs.append((nm + "_lt", nm + "_g"))
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
    fh.close(); onnx.save_model(model, path)
    for f in os.listdir(out):
        if f.startswith("model.onnx_data_") and f not in files: os.remove(os.path.join(out, f))
    return files


def vocab_from_corpus(tok, files, extra_texts=(), min_count=2, always=()):
    """コーパスに出る id を数える。always: 必ず残す id（特殊トークン・バイト列など）。返り値: (keep_ids, coverage 情報)"""
    from collections import Counter
    cnt = Counter()
    def feed(s):
        if s: cnt.update(tok(s, add_special_tokens=False)["input_ids"])
    for f in files:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                try: r = json.loads(line)
                except Exception: continue
                feed(r.get("state", "")); [feed(c) for c in (r.get("context") or [])]
                for q in r.get("questions", []): feed(q.get("instructions", "")); [feed(str(o)) for o in (q.get("options") or q.get("levels") or [])]
    for s in extra_texts: feed(s)
    keep = {i for i, c in cnt.items() if c >= min_count} | set(always)
    return keep, {"tokens_seen": sum(cnt.values()), "unique": len(cnt), "kept": len(keep)}


def common_ids(tok):
    """必ず残す id: 特殊トークン、ASCII 1 文字、かな、記号、よく使う漢字（教育漢字＋常用漢字の一部）、数字・英単語の基本"""
    import string
    chars = list(string.printable) + [chr(c) for c in range(0x3040, 0x30FF)] + [chr(c) for c in range(0xFF01, 0xFF60)] + list("、。・ー「」『』（）【】〜…！？：；／＼％＆＊＋－＝＠＃")
    joyo = "日一国会人年大十二本中長出三同時政事自行社見月分議後前民生連五発間対上部東者党地合市業内相方四定今回新場金員九入選立開手米力学問高代明実円関決子動京全目表戦経通外最言氏現理調体化田当八六約主題下首意法不来作性的要用制治度務強気小七成期公持野協取都和統以機平総加山思家話世受区領多県続進正安設保改数記院女初北午指権心界支第産結百派点教報済書府活原先共得解名交資予川向際査勝面委告軍文反元重近千考判認画海参売利組知案道信策集在件団別物側任引使求所次水半品昨論計死官増係感特情投示変打男基私各始島直両朝革価式確村提運終挙果西勢減台広容必応演電歳住争談能無再位置企真流格有疑口過局少放税検藤町常校料沢裁状工建語球営空職証土与急止送援供可役構木割聞身費付施切由説転食比難防補車優夫研収断井何南石足違消境神番規術護展態導鮮備宅害配副算視条幹独警宮究育席輸訪楽起万着乗店述残想線率病農州武声質念待試族象銀域助労例衛然早張映限額戸央刻況省輪除辺観備"
    ids = set()
    for ch in chars + list(joyo): ids.update(tok(ch, add_special_tokens=False)["input_ids"])
    ids.update(v for v in (tok.all_special_ids or []))
    return ids
