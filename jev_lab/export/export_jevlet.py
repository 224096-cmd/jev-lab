"""
Jevlet（NAME0x0/Jevlet、bge-small 33.5M）をブラウザ用 ONNX（int8、約 34 MB）に変換する。スマホで動く既存 Jev。

  python -m jev_lab.export.export_jevlet --name jevlet_33m

出力 docs/models/<name>/: encoder.onnx（int8; input_ids, attention_mask[B,1,L,L] bool, position_ids, token_type_ids → last_hidden_state）、head.bin（pointer head の query/key 重み）、tokenizer、config.json（kind: jevlet）
必要: pip install git+https://github.com/NAME0x0/Jevlet
"""
from __future__ import annotations
import argparse, json, os
import numpy as np, torch, onnx
from ..adapters.jevlet import load_jevlet

DOCS = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "models")


class Wrap(torch.nn.Module):
    def __init__(self, bb): super().__init__(); self.bb = bb
    def forward(self, input_ids, attention_mask, position_ids, token_type_ids):
        return self.bb(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, token_type_ids=token_type_ids).last_hidden_state


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--hf", default="NAME0x0/Jevlet"); ap.add_argument("--name", default="jevlet_33m"); ap.add_argument("--no-int8", action="store_true")
    a = ap.parse_args(); out = os.path.join(DOCS, a.name); os.makedirs(out, exist_ok=True)
    from jevlet.data import DecisionExample, Question
    model, ckpt = load_jevlet(a.hf); bb = model.backbone.eval(); bb.config._attn_implementation = "eager"
    for layer in bb.encoder.layer: layer.attention.self.config._attn_implementation = "eager"
    col = model.make_collator(); tok = model.tokenizer
    ex = DecisionExample("x", "Turn off the wifi and open the calculator.", [Question("Which action does the command ask for?", ["open app", "toggle wifi", "set reminder"], 0, "choice"), Question("Is it risky to run without asking?", ["True", "False"], 0, "noul")], "inference", "local", "inference")
    b = col([ex]); from jevlet.pretrained import build_pretrained_mask
    mask = build_pretrained_mask(b["branch_ids"], b["role_ids"], b["option_ids"], b["valid_mask"], "block_bidir")   # [B,1,L,L] bool（モデル本体と同じ渡し方）
    args = (b["input_ids"], mask, b["position_ids"] + model.position_offset, torch.zeros_like(b["input_ids"]))
    with torch.no_grad(): h_pt = bb(input_ids=args[0], attention_mask=args[1], position_ids=args[2], token_type_ids=args[3]).last_hidden_state.numpy(); out_pt = model(b)
    tmp = os.path.join(out, "encoder_fp32.onnx"); path = os.path.join(out, "encoder.onnx")
    torch.onnx.export(Wrap(bb), args, tmp, input_names=["input_ids", "attention_mask", "position_ids", "token_type_ids"], output_names=["last_hidden_state"],
                      dynamic_axes={"input_ids": {0: "B", 1: "L"}, "attention_mask": {0: "B", 2: "L", 3: "L2"}, "position_ids": {0: "B", 1: "L"}, "token_type_ids": {0: "B", 1: "L"}, "last_hidden_state": {0: "B", 1: "L"}}, opset_version=17, dynamo=False)
    mo = onnx.load(tmp)
    if not any(op.domain in ("", "ai.onnx") for op in mo.opset_import): mo.opset_import.append(onnx.helper.make_opsetid("", 17))
    for op in mo.opset_import:
        if op.domain == "ai.onnx": op.domain = ""
    onnx.save_model(mo, tmp)
    if a.no_int8: os.replace(tmp, path)
    else:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        try: quantize_dynamic(tmp, path, weight_type=QuantType.QInt8)
        except Exception as e:
            print("quantize_dynamic 失敗:", e, "→ preprocess して再試行"); from onnxruntime.quantization.shape_inference import quant_pre_process
            pre = tmp + ".pre.onnx"; quant_pre_process(tmp, pre, skip_symbolic_shape=True); quantize_dynamic(pre, path, weight_type=QuantType.QInt8); os.remove(pre)
        os.remove(tmp)
    # head（pointer）: logits = (K o) · (Q d) / sqrt(w)
    Wq = model.head.query.weight.detach().float().numpy(); Wk = model.head.key.weight.detach().float().numpy(); w = Wq.shape[0]
    np.concatenate([Wq.flatten(), Wk.flatten()]).astype(np.float32).tofile(os.path.join(out, "head.bin"))
    # 検証（ONNX + JS と同じ手順の head を numpy で）
    import onnxruntime as ort
    s = ort.InferenceSession(path); H = s.run(None, {"input_ids": args[0].numpy(), "attention_mask": args[1].numpy(), "position_ids": args[2].numpy(), "token_type_ids": args[3].numpy()})[0]
    mx = 0
    for rec, zp in zip(out_pt.records, out_pt.logits):
        d = H[0, rec["decide_position"]]; opts = np.stack([H[0, s0:s1].mean(0) for s0, s1 in rec["option_spans"]]); z = (opts @ Wk.T) @ (Wq @ d) / np.sqrt(w)
        sm = lambda v: np.exp(v - v.max()) / np.exp(v - v.max()).sum(); pa, pb = sm(zp.detach().numpy()), sm(z); mx = max(mx, float(np.abs(pa - pb).max())); print(rec["kind"], pa.round(3), pb.round(3))
    tok.save_pretrained(out)
    tc = json.load(open(os.path.join(out, "tokenizer_config.json"), encoding="utf-8")); tc["tokenizer_class"] = "BertTokenizer"; tc.pop("extra_special_tokens", None); json.dump(tc, open(os.path.join(out, "tokenizer_config.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    ids = tok.convert_tokens_to_ids(["[STATE]", "[QUESTION]", "[OPTION]", "[END_OPTION]", "[DECIDE]"]); mc = ckpt["model_config"]
    size = round(sum(os.path.getsize(os.path.join(out, f)) for f in os.listdir(out)) / 1e6, 1)
    c = {"name": a.name, "kind": "jevlet", "hf_id": a.hf, "backbone": mc["backbone"], "hidden": w, "topology": mc["attention_topology"], "option_pool": mc["option_pool"], "max_state_tokens": mc["max_state_tokens"], "max_question_tokens": mc["max_question_tokens"], "max_option_tokens": mc["max_option_tokens"], "max_packed_len": mc["max_packed_len"], "max_position": model.max_position, "position_offset": model.position_offset,
         "temperatures": ckpt.get("temperatures", {}), "special_ids": {"cls": tok.cls_token_id, "sep": tok.sep_token_id, "pad": tok.pad_token_id, "unk": tok.unk_token_id, "state": ids[0], "question": ids[1], "option": ids[2], "end_option": ids[3], "decide": ids[4]},
         "int8": not a.no_int8, "size_mb": size, "onnx_vs_pt_maxdiff": mx, "language": "en", "license": "MIT", "trained": True,
         "note": "Jevlet v6（bge-small 33.5M、英語）。共有 state＋質問分岐を 1 系列に詰め、[DECIDE] と各選択肢の pointer head で決める。自作 30m と同規模の既存 Jev。スマホ可"}
    json.dump(c, open(os.path.join(out, "config.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    ip = os.path.join(DOCS, "index.json"); idx = json.load(open(ip, encoding="utf-8")); idx["models"] = [e for e in idx["models"] if e["name"] != a.name] + [{"name": a.name, "kind": "jevlet", "size_mb": size, "trained": True, "backbone": mc["backbone"], "language": "en", "license": "MIT", "note": c["note"]}]
    json.dump(idx, open(ip, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("exported", out, size, "MB; maxdiff(prob)", mx, "; structure ids", ids)


if __name__ == "__main__":
    main()
