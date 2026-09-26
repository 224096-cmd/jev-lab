"""
ヘッドだけを高速に学習する（エンコーダ固定、ベクトルをキャッシュ）。

  python -m jev_lab.train.train_head --ckpt checkpoints/jev_ja_30m --train data/jevbench_ja/train.jsonl --val data/jevbench_ja/val.jsonl --epochs 30

- 1 回だけエンコーダを回して (u, V, y) を .pt にキャッシュ → 以後はヘッドの学習が数秒/epoch
- ヘッドは v2（LayerNorm あり）で初期化し直す。学習後は ckpt の head.pt / jev_ja_config.json を更新
- Play の「改良（端末内学習）」と同じ手順の PC 版（こちらは全データ・多エポック）
"""
from __future__ import annotations
import argparse, json, os, random, time
import torch, torch.nn.functional as F
from ..adapters.jev_ja import JevJaModel, DecisionHead
from ..bench.metrics import Aggregator
from ..schema import options_of
from .train_jev_ja import parse_questions, gold_index, load_jsonl, fit_temperature

def cache_vectors(model, items, path, max_n=0):
    if os.path.exists(path):
        return torch.load(path)
    ex = []; t0 = time.time()
    with torch.no_grad():
        for i, it in enumerate(items[: max_n or None]):
            qs = parse_questions(it["questions"]); state = it["state"]
            if it.get("context"): state = "[根拠]\n" + "\n".join("- " + c for c in it["context"]) + "\n[状況]\n" + state
            ids, spans, _ = model.encode_example(state, qs); H = model.enc(input_ids=ids.unsqueeze(0)).last_hidden_state[0]
            vec = {t: H[s:e].mean(0).cpu() for t, s, e in spans}
            for j, q in enumerate(qs):
                ex.append({"u": vec[f"q{j}"], "V": torch.stack([vec[f"o{j}_{k}"] for k in range(len(options_of(q)))]), "y": gold_index(q, it["gold"][q.id]), "family": it.get("family", "?"), "type": q.type, "labels": options_of(q)})
            if i % 200 == 0: print(f"  cache {i}/{len(items)} ({time.time()-t0:.0f}s)", flush=True)
    torch.save(ex, path); return ex

def evaluate(head, ex, T=1.0):
    agg = Aggregator(); head.eval()
    with torch.no_grad():
        for e in ex:
            p = torch.softmax(head(e["u"], e["V"]) / T, -1).tolist(); agg.add(e["family"], e["type"], e["labels"], p, e["labels"][e["y"]])
    return agg.summary()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--train", required=True); ap.add_argument("--val", required=True)
    ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--brier", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=64); ap.add_argument("--max-train", type=int, default=0); ap.add_argument("--max-val", type=int, default=0); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); torch.manual_seed(a.seed); random.seed(a.seed)
    model = JevJaModel.load_trained(a.ckpt).eval()
    tr = cache_vectors(model, list(load_jsonl(a.train)), os.path.join(a.ckpt, "cache_train.pt"), a.max_train)
    va = cache_vectors(model, list(load_jsonl(a.val)), os.path.join(a.ckpt, "cache_val.pt"), a.max_val)
    print(f"train {len(tr)} q / val {len(va)} q")
    head = DecisionHead(model.enc.config.hidden_size, norm=True); opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=0.01)
    print("before:", {k: round(v, 3) for k, v in evaluate(head, va).items() if k in ("accuracy", "brier", "ece")})
    log = []
    for ep in range(a.epochs):
        head.train(); random.shuffle(tr); tot = 0.0
        for i in range(0, len(tr), a.batch):
            loss = 0.0
            for e in tr[i:i + a.batch]:
                z = head(e["u"], e["V"]); tgt = torch.tensor([e["y"]]); p = torch.softmax(z, -1)
                loss = loss + F.cross_entropy(z.unsqueeze(0), tgt) + a.brier * ((p - F.one_hot(tgt, z.numel()).float()[0]) ** 2).sum()
            loss = loss / len(tr[i:i + a.batch]); opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        s = evaluate(head, va); log.append({"epoch": ep, "loss": tot / max(1, len(tr) // a.batch), "val": {k: s[k] for k in ("accuracy", "brier", "ece")}})
        print(f"ep{ep} loss={log[-1]['loss']:.3f} val acc={s['accuracy']:.3f} brier={s['brier']:.3f} ece={s['ece']:.3f}", flush=True)
    # 温度
    best_T, best = 1.0, 1e9
    with torch.no_grad():
        zs = [(head(e["u"], e["V"]), e["y"]) for e in va]
        for T in [x / 10 for x in range(5, 31)]:
            nll = sum(F.cross_entropy((z / T).unsqueeze(0), torch.tensor([y])).item() for z, y in zs)
            if nll < best: best, best_T = nll, T
    s = evaluate(head, va, best_T); print(f"T={best_T:.2f} val acc={s['accuracy']:.3f} brier={s['brier']:.3f} ece={s['ece']:.3f}")
    model.head = head; model.temperature = best_T; model.save(a.ckpt)
    rep = json.load(open(os.path.join(a.ckpt, "report.json"))) if os.path.exists(os.path.join(a.ckpt, "report.json")) else {}
    rep["head_v2"] = {"args": vars(a), "log": log, "temperature": best_T, "val_after_T": s}; json.dump(rep, open(os.path.join(a.ckpt, "report.json"), "w"), ensure_ascii=False, indent=2)
    print("saved", a.ckpt)

if __name__ == "__main__":
    main()
