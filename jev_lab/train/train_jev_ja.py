"""
JEV-JA 学習スクリプト（RTX 3060 12GB 想定）

  python -m jev_lab.train.train_jev_ja --train bench/data/train.jsonl --val bench/data/val.jsonl \
      --backbone sbintuitions/modernbert-ja-30m --out checkpoints/jev_ja_30m --epochs 2 --lr 5e-5

損失: CE + λ·Brier（Kotoba typed-decisions と同じ）
拡張（--augment p）: option 順シャッフル / noul 否定形（gold 反転）/ distractor drop
温度: 学習後に val で 1 次元探索して保存
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

import torch
import torch.nn.functional as F

from ..adapters.jev_ja import JevJaModel
from ..bench.metrics import Aggregator
from ..bench.run import load_jsonl
from ..schema import ChoiceQ, NoulQ, ScoreQ, Question, options_of

NEGATION_TEMPLATES = ["{}、という主張は誤りか", "次の命題は偽か: {}"]


def parse_questions(raw: list[dict]) -> list[Question]:
    out = []
    for r in raw:
        t = r["type"]
        out.append(ChoiceQ(**r) if t == "choice" else ScoreQ(**r) if t == "score" else NoulQ(**r))
    return out


def augment(qs: list[Question], gold: dict, p: float, rng: random.Random):
    """gold を構造的に保つ変換。"""
    new_qs, new_gold = [], dict(gold)
    for q in qs:
        if q.type == "choice" and rng.random() < p:
            opts = list(q.options)
            rng.shuffle(opts)
            # distractor drop（gold は残す、2 択以上を保つ）
            if len(opts) > 3 and rng.random() < 0.5:
                drop = rng.choice([o for o in opts if o != gold[q.id]])
                opts.remove(drop)
            q = ChoiceQ(id=q.id, instructions=q.instructions, options=opts, descriptions=q.descriptions)
        elif q.type == "noul" and rng.random() < p:
            q = NoulQ(id=q.id, instructions=rng.choice(NEGATION_TEMPLATES).format(q.instructions))
            g = str(gold[q.id]).lower() in ("yes", "true", "1")
            new_gold[q.id] = "no" if g else "yes"
        new_qs.append(q)
    return new_qs, new_gold


def gold_index(q: Question, g) -> int:
    labels = options_of(q)
    if q.type == "noul":
        g = "yes" if str(g).lower() in ("yes", "true", "1") else "no"
    return labels.index(g)


def loss_fn(logits: torch.Tensor, gi: int, lam: float) -> torch.Tensor:
    tgt = torch.tensor([gi], device=logits.device)
    ce = F.cross_entropy(logits.unsqueeze(0), tgt)
    p = torch.softmax(logits, -1)
    onehot = F.one_hot(tgt, logits.numel()).float()[0]
    br = ((p - onehot) ** 2).sum()
    return ce + lam * br


@torch.no_grad()
def evaluate(model: JevJaModel, items: list[dict], T: float = 1.0) -> dict:
    model.eval()
    agg = Aggregator()
    for it in items:
        qs = parse_questions(it["questions"])
        state = it["state"]
        if it.get("context"):
            state = "[根拠]\n" + "\n".join("- " + c for c in it["context"]) + "\n[状況]\n" + state
        t0 = time.perf_counter()
        logits = model.forward_logits(state, qs)
        agg.add_state_latency((time.perf_counter() - t0) * 1000)
        for q, z in zip(qs, logits):
            probs = torch.softmax(z / T, -1).cpu().tolist()
            g = it["gold"][q.id]
            if q.type == "noul":
                g = "yes" if str(g).lower() in ("yes", "true", "1") else "no"
            agg.add(it.get("family", "?"), q.type, options_of(q), probs, g, getattr(q, "values", None))
    return agg.summary()


@torch.no_grad()
def fit_temperature(model: JevJaModel, items: list[dict]) -> float:
    """val の NLL を最小にする T を粗く探索（0.5〜3.0）。"""
    model.eval()
    cache = []
    for it in items:
        qs = parse_questions(it["questions"])
        state = it["state"]
        if it.get("context"):
            state = "[根拠]\n" + "\n".join("- " + c for c in it["context"]) + "\n[状況]\n" + state
        for q, z in zip(qs, model.forward_logits(state, qs)):
            cache.append((z.detach(), gold_index(q, it["gold"][q.id])))
    best_T, best = 1.0, float("inf")
    for T in [x / 10 for x in range(5, 31)]:
        nll = sum(F.cross_entropy((z / T).unsqueeze(0), torch.tensor([g], device=z.device)).item() for z, g in cache)
        if nll < best:
            best, best_T = nll, T
    return best_T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--backbone", default="sbintuitions/modernbert-ja-30m")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pool", default="span", choices=["span", "marker"])
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--brier", type=float, default=1.0)
    ap.add_argument("--augment", type=float, default=0.0)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--freeze-backbone", action="store_true", help="ヘッドのみ学習（超軽量・教材用）")
    ap.add_argument("--max-train", type=int, default=0, help="学習 state 数の上限（0=全部）。CPU で時間を区切る用")
    ap.add_argument("--max-val", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = JevJaModel(a.backbone, a.pool, a.max_length).to(dev)
    if a.freeze_backbone:
        for p in model.enc.parameters():
            p.requires_grad = False
    params = [{"params": model.head.parameters(), "lr": a.head_lr}]
    if not a.freeze_backbone:
        params.append({"params": model.enc.parameters(), "lr": a.lr})
    opt = torch.optim.AdamW(params, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=(dev == "cuda"))

    train = list(load_jsonl(a.train)); rng.shuffle(train)
    val = list(load_jsonl(a.val)); rng.shuffle(val)
    if a.max_train: train = train[: a.max_train]
    if a.max_val: val = val[: a.max_val]
    log = {"args": vars(a), "loss": [], "val": []}
    step = 0
    for ep in range(a.epochs):
        model.train()
        rng.shuffle(train)
        run = 0.0
        for i, it in enumerate(train):
            qs = parse_questions(it["questions"])
            gold = it["gold"]
            if a.augment > 0:
                qs, gold = augment(qs, gold, a.augment, rng)
            state = it["state"]
            if it.get("context"):
                state = "[根拠]\n" + "\n".join("- " + c for c in it["context"]) + "\n[状況]\n" + state
            with torch.autocast(device_type=dev, dtype=torch.bfloat16, enabled=(dev == "cuda")):
                logits = model.forward_logits(state, qs)
                loss = sum(loss_fn(z.float(), gold_index(q, gold[q.id]), a.brier) for q, z in zip(qs, logits)) / len(qs)
            scaler.scale(loss / a.grad_accum).backward()
            run += loss.item()
            if (i + 1) % a.grad_accum == 0:
                scaler.step(opt); scaler.update(); opt.zero_grad(); step += 1
                if step % 20 == 0:
                    print(f"ep{ep} step{step} loss={run / a.grad_accum / 20:.4f}"); log["loss"].append(run / a.grad_accum / 20); run = 0.0
        model.save(a.out); print("checkpoint saved →", a.out)
        s = evaluate(model, val)
        print(f"== epoch {ep} val acc={s['accuracy']:.3f} brier={s['brier']:.3f} ece={s['ece']:.3f}")
        log["val"].append(s)
        json.dump(log, open(os.path.join(a.out, "report.json"), "w"), ensure_ascii=False, indent=2)

    T = fit_temperature(model, val)
    model.temperature = T
    s = evaluate(model, val, T)
    print(f"== T={T:.2f} val acc={s['accuracy']:.3f} brier={s['brier']:.3f} ece={s['ece']:.3f}")
    log["temperature"] = T
    log["val_after_T"] = s
    model.save(a.out)
    json.dump(log, open(os.path.join(a.out, "report.json"), "w"), ensure_ascii=False, indent=2)
    print("saved", a.out)


if __name__ == "__main__":
    main()
