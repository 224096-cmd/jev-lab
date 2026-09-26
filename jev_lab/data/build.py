"""
JevBench-JA ビルダー: config/datasets.yaml → data/jevbench_ja/{train,val,test,ood}.jsonl

  python -m jev_lab.data.build                       # 全 family
  python -m jev_lab.data.build --families jnli jsts   # 一部
  python -m jev_lab.data.build --device-subset 20     # docs/bench/jevbench_ja_small.jsonl（端末評価用、family ごと N state）

- in-domain: train / val / test（instructions は学習時に候補からランダム、評価時は先頭）
- ood      : test と同じ state に ood_instructions（未見の言い換え／否定形）を付けたもの
             → 「質問を読んでいるか、slot を暗記したか」を分ける検査
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import random

import yaml

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
CFG = os.path.join(ROOT, "config", "datasets.yaml")
OUT = os.path.join(ROOT, "data", "jevbench_ja")


# ---------------------------------------------------------------- sources
def load_hf(src: dict) -> dict[str, list[dict]]:
    """parquet を直接読む（datasets のスクリプト廃止に依存しない）。"""
    import pandas as pd
    from huggingface_hub import hf_hub_download, list_repo_files
    repo, cfg = src["repo"], src.get("config")
    out = {}
    try:
        files = list_repo_files(repo, repo_type="dataset", revision="refs/convert/parquet"); rev = "refs/convert/parquet"
    except Exception:
        files = list_repo_files(repo, repo_type="dataset"); rev = None
    for ours, theirs in src["splits"].items():
        cands = [f for f in files if f.endswith(".parquet") and (cfg is None or f.startswith(cfg + "/") or f"/{cfg}/" in f or cfg in f) and (f"/{theirs}/" in f or f"-{theirs}" in f or f"{theirs}." in f)]
        if not cands:
            # datasets 経由（parquet 変換のないリポ）
            from datasets import load_dataset
            ds = load_dataset(repo, cfg, split=theirs) if cfg else load_dataset(repo, split=theirs)
            out[ours] = [dict(r) for r in ds]
            continue
        dfs = [pd.read_parquet(hf_hub_download(repo, f, repo_type="dataset", revision=rev)) for f in sorted(cands)]
        out[ours] = pd.concat(dfs).to_dict("records")
    return out


def load_fast_decisions(src: dict) -> dict[str, list[dict]]:
    """fastino/fast-decisions: input + classifications[{task, true_label, labels, multi_label}] → 行に展開。"""
    import pandas as pd
    from huggingface_hub import hf_hub_download
    rows = []
    for cfg in src["configs"]:
        df = pd.read_parquet(hf_hub_download(src["repo"], f"{cfg}/train/0000.parquet", repo_type="dataset", revision="refs/convert/parquet"))
        for _, r in df.iterrows():
            cls = r["output"]["classifications"] if isinstance(r["output"], dict) else ast.literal_eval(str(r["output"]))["classifications"]
            tasks = []
            for c in cls:
                if c.get("multi_label"):
                    continue
                labels = [str(x) for x in c["labels"]]; gold = str(list(c["true_label"])[0])
                tasks.append({"task": str(c["task"]), "labels": labels, "gold": gold})
            if tasks:
                rows.append({"domain": cfg, "input": str(r["input"]), "_tasks": tasks})
    random.Random(0).shuffle(rows); sp = src["splits"]; n = len(rows); a = int(n * sp["train"]); b = a + int(n * sp["val"])
    return {"train": rows[:a], "val": rows[a:b], "test": rows[b:]}


def load_csv(src: dict) -> dict[str, list[dict]]:
    import pandas as pd
    df = pd.read_csv(src["path"], sep="\t" if src["path"].endswith(".tsv") else ",")
    rows = df.to_dict("records"); random.Random(0).shuffle(rows)
    sp = src["splits"]; n = len(rows); a = int(n * sp["train"]); b = a + int(n * sp["val"])
    return {"train": rows[:a], "val": rows[a:b], "test": rows[b:]}


def get(row: dict, key: str):
    """'writer.joy' のようなドット記法。文字列化された dict も解釈する。"""
    cur = row
    for k in key.split("."):
        if isinstance(cur, str) and cur.startswith("{"):
            cur = ast.literal_eval(cur)
        cur = cur[k] if isinstance(cur, dict) else cur[k]
    return cur


def to_int(v):
    try:
        return int(float(v))
    except Exception:
        return v


# ---------------------------------------------------------------- conversion
def _norm(v):
    """YAML の yes/no は bool になるので文字列へ戻す。"""
    if v is True: return "yes"
    if v is False: return "no"
    return str(v)


def gold_of(q: dict, row: dict):
    g = get(row, q["gold"])
    if q.get("gold_round"):
        g = int(round(float(g)))
        return q["levels"][max(0, min(len(q["levels"]) - 1, g))]
    if q.get("gold_is_index"):
        return q["_options"][int(g)]
    if "gold_threshold" in q:
        return "yes" if float(g) >= q["gold_threshold"] else "no"
    if "label_map" in q:
        m = {to_int(k) if not isinstance(k, str) or k.lstrip("-").isdigit() else k: v for k, v in q["label_map"].items()}
        return _norm(m[to_int(g)])
    return _norm(g)


def make_question(q: dict, row: dict, instr: str) -> dict:
    out = {"type": q["type"], "id": q["id"], "instructions": instr}
    if q["type"] == "choice":
        out["options"] = [str(row[c]) for c in q["options_from"]] if "options_from" in q else list(q["options"])
    elif q["type"] == "score":
        out["levels"] = list(q["levels"])
    return out


def build_family(name: str, fam: dict, rng: random.Random):
    src = fam["source"]
    splits = load_hf(src) if src["kind"] == "hf" else load_fast_decisions(src) if src["kind"] == "fast_decisions" else load_csv(src)
    if src["kind"] == "fast_decisions":
        return build_fast_decisions(name, fam, splits, rng)
    if "val_test_split" in fam and fam["source"]["splits"].get("val") == fam["source"]["splits"].get("test"):
        rows = list(splits["val"]); rng.shuffle(rows); h = int(len(rows) * fam["val_test_split"])
        splits["val"], splits["test"] = rows[:h], rows[h:]
    limit = fam.get("limit", {})
    result = {"train": [], "val": [], "test": [], "ood": []}
    for sp in ["train", "val", "test"]:
        rows = list(splits[sp]); rng.shuffle(rows); rows = rows[: limit.get(sp, len(rows))]
        for i, row in enumerate(rows):
            state = fam["state"].format(**{k: str(v) for k, v in row.items()})
            mx = fam.get("state_max_chars")
            if mx and len(state) > mx:
                state = state[:mx]
            qs, gold, qs_ood, gold_ood = [], {}, [], {}
            for q in fam["questions"]:
                qq = dict(q)
                if "options_from" in q:
                    qq["_options"] = [str(row[c]) for c in q["options_from"]]
                g = gold_of(qq, row)
                instrs = q["instructions"] if isinstance(q["instructions"], list) else [q["instructions"]]
                qs.append(make_question(q, row, rng.choice(instrs) if sp == "train" else instrs[0])); gold[q["id"]] = g
                if sp == "test" and q.get("ood_instructions"):
                    oi = q["ood_instructions"] if isinstance(q["ood_instructions"], list) else [q["ood_instructions"]]
                    qs_ood.append(make_question(q, row, oi[0]))
                    gold_ood[q["id"]] = ({"yes": "no", "no": "yes"}[g] if q.get("ood_negated") else g)
            item = {"id": f"{name}-{sp}-{i}", "family": name, "split": "in_domain", "state": state, "questions": qs, "gold": gold}
            if fam.get("context_from"):
                item["context"] = [str(row[c]) for c in fam["context_from"]]
            if fam.get("context_from_evidence"):
                ev = row[fam["context_from_evidence"]]
                item["context"] = [str(e[2]) for e in list(ev)[:3]] if len(ev) and len(ev[0]) >= 3 else []
            result[sp].append(item)
            if qs_ood:
                result["ood"].append({**item, "id": f"{name}-ood-{i}", "family": name + "_ood", "split": "ood", "questions": qs_ood, "gold": gold_ood})
    return result


def build_fast_decisions(name, fam, splits, rng):
    """1 入力 × 複数タスク → そのまま Jev 形（Choice、yes/no は Noul）。family は domain ごと。"""
    limit = fam.get("limit", {}); result = {"train": [], "val": [], "test": [], "ood": []}
    for sp in ["train", "val", "test"]:
        rows = list(splits[sp]); rng.shuffle(rows); rows = rows[: limit.get(sp, len(rows))]
        for i, row in enumerate(rows):
            qs, gold = [], {}
            for t in row["_tasks"]:
                if sorted(t["labels"]) == ["no", "yes"]:
                    qs.append({"type": "noul", "id": t["task"], "instructions": f"{t['task'].replace('_', ' ')}?"}); gold[t["task"]] = t["gold"]
                else:
                    qs.append({"type": "choice", "id": t["task"], "instructions": f"Select the {t['task'].replace('_', ' ')}.", "options": t["labels"]}); gold[t["task"]] = t["gold"]
            result[sp].append({"id": f"{name}-{sp}-{i}", "family": f"{name}:{row['domain']}", "split": "in_domain", "state": row["input"][:1200], "questions": qs, "gold": gold})
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="*")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--device-subset", type=int, default=20, help="docs/bench 用に family ごと N state（test から）")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(CFG, encoding="utf-8"))
    rng = random.Random(a.seed)
    os.makedirs(a.out, exist_ok=True)
    allres = {"train": [], "val": [], "test": [], "ood": []}
    manifest = {}
    for name, fam in cfg["families"].items():
        if a.families and name not in a.families:
            continue
        print(f"building {name} …", end=" ", flush=True)
        r = build_family(name, fam, rng)
        for k in allres:
            allres[k] += r[k]
        manifest[name] = {"desc": fam.get("desc"), "license": fam.get("license"), **{k: len(v) for k, v in r.items()}}
        print({k: len(v) for k, v in r.items()})
    for k, items in allres.items():
        rng.shuffle(items)
        with open(os.path.join(a.out, f"{k}.jsonl"), "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
    json.dump(manifest, open(os.path.join(a.out, "manifest.json"), "w"), ensure_ascii=False, indent=2)
    # 端末評価用サブセット（test + ood から family ごと N）
    sub = []
    for name in manifest:
        sub += [it for it in allres["test"] if it["family"] == name][: a.device_subset]
        sub += [it for it in allres["ood"] if it["family"] == name + "_ood"][: a.device_subset]
    dd = os.path.join(ROOT, "docs", "bench"); os.makedirs(dd, exist_ok=True)
    with open(os.path.join(dd, "jevbench_ja_small.jsonl"), "w", encoding="utf-8") as f:
        for it in sub:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print("saved", a.out, {k: len(v) for k, v in allres.items()}, "| device subset", len(sub))


if __name__ == "__main__":
    main()
