# JEV Lab

Jev 級（型付き判断：typed-decision）モデルを **日本語で構築・比較・解説** する研究環境。

- **PC（VS Code）**：学習・アブレーション・大型モデルとの比較・ONNX 化
- **GitHub Pages（スマホ）**：解説・ベンチ結果・JEV-JA の端末内推論（機内モードでも動く）

```
state + [Choice | Score | Noul]* --(1 forward)--> 質問ごとの較正済み確率分布
```

## 検証済みのこと

| 項目 | 結果 |
|---|---|
| JEV-JA（modernbert-ja-30m + 判断ヘッド）が PyTorch で動く | ✅ |
| 学習ループ（CE+Brier、拡張、温度推定、保存・再読込） | ✅ |
| ONNX 書き出し：fp32 は PyTorch と最大差 1e-4、int8 は隠れ状態の cos 類似 0.993、合計 45 MB | ✅ |
| **ブラウザ推論が Python と 3 桁一致**（トークナイズ・byte fallback・スパン平均・ヘッド・softmax） | ✅ headless Chromium で確認 |
| ブラウザでの推論時間：166 トークン・3 質問で 540〜620 ms（WASM、サーバー側 CPU） | ✅ |
| Pages 用 PWA（Service Worker でモデルをキャッシュ、`navigator.storage.persist`） | ✅ 構成済 |

---

## あなたがやる操作（順番どおり）

### 0. 一度だけ：環境を作る（Windows 11 + RTX 3060）

```powershell
# リポを作って中身を入れる（zip を展開したフォルダで）
git init; git add -A; git commit -m "JEV Lab v0.2"
gh repo create 224096-cmd/jev-lab --public --source=. --push     # gh が無ければ GitHub の画面で作成→ git remote add → push

# Python 環境
py -3.11 -m venv .venv; .\.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu124   # GPU 版
pip install -e ".[onnx,notebook]"
```

**意味**：`.venv` はこのプロジェクト専用の Python。`-e` は「このフォルダのコードをそのまま import できるようにする」。以後 VS Code で `jev_lab/` を編集すれば即反映される。

### 0'. 一度だけ：GitHub Pages を有効化

GitHub のリポ画面 → **Settings → Pages → Build and deployment → Source を「GitHub Actions」** にする。
これで `docs/` を変更して push するたびに `.github/workflows/pages.yml` が動き、
`https://224096-cmd.github.io/jev-lab/` が更新される（数十秒〜1分）。

### 1. 日常：試す（Notebook）

VS Code で `notebooks/01_playground.ipynb` を開き、上から実行。
state・questions・モデル名をセルで書き換えて再実行するだけ。ブラウザもサーバーも不要。

### 2. 学習する

```powershell
python -m jev_lab.train.train_jev_ja --train data/train.jsonl --val data/val.jsonl `
   --backbone sbintuitions/modernbert-ja-30m --out checkpoints/jev_ja_30m --epochs 2 --augment 0.7
```

**意味**：`checkpoints/jev_ja_30m/` に学習済みヘッド＋設定＋`report.json`（損失曲線・精度・ECE・温度）が出る。
`config/models.yaml` の `head_path` がここを指しているので、以後のすべての操作で自動的に学習済みが使われる。

よく使う変更：`--pool marker`（読み出し方式の比較）／`--brier 0`（CE のみ）／`--freeze-backbone`（ヘッドのみ・数十秒）／`--seed 1`

### 3. 比較ベンチを回して公開

```powershell
python -m jev_lab.bench.run --data bench/data/sample_ja.jsonl --models jev_ja_30m nli_mdeberta --publish
python -m jev_lab.bench.run --data bench/data/sample_ja.jsonl --models jev_ja_30m --no-context --publish   # RAG 無しのアブレーション
```

**意味**：`reports/` に結果 JSON、`--publish` で `docs/results/` にもコピー。push すると Pages の「解説・結果」ページの表に出る。
`llm_qwen3_*` や `gliner2_*` を入れるときは先に `pip install gliner2` 等が必要。

### 4. スマホで動かす（ONNX 化 → push）

```powershell
python -m jev_lab.export.export_onnx --model jev_ja_30m
git add -A; git commit -m "model update"; git push
```

**意味**：`docs/models/jev_ja_30m/` にエンコーダ（int8）・ヘッド・トークナイザ・設定が出て、`docs/models/index.json` に登録される。
push 後、スマホで `https://224096-cmd.github.io/jev-lab/play.html` を開く → 「モデルを読み込む」（初回 45 MB、以後キャッシュ）→ 判断。
ホーム画面に追加すれば PWA として機内モードでも動く。画面に ms とトークン数が出るので、これが卒論の端末性能の数値になる。

### 5. モデルを増やす

- **JEV-JA の別バックボーン**：`config/models.yaml` にエントリを足す（`backbone:` を HF id に）→ 2〜4 を繰り返す
- **別方式（GLiNER2 / NLI / LLM）**：`models.yaml` の `hf_id` を変えるだけ。ブラウザには載らず PC の比較専用

### 困ったとき

| 症状 | 対処 |
|---|---|
| ブラウザで「ロード失敗」 | PC で `python -m http.server 8765 -d docs` → `http://localhost:8765/play.html` を開き、F12 のコンソールを見る。壊れていても `index.html`（解説・結果）は表示される |
| ONNX 化で落ちる | `--no-int8` で fp32 のまま出す（155 MB、GitHub Pages の 100 MB/ファイル制限を超えるので公開はできない。原因切り分け用） |
| 推論が遅い | 30m で 0.5 秒前後が目安。70m は 2 倍程度。state を短くするか質問数を減らす |
| GitHub に 100 MB 超のファイルを push できない | int8 の 30m/70m は収まる。DeBERTa-large 級は Pages に載せない（PC 専用） |

---

## リポ構成

```
jev_lab/
  schema.py            Choice/Score/Noul → 分布 → 答え（1±2% 正規化規則）
  adapters/            共通 IF decide(state, questions)：jev_ja / open_jev / gliner2 / nli_zeroshot / llm_verbalized
  train/train_jev_ja.py  学習（CE+Brier、拡張、温度）
  bench/               run.py（--publish で docs/results へ）、metrics.py（精度・Brier・ECE・p50/p95・invalid）
  export/export_onnx.py  JEV-JA → docs/models/<name>/（encoder int8 + head + tokenizer + config）
  rag/                 ruri-v3-30m + 文字 bigram BM25 の RRF、Int8 書き出し
config/models.yaml     モデル一覧（ここを編集して切替）
bench/data/sample_ja.jsonl   データ形式の見本（災害トリアージ・SNS 検証・OOD）
docs/                  GitHub Pages：index.html（解説・結果）、play.html + jev.js（端末内推論）、sw.js、models/、results/
notebooks/01_playground.ipynb
.github/workflows/pages.yml
```

## データ形式（JSONL、1 行 = 1 state）

```json
{"id":"d001","family":"disaster_triage","split":"in_domain",
 "state":"...","context":["根拠1","根拠2"],
 "questions":[{"type":"choice","id":"hazard","instructions":"...","options":["..."]},
              {"type":"score","id":"urgency","instructions":"...","levels":["低い","中程度","高い"]},
              {"type":"noul","id":"vertical","instructions":"..."}],
 "gold":{"hazard":"...","urgency":"高い","vertical":"yes"}}
```

`split: ood` は「同じ state に未見の instructions・未見の option 列」（slot 暗記の検査）。

## 次にやること（P1 完成定義に向けて）

- [ ] JevBench-JA 構築：livedoor（Choice）・WRIME（Score）・JNLI/JSICK/JSQuAD（Noul）から gold を生成する `data.py`
- [ ] 災害ドメイン question セット 500 件（Qwen3-8B 合成 → 人手検証）
- [ ] 30m/70m の本学習と int8 での精度低下の測定（fp32 との argmax 一致率）
- [ ] play.html に RAG（ruri-v3-30m ONNX + Int8 索引）を載せる
- [ ] open-jev / gliner2 / llm アダプタの実機確認

## 参考

- Kotoba Labs typed-decisions（Jev 形の再現、span-pool 知見）: https://github.com/kotoba-lang/typed-decisions
- JevBench（5 軸・native/verbalized の区別）: https://benchmarkheaven.com/jev-models/v1
- GLiNER2.5-Decide: https://huggingface.co/fastino/GLiNER2.5-Decide
