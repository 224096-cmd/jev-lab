# JEV Lab — 小さな判断モデル（Jev 級）による公開情報の収集・整理・検証

**主軸**：調べたいことを入れると、公開 API から情報を集め、30〜70M パラメータの日本語判断モデル **JEV-JA** が「関連・種類・緊急度・具体性・拡散依頼・根拠との整合」を確率つきで仕分け、説明できる信頼性スコアと証跡を残す。すべてスマホのブラウザ内（オフライン可）で動く。
**研究**：Jev 級モデル（typed-decision model）の仕組み・学習・較正・他方式との比較を、同じ環境で実験・記録する。災害時の情報整理はその応用の一つ。

```
調べたいこと → 収集（気象庁・Wikipedia・Wikidata・OSM・Bluesky・GDELT）
            → JEV（1 forward で 6 つの型付き質問）→ 信頼性スコア（重み編集可）
            → 類似投稿のクラスタ・地理照合（地名→座標→標高）→ 証跡 JSON / CSV
            → 人手ラベル → 自分の学習データ（osint_ja）→ 再学習 → 比較
```

- 公開ページ：`https://224096-cmd.github.io/jev-lab/`（概要・結果 / 情報収集・判断 / 仕組み）
- コード・学習・比較：このリポ（VS Code）

---

## 1. 構成

| 場所 | 役割 |
|---|---|
| `docs/`（GitHub Pages） | **`play.html`** 情報収集（メイン）・判断 Playground・検証・仕組み可視化・バッチ評価・実験ノート・テンプレ・設定。**`theory.html`** 数式つき解説。**`index.html`** 概要と PC 結果のリーダーボード。`jev.js` 推論エンジン、`osint.js` 収集・地理照合・信頼性スコア、`app.js` UI |
| `jev_lab/adapters/` | 共通 IF `decide(state, questions)`。`jev_ja`（本体・head/cos）、`open_jev`、`gliner2`、`nli_zeroshot`、`llm_verbalized` |
| `jev_lab/data/` | `build.py`（`config/datasets.yaml` → JSONL）、`import_csv.py`（自分の CSV → JSONL） |
| `jev_lab/train/` | `train_jev_ja.py`（CE+Brier、拡張、温度）、`convert_gliner2.py`（既存 Jev モデル改良用） |
| `jev_lab/bench/` | `run.py`（5 軸評価、`--publish` で docs/results へ）、`metrics.py` |
| `jev_lab/export/` | `export_onnx.py`（→ `docs/models/<name>/`、int8） |
| `jev_lab/osint/` | `collect.py`（RSS 含む PC 版コレクタ）、`judge.py`（複数モデルで一括判断・κ 一致度） |
| `config/` | `models.yaml`（モデル一覧）、`datasets.yaml`（データ定義） |
| `data/jevbench_ja/` | 生成物（train 11,280 / val 1,430 / test 2,940 / ood 2,700 state）。git には入れない |

## 2. データ（JevBench-JA v0、10 family）

| family | 元 | 型 | 情報収集での意味 |
|---|---|---|---|
| livedoor_topic | livedoor ニュース | Choice 9 | 話題分類 |
| jnli / claim_check_ja | JGLUE JNLI | Choice 3 + Noul | 根拠×主張の整合（事実検証） |
| jsts | JGLUE JSTS | Score 6 | 類似度（重複・コピペ） |
| jcqa | JGLUE JCommonsenseQA | Choice 5 | 常識推論 |
| wrime_sentiment | WRIME v2 | Score 5 + Noul | SNS 感情 |
| jcola | JCoLA | Noul | 文の自然さ |
| fever_en | FEVER gold evidence | Choice 3 + Noul | 事実検証（英語。open-jev / GLiNER2 比較用） |
| tweet_sentiment_en | TweetEval | Score 3 | SNS 感情（英語） |
| fast_decisions_en:* | fastino/fast-decisions 8 ドメイン | Choice/Noul 複数 | GLiNER2.5-Decide の公式ベンチと同条件 |
| osint_ja | Play で収集＋人手ラベル | Choice + Score | 本ラボ独自 |

各 family に in-domain と **OOD**（未見の質問文・否定形で gold 反転）がある。ライセンスは `config/datasets.yaml`。

**自分のデータを足す**：
- CSV から：`python -m jev_lab.data.import_csv --file x.csv --family my --text answer --q "id:type:質問文:選択肢|区切り:gold列"`
- 公開データから：`config/datasets.yaml` に family を 1 つ書き足して `python -m jev_lab.data.build --families my`
- Play の情報収集で人手ラベル → 「学習データ JSONL」ボタン → `data/osint_ja/` に置く

## 3. 日常の操作（変更の反映手順）

### A. コードや設定を変えた → 公開に反映する（毎回これ）
1. VS Code で編集して保存
2. ターミナル（`.venv` 有効）で
   ```powershell
   git add -A
   git commit -m "変更内容を一言"
   git push
   ```
3. GitHub → Actions で「pages」が ✅ になるのを待つ（約 1 分）
4. スマホ／PC で `https://224096-cmd.github.io/jev-lab/play.html` を**開き直す**（古い版が出たらもう一度リロード。Service Worker が更新を取り込む）
5. 変更が見えない場合：Play →「保存・設定」→ 端末情報で Service Worker が「制御中」か確認 → ブラウザのサイトデータを消さずに、タブを閉じて開き直す

### B. モデルを学習し直した → スマホに載せる
1. 学習：`python -m jev_lab.train.train_jev_ja --train data/jevbench_ja/train.jsonl --val data/jevbench_ja/val.jsonl --out checkpoints/jev_ja_30m --epochs 2 --augment 0.7`
2. 変換：`python -m jev_lab.export.export_onnx --model jev_ja_30m`（`docs/models/jev_ja_30m/` が更新される）
3. 評価して公開用結果を作る：`python -m jev_lab.bench.run --data data/jevbench_ja/test.jsonl --models jev_ja_30m jev_ja_30m_cos --publish`（`docs/results/` に追加）
4. A の手順で push
5. スマホの Play →「保存・設定」→ 旧モデルを**削除** → 「モデルを読み込む」（新しい重みが IndexedDB に入る）

### C. 新しいモデル（別バックボーン・既存 Jev モデル）を足す
1. `config/models.yaml` にエントリを追加（`backbone:` を HF id に。既存 Jev モデルなら `adapter:` を変える）
2. B-1〜3（JEV-JA 系なら学習 → 変換 → 評価。PC 専用モデルなら評価のみ）
3. `docs/models/index.json` は変換時に自動更新される → push

### D. データを足した
1. `config/datasets.yaml` に family を追加、または `import_csv.py`
2. `python -m jev_lab.data.build`（`docs/bench/jevbench_ja_small.jsonl` も更新される）
3. `docs/bench/index.json` を更新（build が自動で書く）→ push → Play「バッチ評価」のプルダウンに出る

### E. 実験結果を卒論用に集める
- 端末：Play「実験ノート」→「全件 CSV」（モデル・手法・pool・T・データ・端末・精度・Brier・ECE・ms）
- PC：`reports/*.json` と `docs/results/*.json`（`--publish`）。`index.html` のリーダーボードに統合表示
- 情報収集：「証跡 JSON」「CSV」「実験ノートに保存」

## 4. セットアップ（初回のみ）

```powershell
py -3.12 -m venv .venv; .\.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[onnx,notebook]"      # 大型モデル比較は追加で: pip install gliner2 datasets
```
GitHub → Settings → Pages → Source を **GitHub Actions** にしておく。

## 5. 検証済みのこと

- ブラウザ推論が Python と 3 桁一致（トークナイザ byte fallback を JS で再現）
- スマホ実機：1 判断（166 トークン・3 質問）**167 ms**（WASM）。オフライン再読込 → 自動ロード → 判断まで確認
- 公開 API：気象庁 bosai JSON・Nominatim・GSI 標高タイル・Wikipedia/Wikidata・HN は GitHub Pages から CORS で取得可。Bluesky 公開検索は実ブラウザから可。GDELT は 5 s/req
- Play 全タブ（情報収集→整理→信頼性、質問ビルダー、T/pool/head-cos 切替、検証 6 種、可視化、バッチ、ノート）を headless Chromium で動作確認

## 6. 研究ロードマップ（P1 → P2）

| 段階 | 内容 | 成果物 |
|---|---|---|
| P1-a（済） | 基盤：スキーマ・アダプタ・データ 10 family・学習・ONNX・Pages・OSINT フロー | このリポ |
| P1-b | JEV-JA 30m/70m を GPU で本学習（2〜3 ep、augment 0.7）。in-domain / OOD / 較正 / 端末 ms | 結果表、公開モデル |
| P1-c | 比較：cos ゼロショット、NLI、GLiNER2.5-multi、open-jev（英語 family）、Qwen3 verbalized。fast_decisions で GLiNER2 と同条件 | リーダーボード |
| P1-d | 既存 Jev 級モデルの改良：mDeBERTa-base に JEV ヘッド、GLiNER2.5-multi の追加学習 | 5 軸比較 |
| P1-e | 情報収集の評価：Play で 5〜10 テーマを収集 → 人手ラベル 300〜500 件（osint_ja）→ 学習前後の信頼性判定精度、モデル間 κ、新着検出 | 独自データセット |
| P1-f | アブレーション：span/marker、λ、拡張、int8、温度 | 卒論の実験章 |
| P2 | 応用：事前同期したハザード・避難所データと組み合わせた災害時オフライン運用（同じ JEV・同じ地理照合） | 応用章 |

## 7. 参考
- Kotoba Labs typed-decisions（Jev 形の再現・span-pool 知見）: https://github.com/kotoba-lang/typed-decisions
- JevBench（5 軸・native/verbalized）: https://benchmarkheaven.com/jev-models/v1
- GLiNER2.5-Decide / fast-decisions: https://huggingface.co/fastino/GLiNER2.5-Decide
