# JEV Lab — 小さな判断モデル（Jev 級）による公開情報の収集・整理・検証（v1.1）

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
4. 既存 Jev をブラウザに載せる変換：cross-encoder 形は `python -m jev_lab.export.export_crossenc --hf <id> --name <name>`、Laya 形は `python -m jev_lab.export.export_laya --hf convaiinnovations/laya-multilingual --name laya_multi_322m`（空き容量 2 GB・空きメモリ 6 GB 以上。fp32 → 埋め込み int8 化 → 量子化 → 90 MB 分割）

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

## 4'. Play の画面構成（v0.6）

上部のモデルバー（読み込む → 説明行に言語・サイズ・端末適性）と、初回だけ出る「①読み込む ②キーワード ③JEV が仕分ける」の 3 ステップ案内。

| 群 | タブ | 何ができるか |
|---|---|---|
| 使う | **情報収集** | 大きな入力欄にキーワード（カンマで複数、Enter）→ 情報源チップ → 収集 → 自動で JEV 整理 → **「JEV の要約」**（種類の内訳・信頼性上位・緊急・根拠と矛盾の疑い・要確認）→ カード表示（信頼性の大きな数字、種類・緊急・具体・拡散依頼・整合のフラグ）→ 並び替え（信頼性／新しい順／出典／緊急度）・絞り込み（新着／公的・報道／要確認）→ ＋根拠 → 証跡 / CSV / MD / ノート / 人手ラベル。**🔔 保存した検索の新着を確認**（履歴をまとめて再収集し新着だけ表示＝定点観測） |
| 使う | **文章を判断** | 1 件を詳しく。テンプレ（既定は空）、質問ビルダー（質問文＋選択肢、id は自動）、「よく使う」質問の 1 クリック追加、**答えカード**（大きな判定・確信度の色分け・分布は折りたたみ）、不確かな問いへの改善ヒント。温度・読み出し・手法は詳細設定に |
| 使う | **一括判断** | 複数テキスト（貼り付け／.txt／.csv）× 同じ質問 → 集計（Choice の内訳・Noul のはい件数）＋表＋CSV。問い合わせ振り分け・アンケート集計向け |
| 使う | **JEV チャット** | 文章を送ると質問セットで判定。**`? 返信が必要か`** で直前の文章に はい/いいえ、**`? 担当は \| 請求 \| 配送`** で選択式（読み物アシスタントとしての使い方）。`/osint` で収集→要約→要確認 |
| 使う | 調査ツール | dork ビルダー・ドメインの素性・EXIF/GPS・逆画像検索。結果は根拠へ |
| 研究 | 検証 | 6 種の検証それぞれに「何を測るか・良い結果の目安」と**自動の解釈文**（根拠を参照しているか、位置バイアス、否定形の反転、マトリクスの一致問い数） |
| 研究 | バッチ評価 | 精度に **95% 信頼区間（ブートストラップ）**、**多数決／ランダムのベースライン**、残り時間、較正図 PNG、自動解釈（ベースライン超えか、ECE の良否） |
| 研究 | 改良（端末内学習） | ヘッド再学習、学習前後を val で比較、保存 |
| 研究 | 実験ノート | **「卒論用の表（Markdown）」**（モデル×設定×データ×精度(CI)×多数決×Brier×ECE×ms×端末）をクリップボードへ。CSV / JSON / 差分 |
| 学ぶ・設定 | 仕組み・可視化 | 入力列・cos・logit→softmax・PCA（PNG）・時間 |
| 学ぶ・設定 | モデル | 使えるモデル表、HF 検索（最終更新順）、JSON 追加（open-jev-onnx / crossenc / laya） |
| 学ぶ・設定 | 保存・設定 | 保存モデル、永続化、更新、ダーク／文字サイズ／案内表示、端末情報 |

### v0.6 で直した「システムとしておかしい／分かりにくい」点（16 件）

| # | 問題 | 対応 |
|---|---|---|
| 1 | 初めて開いても何をすればよいか分からない | 3 ステップ案内、モデル説明行、ボタン無効時の理由表示（「先にモデルを読み込む」） |
| 2 | 情報収集の入力欄が設定だらけで、肝心のキーワード欄が小さい | 大きな検索欄＋チップ＋例、設定は「詳細設定」に折りたたみ |
| 3 | 収集結果が 13 列の表で読めない | カード表示（信頼性の大きな数字・フラグ・内訳・全文）、並び替え・絞り込み |
| 4 | JEV の判定が各行に散らばり「結局どうなのか」が無い | **JEV の要約**（内訳・上位・緊急・矛盾・要確認）と「要確認のみ」フィルタ |
| 5 | 判定結果がバー表の羅列で、質問 id しか出ない | 答えカード（質問文・大きな判定・確信度の色・分布は折りたたみ・改善ヒント） |
| 6 | 質問ビルダーで id を手で付けさせられる | 質問文から自動生成、「よく使う」質問を 1 クリック追加、型の意味を併記 |
| 7 | テンプレタブと Playground が二重 | 「文章を判断」（1 件）と「一括判断」（複数）に整理。テンプレは両方のプルダウンから |
| 8 | チャットで JEV の使いどころが分からない | `?` で直前の文章に問い直す（はい/いいえ・選択式）、返答を「質問文 → 判定」の文に |
| 9 | 検証タブの結果が数表だけで意味が分からない | 各項目に測定の意味・目安、自動の解釈文 |
| 10 | バッチ評価の精度が「高いのか低いのか」判断できない（研究として不十分） | 95% CI、多数決／ランダムのベースライン、自動解釈。ノート・CSV・卒論表にも記録 |
| 11 | 卒論の表を作るのに手作業が要る | 「卒論用の表（Markdown）」ボタン |
| 12 | 地名＋出来事語（「津市 大雨」）で Nominatim が中国の地名を返す | 出来事語を除いた地名で検索、日本語なら国内限定 |
| 13 | 弱いモデルだと全件が「根拠と矛盾」になる | 矛盾判定を p(整合) < 0.35 に限定（確信をもって否定したときだけ） |
| 14 | 定点観測（同じテーマを毎日見る）ができない | 🔔 保存した検索の新着確認 |
| 15 | ONNX Runtime の Worker 実行で JEV-JA の 2 回目のヘッド推論が落ちる（転送済みバッファ再利用） | 入力を複製して渡す（バグ修正） |
| 16 | 既存 Jev の比較対象が argos と open-jev だけ | **Laya multilingual（HF で最も使われている小型 Jev 系、Apache-2.0）をブラウザ・PC 両方に追加**（下 3-A） |


## 4-4. v0.7：調査（対象指定）・OSINT Framework・ツール箱・「JEV だから効率が上がる」設計

### 画面（使う）
| タブ | 何をするか |
|---|---|
| 情報収集（話題） | キーワード → 公開 API（気象庁・Wikipedia/Wikidata・Nominatim・Bluesky・**Mastodon**・GDELT・HN・**Stack Exchange・GitHub・Crossref・arXiv・Semantic Scholar**）→ 届いた分から JEV が即判定 → 要約・仕分け（自動採用／人手確認／除外のしきい値）→ 証跡 |
| **調査（対象を指定）** | ドメイン・URL・IP・メール・@ユーザー名・電話番号・座標・地名・会社名・BTC/ETH アドレス・ハッシュ・CVE・便名・船舶を**自動判定**（複数可）。種類ごとに鍵不要の公開 API を自動で叩き（DNS/RDAP/SPF・DMARC/証明書ログ/ホスト名/Wayback 履歴＋URL 一覧/AlienVault OTX/ipinfo/逆引き/Gravatar/GitHub/mempool.space/Blockscout/NVD/逆ジオ/標高/OSM ノート）、残りは各サービスの検索ページをリンクで開く。JEV が根拠行を読んで「正規か／なりすましの疑い／注意点／追加確認」を判定。関係図（Maltego 風）。OSINT Framework 一覧（自動／端末内／リンク／実装しない理由） |
| ツール箱 | **dork ビルダー**（演算子 19 種の一覧、用途別ライブラリ 20 式、OR・数値範囲・intext、各エンジンで開く）、画像 **EXIF/GPS + ELA（改ざん痕）**＋逆画像検索、**文書メタデータ**（PDF /Info・XMP、docx/xlsx/pptx core.xml）、**変換**（Base64/Hex/URL/HTML/ROT13/UNIX 時刻/JWT/Punycode）＋文字列の種類判定＋SHA、**ファイルのハッシュ → VirusTotal / MalwareBazaar 検索**（送信しない）、**太陽の方位・高度**（影と撮影時刻の整合）、電話番号の書式判定、**OpSec**（ブラウザの露出情報、パスワード漏えい k-匿名性確認）、学ぶ・証拠保全リンク |

### OSINT Framework の各カテゴリの扱い（合法・受動的なものだけ）
| カテゴリ | 扱い |
|---|---|
| ユーザー名 | GitHub 公開 API のみ自動、他はリンク。自分のアカウント名の露出確認か同意のある調査に限る（Sherlock 型の横断自動照会はしない） |
| メール | SPF/DMARC/MX・使い捨て判定・Gravatar を自動。HIBP は自分のアドレスでサイト上で |
| ドメイン / IP | 全自動（上記）＋ Shodan/Censys/VirusTotal/Safe Browsing はリンク |
| 画像・動画・文書 | EXIF・ELA・メタデータは端末内。逆画像検索・InVID はリンク |
| SNS / IM | Bluesky・Mastodon は自動、X/Reddit/YouTube/TikTok/Telegram 公開チャンネル/Discord はリンク |
| 電話番号 | 書式・種別・市外局番を端末内で。持ち主の特定はしない |
| 公的記録 / 企業記録 | 法人番号・gBizINFO・官報・e-Gov・判例・NDL・e-Stat・EDINET・OpenCorporates・Crunchbase・J-PlatPat へリンク、Wikidata は自動 |
| 交通 | FlightRadar24/Flightaware/MarineTraffic/VesselFinder へリンク（OpenSky API は CORS 不可） |
| 位置情報 | 逆ジオ・標高・OSM ノート・太陽位置を自動、Google/OSM/地理院/ハザードマップ/Sentinel へリンク |
| 検索エンジン / フォーラム / アーカイブ / 翻訳 / メタデータ / エンコード | 上記ツール箱と収集ソース。翻訳は MyMemory 公開 API で収集結果をその場で日本語化 |
| 暗号資産 | BTC（mempool.space）・ETH（Blockscout）を自動、Chainabuse へリンク |
| 悪意のあるファイル / エクスプロイト / 脅威インテリジェンス | ハッシュ検索（送信なし）、NVD 自動、Exploit-DB/ATT&CK/JVN/CISA KEV/URLhaus/ThreatFox はリンク。OTX は自動 |
| OpSec / 証拠保全 / トレーニング | 端末内チェック、証跡 JSON（本文ハッシュ付き）、Wayback 保存・Webrecorder・TraceLabs・Bellingcat へリンク |
| **人物検索・出会い系・ダークウェブ・テロリズム・クラシファイド広告** | **実装しない**（個人の特定・プロファイリング、違法コンテンツ接触、規約違反・閉鎖サービス。一覧に理由を明記） |

動画で扱われる手法の対応：Google dorks（ビルダー＋ライブラリ）、Shodan/Censys（リンク。鍵が要る）、theHarvester/Sherlock/Maltego（ホスト名・証明書ログの自動収集／ユーザー名はリンク集／関係図）、WHOIS・Wayback・EXIF・逆画像検索（自動／端末内）。

### 「JEV を使うと従来の AI（LLM）より何が効率的か」を設計に落とした点
| 観点 | LLM に文章で聞く場合 | JEV（本アプリ） | 実装 |
|---|---|---|---|
| 速度・逐次処理 | 全件そろえてプロンプト → 数秒〜数十秒 | 1 件 6 問 ≈ 100 ms（30m）。**届いた分から即判定** | 情報源ごとに取得→判定→描画 |
| 出力の型 | 自由文。件数集計・しきい値運用ができない | 型付き分布（Choice/Score/Noul）＋較正済み確率 | 仕分けしきい値（自動採用／人手確認／除外）、「読む件数 N → M」を表示 |
| 幻覚 | 事実を作る | **生成しない**。与えた根拠と本文の整合を確率で返すだけ | 「根拠と整合」「具体性」「拡散依頼」の判定、調査の根拠行判定 |
| プライバシー・OpSec | 収集内容を外部 API に送る | **端末内**で判定。オフライン可 | 判定件数と時間をパネルに表示、「外部 AI に送らない」を明記 |
| 再現性 | 温度・プロンプトで揺れる | 同じ入力→同じ分布。順序・否定形の検証タブで安定性を測れる | 検証タブ、実験ノート、95% CI |
| コスト | トークン課金 | 無料・端末の CPU | — |
| 改良 | プロンプト調整 | 人手ラベル数百件で端末内再学習（数十秒） | 改良タブ、人手ラベル → JSONL |

### 無料枠を超えない仕組み（`docs/quota.js`、v0.7.1）
すべての公開 API 呼び出しは `apiFetch(api, url)` を通す。(1) **予算**：API ごとに 1 日／1 時間／1 分の回数と 1 日の文字数（翻訳）を端末内で数え、無料枠より少なめの上限（hackertarget 40/日、MyMemory 4,500 文字/日、GitHub 8/分・50/時、NVD 6.5 秒間隔、Nominatim 1.1 秒間隔、arXiv 3.1 秒、GDELT 5 秒 など）に達したら**呼ばずにスキップ**して「上限のためスキップ」と表示。(2) **間隔**：利用規約の最短間隔を守って待つ。(3) **キャッシュ**：同じ URL の応答を TTL 付きで IndexedDB に保存し、キャッシュ命中は枠を消費しない（同じ対象の再調査は無料枠を減らさない）。使用状況は「保存・設定 → 公開 API の無料枠」に表示。鍵が要る API は一切使わない。

### 追加した情報源（情報収集）
Mastodon（タグの公開タイムライン）、Stack Exchange、GitHub リポジトリ、Crossref、arXiv、Semantic Scholar。プリセット「論文サーベイ」を追加（卒論の関連研究収集用）。証跡 JSON に本文の指紋（FNV-1a）を付けた。

## 4-5. v0.8：スマホ向け再設計・ホーム・チャット刷新・Jevlet（スマホで動く既存 Jev）

- **画面構成を再配置**：PC は上部タブ、スマホは下部バー（🏠 ホーム／🔎 収集／🕵️ 調査／⚖️ 判断／💬 チャット／⋯ その他）。「判断」は「1 件を詳しく」「複数を一括」の切替。その他＝ツール箱・検証・バッチ評価・改良・実験ノート・可視化・モデル・設定（タイルから選ぶ）。モデルバーは折りたたみ（読み込み後は 1 行に）。
- **ホーム**：JEV とは何か、**LLM との比較表（速さ・答えの形・捏造・データの行き先・再現性・改良）**、このセッションの実測（判定数・平均 ms・端末内）、「まず試す」タイル（収集／調査／SNS 検証／問い合わせ振り分け／チャット／精度測定）が 1 タップで実行、機能一覧。
- **スマホのバグ修正**：横スクロールの原因（長い選択肢を持つ `<select>` の固有幅、グリッド子要素の `min-width`）を修正し全タブで scrollWidth＝画面幅を確認。入力は 16px（iOS のズーム防止）、ボタン最小 40px、主ボタンは横いっぱい、二次操作は「⋯ 保存・書き出し」メニューに集約、結果へ自動スクロール。
- **チャット刷新**：メッセージアプリ風（固定の質問セットバー／メッセージ領域／クイック返信チップ／丸い送信ボタン、textarea 自動伸長）。返答は「質問 → 判定 ＋ 確信度」の行、`?` の問い直しは分布バー付き。スマホでは Enter で改行、送信はボタン。
- **Jevlet v6（NAME0x0/Jevlet、MIT）を追加**：bge-small 33.5M の既存 Jev。`export_jevlet.py` で int8 ONNX **35.6 MB**（スマホ可）。packing（[CLS][STATE] state｜[QUESTION] q [OPTION] o [END_OPTION]… [DECIDE]、block_bidir の 2 次元マスク、位置再開）を JS で再現し、Python と id・位置・分岐が完全一致、確率は int8 で最大 ±0.06。英語モデルなので日本語データでは 38.7% だが tweet_sentiment_en 67〜80%、**p50 40 ms**（PC CPU）。自作 jev_ja_30m（37M、日本語）と同規模の既存モデルとして比較の相手になる。PC 用アダプタ `jevlet_33m`（`pip install git+https://github.com/NAME0x0/Jevlet`）。
- **dork の高度な演算子**：allintitle / inanchor / AROUND(n) / related / imagesize / source / loc / ip / contains / lang を入力欄に追加。エンジン別に自動で書き換え（Bing は intext→inbody、期間指定を除去、Yandex は filetype→mime、DDG は非対応演算子を除去）、Google 専用・Bing 専用の演算子を注記。演算子一覧を 29 種に拡張。

| モデル（jevbench_ja_small、PC CPU） | 学習 | 精度 | Brier | ECE | p50 ms | 大きさ | 端末 |
|---|---|---|---|---|---|---|---|
| argos_ja_310m | 日本語 fine-tune | 54.6% | 0.607 | 0.167 | 985 | 316 MB | PC ブラウザ |
| laya_multi_322m | なし（多言語） | 50.8% | 0.743 | 0.292 | 274 | 323 MB | PC ブラウザ |
| jev_ja_30m（自作） | JevBench-JA | 40.8% | 0.680 | 0.130 | 27 | 47 MB | スマホ |
| **jevlet_33m**（既存・英語） | 英語コマンド | 38.7%（英語 family 67〜80%） | 0.852 | 0.250 | **40** | **36 MB** | スマホ |

## 4-6. v0.9：既存アプリとの連携・100 MB 以下のモデル 4 つ・検索プラン・ダッシュボード

### 既存アプリとの接続（4 方式の検討と採用）
| 方式 | 採用 | 実装 |
|---|---|---|
| ④ 共有メニュー／URL 起動 | **主軸（スマホ）** | PWA の `share_target`（Android の共有 → JEV Lab）、iOS ショートカット（共有シート → URL）、ブックマークレット、URL パラメータ `?text=&url=&tmpl=&run=1` / `?q=` / `?target=` |
| ① ブラウザ拡張 | **PC** | `extension/`（Manifest V3）：右クリック「JEV で判断／調査／収集」、サイドパネルに JEV Lab を表示、`Alt+J`。`chrome://extensions` に読み込むだけ（ストア不要） |
| ③ ノーコード連携（Zapier / Make / n8n） | 裏で流す | 既存アプリ → Google シート → 「シート取り込み」＋「監視」で JEV が自動仕分け |
| ② iframe 埋め込み | 不採用 | X・Gmail 等は iframe を拒否。逆に JEV Lab は埋め込み可（拡張のサイドパネル） |

直接 API：**Bluesky**（アプリパスワードでログイン → ホーム／通知を取り込み）、**Mastodon**（アクセストークン → ホーム／通知）、**Google スプレッドシート／CSV**（列を選んで一括判断）、**Discord / Slack Webhook・OS の共有**（結果を送る）。**監視**：一定間隔で取り込み直し、新しく「要確認」になった投稿だけをブラウザ通知。JEV の役割は、届いた文章をその場で・端末内で仕分け、要確認だけを残すこと（LLM のように送信・生成しない）。

### 100 MB 以下のモデル（HF を再調査し、int8／4 bit＋語彙の間引きで作成）
| モデル | 元 | 大きさ | 言語 | 端末 | jevbench_ja_small 精度 |
|---|---|---|---|---|---|
| jev_ja_30m（自作） | ModernBERT-Ja-30m | 47 MB | ja | スマホ・PC | 40.8% |
| jevlet_33m | NAME0x0/Jevlet（bge-small 33.5M） | 36 MB | en | スマホ・PC | 38.7%（英語 67〜80%） |
| openjev_base_60m | sshalimov04/open-jev-base（mmBERT-small 140M、Jev 1.13 蒸留） | 58 MB ＋ tokenizer 34 MB | en/ru（多言語 backbone） | スマホ・PC | 42.1%（PC 版） |
| laya_multi_q4 | convaiinnovations/laya-multilingual（322M） | 92 MB ＋ tokenizer 34 MB | 100+ 言語 | PC（強いスマホ可） | 4 bit（block 64・非対称）。argmax は概ね一致、確率は最大 ±0.3 ずれる例あり |
| argos / laya 322m（参考） | — | 316 / 323 MB | — | PC | 54.6 / 50.8% |

`onnx_utils.py`：語彙埋め込みを JevBench-JA のコーパスで使う id だけに間引き（256k → 約 3 万行、id→行 の対応表を Gather で引くので tokenizer は無変更）、埋め込み行スケール int8、MatMul 4 bit（MatMulNBits、onnxruntime-web の WASM で動作確認）、100 MB 超のテンソル分割。
**モデル読み込みの堅牢化**：3 回の再試行・Content-Length 検証・0 byte キャッシュの破棄・Worker 失敗時のメインスレッド再試行・原因別のメッセージ（メモリ不足→小さいモデルを案内）。

### 検索の補完・検索プラン・ダッシュボード
- 収集の入力欄：履歴・プリセット・Wikipedia 見出し候補を補完。キーワードから種類（災害／噂・詐欺／組織・サイト／技術・研究／一般）を推定し、**検索プラン**（推奨の情報源を自動選択＋ Google 検索式のチップ：公的発表／報道（直近）／SNS／PDF／ファクトチェック／初出）を表示
- 収集結果の**ダッシュボード**（SVG）：種類の内訳・信頼性の分布・出典別件数
- 調査の「次に調べる」：対象の種類ごとに dork（同名サイトの重複・被害注意喚起・サブドメイン…）と「この話題を収集」
- 太陽の位置ツールは削除（研究の趣旨外）

## 4-7. v1.0：サイト構成をやり直し（入力は 1 か所、あとは自動）

**操作は「入れて → 調べる」だけ。** 上部の 1 つの入力欄に、話題（津市 大雨）・対象（jma.go.jp、IP、メール、@名前、座標…）・文章（SNS 投稿や問い合わせを貼る）のどれを入れても、種類を自動判定して必要な処理を全部行う。モデルも端末に合わせて自動で読み込む（スマホ jev_ja_30m、PC はメモリに応じて argos / open-jev-base）。

| 入力 | 自動で行うこと | 結果 |
|---|---|---|
| 話題 | 検索プランで情報源を選び収集 → 届いた分から JEV が仕分け → 信頼性スコア | 要約・ダッシュボード・自動採用／人手確認／除外・カード一覧・**検索式（コピペ用）**・**OSINT ツールリンク** |
| 対象 | 公開 API を横断（DNS・RDAP・証明書・Wayback・OTX・ipinfo…）→ JEV が正規性・なりすましの疑い | 注意点・カード・根拠・関係図・次に調べる・検索式・ツールリンク |
| 文章 | 種類・緊急度・具体性・拡散依頼を判定 → **信頼性 0–100 と判定文** | 信頼性カード・答えカード・（質問・根拠を編集して判断し直す） |

ナビは 5 つ：🔍 調べる／💬 チャット／🧰 ツール（検索式の自動生成・OSINT Framework ツール集・端末内ツール）／📓 記録／⋯ その他（一括判断・連携・検証・バッチ・改良・可視化・モデル・設定）。
**Google dorks**：キーワードから用途別 25 式を自動生成（おすすめ＋ライブラリ）。各式に「コピー／Google／Bing／DDG」。演算子を自分で入力する必要はなく、手動で組む場合だけ「手動で組む」を開く。**OSINT Framework**：対象を入れると全カテゴリのリンクが対象入りで生成され、自動実行済みのものは「自動」と表示。

## 4-8. v1.1：検証して直した 12 点

| # | 問題 | 対応 |
|---|---|---|
| 1 | 可視化ページが空（PC 既定の argos など JEV-JA 以外のモデルでは何も出なかった） | 全モデルで「何を・どのモデルで」「logit → 確率」「時間」を表示。入力列・cos・PCA は JEV-JA のみと明記。未判断のときは「調べる」への導線 |
| 2 | チャットで返答が来るまで何も表示されず、壊れたように見える（大きいモデルは数秒） | 「判定中…（質問 3/5）」の入力中バブル、送信ロック（連打防止）、モデル名と「大きいモデルは数秒」の注意 |
| 3 | チャットの使い方が分からない | 上部に使い方パネル（3 ステップ・例文チップ）、質問セットに「問い合わせの振り分け」「アンケート回答」を追加、返答に「信頼性の内訳」「🔍 調べるで詳しく」「? で問い直す」ボタン |
| 4 | モデルを変えても何が変わるのか分からない | モデル欄で切替先を選ぶと「変わる場所」を説明。同じ文章をもう一度調べると**モデルによる違い**の表（≠ 印つき）が結果の下に出る |
| 5 | dork の条件を演算子で入力しなければならない | **ボタンで足すコンポーザー**：何を探す（PDF・Excel・画像・動画・ニュース・公的発表・SNS・掲示板・コード…）／いつ（今日〜1 年）／言語・地域（.jp・英語・lang:・loc:）／絞り方／除外／疑いを調べる。押すたびに式が更新され、コピー・Google・Bing・DDG・Yandex |
| 6 | 文章モードで OSINT ツールの長いリンク集が結果を押し下げる | 折りたたみに（対象モードでは展開） |
| 7 | 問い合わせ文でも「疑わしい（拡散しない）」と出る | 判定文を内容に合わせて変更（拡散依頼が無ければ「根拠が弱い」） |
| 8 | 推奨モデルに戻せない | 設定に「推奨モデルに戻す」 |
| 9 | オフライン時に何ができるか分からない | 入力欄下に「判断は動く／収集・調査は不可」を表示 |
| 10 | 収集結果の検索式から条件を変えられない | 検索式カードに「条件をボタンで足す（ツール）」導線 |
| 11 | チャットの初期メッセージが長い | 1 行に |
| 12 | Enter の挙動がスマホで送信になり改行できない | スマホは Enter で改行・送信はボタン（PC は Enter 送信） |

## 4-9. v2.0：機能の意味を定義し直し、既存 Jev の使い方例を機能に、画面を一から再構成

**なぜ作り直したか**：v1.x は「検索」「チャット」「一括判断」など機能名が手段（UI の形）を指していて、何のために使うのかが分からなかった。
既存 Jev（TypeSafe Jev の公式 patterns / cookbooks、Laya の presets、Jevlet、open-jev）の**文書化された使い方例**を調べ、
「JEV が情報に付ける付加価値」ごとに 4 つの動詞へまとめ、画面をそれに合わせて一から作り直した（`docs/app.js` `lab.js` `presets.js` `ui.js` `app.css`）。

| 画面 | 何をする | 元にした既存 Jev の使い方例 | JEV が付ける付加価値 |
|---|---|---|---|
| **調べる** | 話題を公開 API から集め、1 件ずつ種類・具体性・拡散依頼・緊急度・根拠との整合を判定 | RAG passage classification、re-ranking（TypeSafe cookbook） | 信頼性スコア順・「自動採用／人が確認／除外」・読む件数 N→M・信頼性の高い出典を上に |
| **見極める** | 投稿・メール・主張を 7〜8 問に**1 回で**答えて、信頼性と危険度を出す | speculative fan-out、guardrails、citation check、entity alignment | 判定から「確認すべきこと」を自動生成（公的発表の検索式・初出探し・リンク先調査）、主張と根拠を行ごとに支持／矛盾、同一かの判定、別モデルで照合 |
| **仕分ける** | 問い合わせ・メール・投稿・アンケートを何十件でも同じ問いで仕分け | intent routing、confidence-gated routing、Laya の triage / email / guard / moderation / router presets | 9 つのプリセット（質問セット＋経路ルール）、確信度のしきい値スライダーで再判定なしに経路が変わる、人手ラベル → JSONL |
| **聞く** | 文書を置いて、はい／いいえ・選択・段階・「どの行に書いてある？」で問う | line-by-line semantic find（TypeSafe cookbook）、Jevlet の command palette | 旧「チャット」を置き換え。生成しないので答えは常に確率つき、該当行をハイライト、「該当なし」を許す |

- **チャット欄の意味**：Jev 級モデルは文を生成しないので、会話 UI は誤解を招いていた。v2.0 では「文書に問う」だけの画面（聞く）にし、答えは分布と確信度で返す。
- **検索の意味**：「調べる」は収集＋仕分けの入口。長文を貼れば「見極める」へ、URL・IP などは「対象の調査」へ自動で振り分ける。
- **共通の仕組み**：確信度のしきい値（GATE、既定 0.7）を全画面で共有。しきい値未満は必ず「人が確認」へ落ちる（TypeSafe の confidence-gated routing）。
- 研究向け機能（検証・評価・学習・可視化・ノート・モデル・連携・設定・ツール）は「もっと」の下に集約（中身は v1.1 と同じ、`lab.js`）。
- 既存 Jev の使い方例の調査結果（出典つき）は `docs/theory.html` ではなく本節の表と `docs/presets.js` の `origin` に記載。

## 4''. 判断ヘッド v2 と「改良の仕組み」

学習ログの精査で、**ヘッドの入力（ModernBERT-Ja の隠れ状態、|u|≈85）が正規化されておらず tanh が飽和して全選択肢に同じ logit を出していた**ことを確認した（pre-activation ≈117、logit の標準偏差 1e-7）。v0.4 でヘッドに LayerNorm を入れ（`DecisionHead(norm=True)`）、エンコーダ固定でヘッドだけを高速に学習する `train_head.py` を追加した（ベクトルをキャッシュし 1 epoch 数秒）。

改良の入口は 3 段階：
1. **端末内**（Play「改良」）：ヘッドのみ、数百件・数十秒。人手ラベルからその場で改良
2. **PC ヘッドのみ**（`train_head.py`）：全データ・数十 epoch・数分
3. **PC 全体**（`train_jev_ja.py`）：エンコーダごと更新、GPU 推奨

既存モデルの改良は `jev_ja_mdeberta_base`（多言語 DeBERTa ＋ JEV ヘッド）と `convert_gliner2.py`（GLiNER2.5-multi の追加学習用データ変換）。

## 4-3. 最初の実測（jev_ja_30m、CPU 学習：エンコーダ 1 epoch ＋ ヘッド v2 40 epoch）

| データ | head（学習ヘッド） | cos（ゼロショット・同じエンコーダ） | 偶然 |
|---|---|---|---|
| JevBench-JA test（in-domain、2,940 state） | **精度 47.9%** / Brier 0.597 / ECE 0.026 | 23.6% / 0.661 / 0.110 | ≈30% |
| JevBench-JA OOD（未見の質問文・否定形、2,700 state） | 33.6% / 0.768 / ECE 0.182 | **39.4%** / 0.653 / 0.129 | ≈30% |

- in-domain では学習ヘッドがゼロショットの 2 倍。OOD では**逆転**し較正も崩れる ＝ 質問文を読まずに slot を暗記している（英語での Kotoba の報告を日本語で再現）。これが P1-f（拡張・否定形学習）の出発点
- 端末内学習（Play「改良」）: jnli/jcola/wrime 96 件・6 epoch・14 秒で val 33.3% → 41.7%、Brier 0.712 → 0.658
- 端末推論: 1 判断 26 ms（PC CPU）、スマホ実機 167 ms

## 3-A. 「jev_ja_30m は Jev なのか」と、既存 Jev モデルの扱い

**Jev 級（typed-decision model）の定義**：状況（state）＋型付き質問（Choice / Score / Noul）を 1 回の forward で読み、各質問について選択肢上の**較正済み確率分布**を返す。生成しない。TypeSafe AI の製品 Jev がこの形の原点で、open-jev（Kotoba Labs）、Laya（Convai）、Tiny-Jev、Jev-Style、modernbert-ja-310m-jev などのコミュニティ実装が同じ形を再現している。
**jev_ja_30m はこの定義を満たす**（同じ入力列設計 [STATE]/[Q]/[OPT]、1 forward、native 分布、温度較正）。TypeSafe の Jev 本体とは無関係の自作実装で、モデルカード上の他の "open-jev" 系と同じ位置づけ。NLI ゼロショットや文埋め込み（v0.4 で試験的に載せたもの）は Jev 級ではないので v0.5 で外した。

**このラボで比較する既存 Jev 級モデル**（2026-09-27 に HF Hub を "jev / open-jev / typed-decision / laya / tiny-jev / jev-style" で最終更新順に再検索し、小型・高性能・ライセンス明記・入力形式が公開されているものを選定。ONNX 同梱でも入力形式が独自なもの（例 JevK5-Lite）や、GGUF＋独自ヘッドで PC 専用のもの（fukayatti0/jev-japanese-judgment-v2）は表に残すが端末には載せない）

| モデル | 形 | 大きさ | 言語 | ライセンス | どこで動く | 備考 |
|---|---|---|---|---|---|---|
| **jev_ja_30m**（自作） | [STATE][Q][OPT] 1 系列 ＋ MLP ヘッド | 37M / int8 47MB | ja | MIT(backbone) | 端末内（スマホ可）・学習可 | 本研究の対象 |
| **argos1111/modernbert-ja-310m-jev** | 「質問: …\n状況: …」×候補 の cross-encoder | 310M / int8 316MB | ja | CC BY-SA 4.0 | 端末内（PC ブラウザ）・PC | JGLUE 系で検証精度 0.90。同じデータで学習された**最重要の比較対象** |
| **onnx-community/open-jev-deberta-v3-large-ONNX** | open-jev 形（seg / pair_q / pair_opt） | 435M / q4 480MB | en | Apache-2.0 | 端末内（PC ブラウザ、HF から取得） | Kotoba の参照実装 |
| lostargon/Tiny-Jev | Qwen3-0.6B ベース、System-One API | 0.6B | en | Apache-2.0 | PC | `tiny_jev` アダプタ |
| chaoliangUNSW/Jev-Style-0.8B-Decision-v3 | Qwen3.5-0.8B ベース | 0.8B / 4bit 0.53GB | 多言語（ja 含む） | Apache-2.0 | PC（`pip install "jev-style[torch]"`） | Banking77 68%、JevBench 64% |

| fukayatti0/jev-japanese-judgment-v2 | LFM2.5-1.2B-JP ＋ ヘッド | 1.2B | ja | LFM（要確認） | PC（独自コード） | JNLI 85.8%、JSTS 58% |

**最初の比較（docs/bench/jevbench_ja_small.jsonl、270 state、in-domain + OOD 混在）**

| モデル | 学習 | 精度 | Brier | ECE | p50 ms（PC CPU） | 大きさ |
|---|---|---|---|---|---|---|
| argos_ja_310m（既存、日本語） | JGLUE 等で fine-tune 済 | **54.6%** | 0.607 | 0.167 | 985 | 316MB |
| **laya_multi_322m（既存、Laya 多言語、v0.6）** | 日本語データでは未学習（ゼロショット） | 50.8% | 0.743 | 0.292 | 274 | 323MB |
| jev_ja_30m（自作 v2） | JevBench-JA で学習 | 40.8% | 0.680 | 0.130 | **27** | 47MB |
| jev_ja_30m_cos（ゼロショット） | なし | 31.5% | 0.653 | **0.047** | 25 | 47MB |

→ 既存の 310M は精度で 14 pt 上、自作 30m は 36 倍速く 7 分の 1 の大きさ。**Laya は日本語を一切学習していないのに 50.8%**（jnli 87%、fever_en 77%、jcqa 60〜73%）で、汎用の Jev 級モデルとして最も実用的。ただし ECE 0.29 と**過信**（モデルカードの記載どおり。温度 T を上げるか再較正が必要）。argos は日本語 fine-tune の効果で jnli 100%・jcola 87% だが OOD（jcola_ood 40%、fever_en_ood 33%）で落ちる。JevBench 流の「5 軸で見せる」比較がそのまま成立し、「小さく速い自作」対「大きく強い既存」対「学習なしで汎用の Laya」という三者比較が卒論の軸になる。

## 3-B. 調査ツール（動画の手法のうち合法・受動的なもの）

| 手法 | 実装 | 情報源 |
|---|---|---|
| 検索演算子（Google dorks） | URL を組み立てて開く（自動巡回しない）。site: / filetype: / intitle: / inurl: / 除外 / 期間 | Google・Bing・DDG・Yahoo!JAPAN・Google ニュース・X 検索・YouTube・Wikipedia・Wayback |
| Wayback Machine | 年ごとの最寄りスナップショット、最古の記録 | archive.org availability API |
| DNS | A / AAAA / MX / NS / TXT / CNAME | Google Public DNS（DoH） |
| WHOIS / RDAP | 登録日・期限・レジストラ・ステータス（.jp は JPRS WHOIS へ案内） | rdap.org |
| サブドメイン | 証明書透明性ログからホスト名を列挙（受動的） | crt.sh |
| ドメインの素性まとめ | 上記を 1 つの「根拠」にして情報収集の照合へ | — |
| EXIF / GPS | 自分の画像を端末内で読む（送信しない）。撮影日時・機種・座標→地図 | ブラウザ内 |
| 逆画像検索 | 画像 URL を Google Lens / Bing / Yandex / TinEye で開く | リンク |
| 実装しないもの | Shodan・HIBP（API キー・個人情報）、ユーザー名の横断照会（個人対象）、ログインが必要な自動操作、robots.txt 無視の収集 | — |

## 5. 検証済みのこと

- ブラウザ推論が Python と 3 桁一致（トークナイザ byte fallback を JS で再現）
- スマホ実機：1 判断（166 トークン・3 質問）**167 ms**（WASM）。オフライン再読込 → 自動ロード → 判断まで確認
- 公開 API：気象庁 bosai JSON・Nominatim・GSI 標高タイル・Wikipedia/Wikidata・HN は GitHub Pages から CORS で取得可。Bluesky 公開検索は実ブラウザから可。GDELT は 5 s/req
- Play 全タブ（情報収集→整理→信頼性、質問ビルダー、T/pool/head-cos 切替、検証 6 種、可視化、バッチ、ノート）を headless Chromium で動作確認
- v0.7：調査タブ（URL・座標・会社・ETH ほか）、ツール箱（変換・太陽位置・電話・露出情報・dork ライブラリ）、情報収集の逐次判定と仕分けしきい値を headless Chromium で確認（公開 API はサンドボックスから届く範囲）
- v0.6：Laya multilingual のブラウザ推論が Python（`laya` 公式 Agent）とトークン列で完全一致、確率は int8 化で最大 ±0.13（argmax は一致）。ブラウザで 6 問 4.7 秒（2 コア WASM）。一括判断・チャット `?`・要約・CI 付きバッチ・卒論表を headless Chromium で確認

## 6. 研究ロードマップ（P1 → P2）

| 段階 | 内容 | 成果物 |
|---|---|---|
| P1-a（済） | 基盤：スキーマ・アダプタ・データ 10 family・学習・ONNX・Pages・OSINT フロー・チャット・端末内学習・HF モデル追加 | このリポ |
| P1-b | JEV-JA 30m/70m を GPU で本学習（2〜3 ep、augment 0.7）。in-domain / OOD / 較正 / 端末 ms | 結果表、公開モデル |
| P1-c | 比較：cos ゼロショット、NLI、GLiNER2.5-multi、open-jev（英語 family）、Qwen3 verbalized。fast_decisions で GLiNER2 と同条件 | リーダーボード |
| P1-d | 既存 Jev 級モデルの改良：mDeBERTa-base に JEV ヘッド、GLiNER2.5-multi の追加学習、**Laya multilingual の日本語 fine-tune（`laya` の RLCD ノートブック、Kaggle 2×T4 で 4〜5 h）と再較正（ECE 0.29 → 0.1 台）** | 5 軸比較 |
| P1-e | 情報収集の評価：Play で 5〜10 テーマを収集 → 人手ラベル 300〜500 件（osint_ja）→ 学習前後の信頼性判定精度、モデル間 κ、新着検出 | 独自データセット |
| P1-f | アブレーション：span/marker、λ、拡張、int8、温度 | 卒論の実験章 |
| P2 | 応用：事前同期したハザード・避難所データと組み合わせた災害時オフライン運用（同じ JEV・同じ地理照合） | 応用章 |

## 7. 参考
- Kotoba Labs typed-decisions（Jev 形の再現・span-pool 知見）: https://github.com/kotoba-lang/typed-decisions
- JevBench（5 軸・native/verbalized）: https://benchmarkheaven.com/jev-models/v1
- GLiNER2.5-Decide / fast-decisions: https://huggingface.co/fastino/GLiNER2.5-Decide
