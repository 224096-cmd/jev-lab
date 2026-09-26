/* 用途テンプレート（実用レベルの見本）。既定では何も入れず、選んだときだけ Playground に入る */
export const TEMPLATES = [
  {
    id: "support", name: "問い合わせメールの振り分け（企業・学校窓口）",
    desc: "実際の問い合わせ文から担当・行動・緊急度・返信要否を一括判定。Tiny-Jev / Laya のベンチと同じ型。",
    state: "件名: 二重に引き落とされています\n\n先週 9/19 に注文番号 A-20931 で購入した際、クレジットカードに 4,980 円が 2 回請求されていました。カード会社にも確認済みです。すでに一度チャットで問い合わせましたが「担当に回します」のまま返事がありません。今週中に返金されない場合はキャンセルと消費生活センターへの相談を検討します。至急対応をお願いします。",
    context: [],
    questions: [
      { type: "choice", id: "dept", instructions: "この問い合わせを担当すべき部署", options: ["請求・返金", "配送・物流", "技術サポート", "営業・契約", "法務・コンプライアンス", "その他"] },
      { type: "choice", id: "action", instructions: "次に取るべき対応", options: ["返信して解決", "上位担当へエスカレーション", "返金処理を開始", "情報を追加で確認", "対応不要"] },
      { type: "score", id: "urgency", instructions: "対応の緊急度", levels: ["0", "1", "2", "3", "4", "5"] },
      { type: "noul", id: "needs_reply", instructions: "顧客への返信が必要か" },
      { type: "noul", id: "churn_risk", instructions: "顧客が解約や外部機関への相談を示唆しているか" },
    ],
    texts: ["配送状況を教えてください。追跡番号は 1234-5678 です。特に急いでいません。", "ログインしようとすると「セッションが無効」と出ます。パスワードリセットも試しました。仕事で今日中に使う必要があります。", "貴社サービスの法人契約について、10 名分の見積もりをお願いできますか。"],
  },
  {
    id: "sns_verify", name: "SNS 投稿の検証（拡散前チェック）",
    desc: "投稿の種類・具体性・拡散依頼・根拠との整合を判定。context に公的情報や地理情報を入れると「物理的にあり得るか」を見られる。",
    state: "【拡散希望】午後 3 時ごろ、津市栗真町屋町の志登茂川が氾濫して県道 10 号が完全に冠水、車が流されているのを見ました。市役所は何も発表していません。近くの人はすぐ逃げて！ RT お願いします！",
    context: ["気象庁（三重県）: 14:30 時点で津市に大雨警報（浸水害）・洪水警報を発表中", "志登茂川: 県管理の二級河川。栗真町屋町付近は洪水浸水想定区域（想定最大 0.5〜3.0m）", "津市: 15:10 に栗真地区へ高齢者等避難（警戒レベル 3）を発令"],
    questions: [
      { type: "choice", id: "kind", instructions: "この情報の種類", options: ["公的機関の発表", "報道", "一般の投稿・目撃", "意見・感想", "宣伝・無関係"] },
      { type: "noul", id: "has_specifics", instructions: "日時・場所・数量など検証可能な具体情報が含まれているか" },
      { type: "noul", id: "asks_spread", instructions: "拡散や転送を呼びかける表現があるか" },
      { type: "noul", id: "supported", instructions: "根拠（context）の内容と整合しているか" },
      { type: "choice", id: "verdict", instructions: "この投稿の扱い", options: ["整合（要確認事項あり）", "確認不能", "矛盾あり"] },
      { type: "score", id: "urgency", instructions: "今すぐ対応や確認が必要な度合い", levels: ["0", "1", "2", "3", "4", "5"] },
    ],
    texts: ["津市役所【公式】: 15:10、栗真地区に高齢者等避難（警戒レベル 3）を発令しました。指定避難所は栗真小学校です。", "美杉小学校が津波で水没したらしい。生徒が取り残されてるって。拡散して！", "志登茂川の橋のあたり、水位がだいぶ上がってきてる。写真は 15:05 撮影。"],
  },
  {
    id: "news_triage", name: "ニュース記事の仕分け（テーマ・立場・一次情報）",
    desc: "記事本文からテーマ、一次情報の有無、論調を判定。情報収集で集めた報道の整理に。",
    state: "総務省は 26 日、生成 AI を用いた偽・誤情報対策に関する検討会の報告書案を公表した。報告書案では、プラットフォーム事業者に対し、AI 生成コンテンツへのラベル付けや、選挙期間中の対応体制の整備を求めている。一方、表現の自由への配慮から、法的義務ではなく自主的な取り組みを促す形とし、来年の通常国会での法改正は見送る方向だ。有識者からは「実効性に欠ける」との指摘も出ている。",
    context: [],
    questions: [
      { type: "choice", id: "topic", instructions: "記事の主なテーマ", options: ["AI 規制・政策", "AI 技術・製品", "災害・防災", "教育", "経済・企業", "その他"] },
      { type: "noul", id: "primary", instructions: "公的機関や当事者の一次情報（発表・文書）に基づいているか" },
      { type: "choice", id: "stance", instructions: "記事の論調", options: ["中立・事実報道", "肯定的", "批判的", "不明"] },
      { type: "score", id: "importance", instructions: "研究テーマ（小型判断モデルによる情報整理）との関連度", levels: ["低い", "やや低い", "中程度", "高い", "非常に高い"] },
    ],
    texts: ["国内スタートアップが、スマートフォン上で動作する 0.5B パラメータの判断モデルを公開した。生成を行わず、入力文に対する分類・スコアリングを 1 回の推論で返す。", "県内の中学校で、生成 AI の出力を生徒が批判的に検証する授業が始まった。"],
  },
  {
    id: "survey", name: "授業・行事アンケートの自由記述分析",
    desc: "kuwabe-aki-analyzer の AI 分析を JEV に置き換える想定。感情・要望・話題を端末内で付与し集計へ。",
    state: "班活動の時間が短くて、最後まで話し合えなかったのが残念でした。でもタブレットで調べながら進めるのは楽しかったし、普段話さない人とも協力できました。次はもう少し時間をとってほしいです。あと、発表のときにマイクが聞き取りにくかったです。",
    context: [],
    questions: [
      { type: "score", id: "sentiment", instructions: "この回答の全体的な感情", levels: ["否定的", "やや否定的", "中立", "やや肯定的", "肯定的"] },
      { type: "noul", id: "request", instructions: "授業や行事への要望・改善提案が含まれているか" },
      { type: "choice", id: "topic", instructions: "主に何について書かれているか", options: ["時間配分", "教材・機器", "班活動・人間関係", "内容の難易度", "先生の説明", "その他"] },
      { type: "noul", id: "positive_peer", instructions: "他の生徒との協力を肯定的に述べているか" },
    ],
    texts: ["説明が早くてついていけなかった。プリントの字も小さい。", "実験の結果が予想と違って面白かった。もっとやりたい。", "特にない。"],
  },
  {
    id: "reflection", name: "生徒の振り返り文の評価（技術科）",
    desc: "振り返りから理解の深さ・原因分析の有無・次に必要な支援を判定。教師のコメント作成の下書きに。",
    state: "はんだ付けで最初は失敗して、はんだが玉になってしまった。理由を考えると、こて先の温度が上がる前に付けていたのと、部品の足を温める時間が短かったからだと思う。2 回目は 3 秒数えてから付けたらうまくいった。次は基板の向きを間違えないように、印刷を見てから差し込むようにしたい。",
    context: [],
    questions: [
      { type: "score", id: "understanding", instructions: "学習内容の理解の深さ", levels: ["1", "2", "3", "4", "5"] },
      { type: "noul", id: "reflective", instructions: "失敗の原因を自分で分析できているか" },
      { type: "noul", id: "next_step", instructions: "次の行動を具体的に書いているか" },
      { type: "choice", id: "support", instructions: "次の授業で最も必要な支援", options: ["基礎の再説明", "手順の確認", "発展課題", "安全指導", "特になし"] },
    ],
    texts: ["よくわからなかった。", "全部できた。簡単だった。", "LED が光らなかった。理由はわからない。"],
  },
  {
    id: "disaster", name: "災害時の状況トリアージ（応用例）",
    desc: "被災者の状況文から災害種・緊急度・垂直避難の適否を判定。context にハザード情報を入れる。P2 の応用。",
    state: "津市栗真町屋町の自宅 2 階にいます。窓から見ると道路が 30cm くらい冠水し、雨はまだ強いです。足首を捻挫していて走れません。同居の母（82 歳）は歩けますが階段は手すりが必要です。",
    context: ["栗真町屋町周辺は志登茂川の洪水浸水想定区域（想定最大 0.5〜3.0m）", "最寄りの指定避難所: 栗真小学校（洪水対応◯、標高 4m、徒歩 12 分）"],
    questions: [
      { type: "choice", id: "hazard", instructions: "この状況で主に起きている災害は何か", options: ["洪水・内水氾濫", "津波", "土砂災害", "地震の建物被害", "不明"] },
      { type: "score", id: "urgency", instructions: "今すぐ移動を始める必要性", levels: ["低い", "やや低い", "中程度", "高い", "非常に高い"] },
      { type: "noul", id: "vertical", instructions: "屋外への水平避難より、その場で上階にとどまる垂直避難が適切か" },
      { type: "noul", id: "vulnerable", instructions: "移動に配慮が必要な人（負傷者・高齢者・乳幼児）がいるか" },
    ],
    texts: ["強い揺れが 1 分ほど続いた。今は海沿いの駐車場。津波注意報が出たとラジオで聞いた。家族全員無事で歩ける。", "裏山から水が濁って流れてきて、地鳴りのような音がする。家は山のすぐ下。"],
  },
];

/* 情報収集のプリセット（調べたいこと・向いている情報源・言語）。既定は未選択 */
export const OSINT_PRESETS = [
  { id: "jev", name: "Jev / 型付き判断モデルの最新動向", query: "Jev typed decision model, open-jev, typed-decisions", sources: ["hn", "github", "arxiv", "crossref", "bluesky", "mastodon"], lang: "en", desc: "TypeSafe Jev、open-jev、Laya、GLiNER2 など「生成しない判断モデル」の話題。種類（発表/報道/投稿/意見）と具体性で仕分ける。関連研究の更新に。" },
  { id: "ai_jp", name: "AI（日本語圏：規制・教育・製品）", query: "生成AI 規制, AI 教育 学校, 小型言語モデル", sources: ["bluesky", "gdelt", "wikipedia", "wikidata"], lang: "ja", desc: "日本語の AI 関連ニュースと投稿。公的発表（総務省・文科省）と意見を分ける練習に。" },
  { id: "ai_ondevice", name: "端末内推論（ONNX / WebGPU / 小型モデル）", query: "onnx runtime web webgpu, small language model on-device", sources: ["hn", "gdelt"], lang: "en", desc: "本ラボの技術基盤に関する技術系ニュース。" },
  { id: "disaster", name: "災害（気象庁＋地域名）", query: "津市 大雨", sources: ["jma", "wikipedia", "nominatim", "bluesky"], lang: "ja", desc: "公的発表（気象庁）と一般投稿を地理照合つきで並べる。" },
  { id: "factcheck", name: "噂の検証（地名・施設名で）", query: "美杉小学校 津波", sources: ["wikipedia", "wikidata", "nominatim", "bluesky"], lang: "ja", desc: "地名→座標→標高と Wikipedia の基礎情報を根拠にして、投稿の物理的整合を判定する。" },
  { id: "paper", name: "論文サーベイ（arXiv・Crossref・Semantic Scholar）", query: "typed decisions calibrated classification", sources: ["arxiv", "crossref", "semanticscholar", "github"], lang: "en", desc: "卒論の関連研究用。論文・コードを一次情報として集め、JEV で関連度と種類を仕分ける。" },
  { id: "edu", name: "教育・技術科（学習指導要領・ICT）", query: "技術科 プログラミング教育, 情報モラル 授業", sources: ["gdelt", "bluesky", "wikipedia"], lang: "ja", desc: "教材研究用。報道と現場の声を分ける。" },
];

/* よく使う質問（「文章を判断」の「よく使う」から 1 クリックで追加） */
export const QUICK_QUESTIONS = [
  { type: "choice", id: "kind", instructions: "この情報の種類", options: ["公的機関の発表", "報道", "一般の投稿・目撃", "意見・感想", "宣伝・無関係"] },
  { type: "score", id: "urgency", instructions: "今すぐ対応や確認が必要な度合い", levels: ["0", "1", "2", "3", "4", "5"] },
  { type: "noul", id: "has_specifics", instructions: "日時・場所・数量など検証可能な具体情報が含まれているか" },
  { type: "noul", id: "asks_spread", instructions: "拡散や転送を呼びかける表現があるか" },
  { type: "noul", id: "supported", instructions: "根拠（context）の内容と整合しているか" },
  { type: "noul", id: "needs_reply", instructions: "返信が必要か" },
  { type: "score", id: "sentiment", instructions: "この文章の全体的な感情", levels: ["否定的", "やや否定的", "中立", "やや肯定的", "肯定的"] },
  { type: "choice", id: "dept", instructions: "この問い合わせを担当すべき部署", options: ["請求・返金", "配送・物流", "技術サポート", "営業・契約", "その他"] },
  { type: "choice", id: "topic", instructions: "主なテーマ", options: ["AI 規制・政策", "AI 技術・製品", "災害・防災", "教育", "経済・企業", "その他"] },
  { type: "noul", id: "is_ad", instructions: "宣伝・広告か" },
];
