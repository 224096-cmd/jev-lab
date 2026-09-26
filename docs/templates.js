/* 用途テンプレート：state / context / questions / トリアージ用テキストの見本
   「どんな用途で使えるか」を具体例で示す。Play の「テンプレ」から 1 クリックで読み込める */
export const TEMPLATES = [
  {
    id: "disaster", name: "災害時トリアージ（P2 の中核）",
    desc: "被災者の状況文から災害種・緊急度・垂直避難の適否を一括判断。context にハザード DB の根拠を入れる。",
    state: "津市栗真町屋町の自宅2階にいます。窓から見ると道路が30cmくらい冠水し、雨はまだ強いです。足首を捻挫していて走れません。",
    context: ["栗真町屋町周辺は志登茂川の洪水浸水想定区域（想定最大 0.5〜3.0m）", "最寄りの指定避難所: 栗真小学校（洪水対応◯、標高 4m、徒歩 12 分）"],
    questions: [
      { type: "choice", id: "hazard", instructions: "この状況で主に起きている災害は何か", options: ["洪水・内水氾濫", "津波", "土砂災害", "地震の建物被害", "不明"] },
      { type: "score", id: "urgency", instructions: "今すぐ移動を始める必要性", levels: ["低い", "やや低い", "中程度", "高い", "非常に高い"] },
      { type: "noul", id: "vertical", instructions: "屋外への水平避難より、その場で上階にとどまる垂直避難が適切か" },
      { type: "noul", id: "injured", instructions: "本人に移動を妨げる負傷があるか" },
    ],
    texts: ["強い揺れが1分ほど続いた。今は海沿いの駐車場。津波注意報が出たとラジオで聞いた。家族全員無事で歩ける。", "裏山から水が濁って流れてきて、地鳴りのような音がする。家は山のすぐ下。", "停電しているが家は無事。近所の川は普段どおり。"],
  },
  {
    id: "sns", name: "SNS 投稿の検証（デマ・拡散依頼・地理矛盾）",
    desc: "投稿の種類・緊急度・地理的矛盾・拡散依頼の強さを判断。context に標高や想定浸水深を入れると「物理的にあり得ない」を検出できる。",
    state: "【拡散希望】津市の美杉小学校が津波で完全に水没しました！生徒が取り残されています！RTお願いします！",
    context: ["美杉小学校: 標高 約 250m、津市美杉町（内陸、海岸から約 25km）", "三重県の津波浸水想定の最大浸水深は沿岸部で 5m 程度"],
    questions: [
      { type: "choice", id: "kind", instructions: "投稿の種類", options: ["公式発表", "現地報告", "救助要請", "拡散依頼", "不明"] },
      { type: "noul", id: "geo_contra", instructions: "根拠の地理情報に照らして、この投稿の内容は物理的に矛盾しているか" },
      { type: "choice", id: "verdict", instructions: "この投稿の扱い", options: ["矛盾あり", "確認不能", "整合"] },
      { type: "score", id: "spread", instructions: "拡散を促す表現の強さ", levels: ["なし", "弱い", "強い"] },
    ],
    texts: ["安濃川の橋のあたり、水位がかなり上がってる。近くの人は気をつけて。", "津市役所: 12時00分、津市全域に高齢者等避難（警戒レベル3）を発令しました。", "動物園からライオンが逃げたらしい！みんな気をつけて！！拡散して！"],
  },
  {
    id: "survey", name: "授業・行事アンケートの自由記述分析",
    desc: "kuwabe-aki-analyzer の AI 分析を JEV に置換する想定。感情・要望の有無・カテゴリを端末内で高速に付与し、集計へ渡す。",
    state: "班活動の時間が短くて、最後まで話し合えなかった。でもタブレットで調べるのは楽しかった。",
    context: [],
    questions: [
      { type: "score", id: "sentiment", instructions: "この回答の全体的な感情", levels: ["否定的", "やや否定的", "中立", "やや肯定的", "肯定的"] },
      { type: "noul", id: "request", instructions: "授業や行事への要望・改善提案が含まれているか" },
      { type: "choice", id: "topic", instructions: "主に何について書かれているか", options: ["時間配分", "教材・機器", "班活動・人間関係", "内容の難易度", "先生の説明", "その他"] },
    ],
    texts: ["説明が早くてついていけなかった。", "友達と協力できて楽しかった。また やりたい。", "プリントの字が小さくて読みにくい。大きくしてほしい。", "特にない。"],
  },
  {
    id: "reflection", name: "生徒の振り返り文の評価（技術科）",
    desc: "振り返りの記述から理解度と次に必要な支援を判断。教師のコメント作成の下書きに。",
    state: "はんだ付けで最初は失敗したけど、こて先の温度を待ってから付けるとうまくいった。次は基板の向きを間違えないようにしたい。",
    context: [],
    questions: [
      { type: "score", id: "understanding", instructions: "学習内容の理解の深さ", levels: ["1", "2", "3", "4", "5"] },
      { type: "choice", id: "support", instructions: "次の授業で最も必要な支援", options: ["基礎の再説明", "手順の確認", "発展課題", "安全指導", "特になし"] },
      { type: "noul", id: "reflective", instructions: "失敗の原因を自分で分析できているか" },
    ],
    texts: ["よくわからなかった。", "全部できた。簡単だった。", "LEDが光らなかった。理由はわからない。"],
  },
  {
    id: "routing", name: "問い合わせの振り分け（学校・自治体窓口）",
    desc: "メールやフォームの文面を担当・緊急度・人の対応要否に振り分ける。GLiNER2.5-Decide の典型用途の日本語版。",
    state: "来週の体育祭について、保護者の駐車場はありますか。祖父母も来る予定で、足が悪いので近くに停めたいです。",
    context: [],
    questions: [
      { type: "choice", id: "dept", instructions: "担当部署", options: ["教務", "生徒指導", "事務・会計", "行事担当", "保健", "その他"] },
      { type: "score", id: "urgency", instructions: "対応の緊急度", levels: ["0", "1", "2", "3", "4", "5"] },
      { type: "noul", id: "needs_human", instructions: "定型文ではなく担当者の個別対応が必要か" },
    ],
    texts: ["子どもが昨日から熱があり、今日は休ませます。", "給食費の引き落としが二重になっています。至急確認してください。", "PTA だよりの PDF が開けません。"],
  },
  {
    id: "news", name: "ニュース記事のトピック分類（ベンチ用）",
    desc: "livedoor ニュースコーパスと同じ 9 カテゴリ。JevBench-JA の Choice family と同じ設定で、学習前後の差を見る。",
    state: "新型スマートフォンの発表会が開かれ、カメラ性能の向上とバッテリー持続時間の改善が強調された。",
    context: [],
    questions: [
      { type: "choice", id: "topic", instructions: "記事のカテゴリ", options: ["トピックニュース", "スポーツ", "映画", "家電・IT", "ライフハック", "女性向け", "独身男性向け", "エンタメ", "モバイル"] },
    ],
    texts: ["決勝戦は延長の末、PK 戦で決着した。", "映画の続編が来夏公開と発表された。", "節約のコツは固定費の見直しから。"],
  },
];


/* 情報収集のプリセット（調べたいこと・向いている情報源・言語） */
export const OSINT_PRESETS = [
  { id: "jev", name: "Jev / 型付き判断モデルの最新動向", query: "Jev typed decision model", sources: ["hn", "gdelt", "bluesky", "wikipedia"], lang: "en", desc: "Jev（TypeSafe AI）、open-jev、GLiNER2 など「生成しない判断モデル」の話題を集め、種類（発表/報道/投稿/意見）と具体性で仕分ける。卒論の関連研究の更新に。" },
  { id: "ai_jp", name: "AI（日本語圏の話題）", query: "生成AI 規制", sources: ["bluesky", "gdelt", "wikipedia", "wikidata"], lang: "ja", desc: "日本語の AI 関連ニュースと投稿。公的発表（総務省・経産省）と意見を分ける練習に。" },
  { id: "ai_onnx", name: "小型モデルの端末内推論（ONNX / WebGPU）", query: "onnx runtime web webgpu", sources: ["hn", "gdelt"], lang: "en", desc: "本ラボの技術基盤に関する技術系ニュース。" },
  { id: "disaster", name: "災害（気象庁＋地域名）", query: "津市 大雨", sources: ["jma", "wikipedia", "nominatim", "bluesky"], lang: "ja", desc: "公的発表（気象庁）と一般投稿を地理照合つきで並べる。P2 の応用。" },
  { id: "factcheck", name: "噂の検証（地名・施設名で）", query: "美杉小学校 津波", sources: ["wikipedia", "wikidata", "nominatim", "bluesky"], lang: "ja", desc: "地名→座標→標高と Wikipedia の基礎情報を根拠にして、投稿の物理的整合を判定する。" },
  { id: "edu", name: "教育・技術科（学習指導要領・ICT）", query: "技術科 プログラミング教育", sources: ["gdelt", "bluesky", "wikipedia"], lang: "ja", desc: "教材研究用。報道と現場の声を分ける。" },
];
