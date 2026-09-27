/* 用途プリセット（v2.0）— 既存 Jev の公式な使い方例（TypeSafe の patterns/cookbooks、Laya の presets、Jevlet、open-jev）を日本語の用途に置き換えたもの
   各プリセット = 質問セット（1 forward で全部問う「fan-out」）＋ 経路ルール（答えと確信度で行き先を決める「confidence-gated routing」）
   ルールは上から順に評価し、最初に当たったものが行き先。when(A, gate) は答え辞書 A（id → answer）と確信度しきい値 gate を受ける */
const L5 = ["0", "1", "2", "3", "4", "5"];
const low = (A, ids, gate) => ids.some(id => A[id] && A[id].confidence < gate);
export const TRIAGE_PRESETS = [
  { id: "support", name: "問い合わせの振り分け", icon: "📨", origin: "TypeSafe「speculative fan-out」/ Laya triage preset", desc: "窓口に届いた文を、担当・次の対応・緊急度・返信要否・離反リスクに一括で仕分ける。",
    questions: [
      { type: "choice", id: "dept", instructions: "この問い合わせを担当すべき部署", options: ["請求・返金", "配送・物流", "技術サポート", "営業・契約", "法務・苦情", "その他・判断できない"] },
      { type: "choice", id: "action", instructions: "次に取るべき対応", options: ["返信して解決", "上位担当へエスカレーション", "返金処理を開始", "情報を追加で確認", "対応不要"] },
      { type: "score", id: "urgency", instructions: "対応の緊急度", levels: L5 },
      { type: "noul", id: "needs_reply", instructions: "相手への返信が必要か" },
      { type: "noul", id: "churn_risk", instructions: "解約・返金要求・外部機関への相談など、離反や紛争を示唆しているか" }],
    routes: [
      { name: "🔴 今すぐ・上位へ", cls: "bad", when: (A, g) => A.churn_risk?.noul && A.churn_risk.confidence >= g || A.urgency?.score >= 4 },
      { name: "🟡 人が確認", cls: "warn", when: (A, g) => low(A, ["dept", "action"], g) || /判断できない/.test(A.dept?.choice || "") },
      { name: "🟢 担当へ自動転送", cls: "ok", when: (A, g) => A.needs_reply?.noul },
      { name: "⚪ 対応不要", cls: "muted", when: () => true }],
    columns: ["dept", "action", "urgency", "needs_reply", "churn_risk"], sort: "urgency",
    examples: ["先週の注文が二重に引き落とされています。今週中に返金がなければ消費生活センターに相談します。", "配送状況を教えてください。追跡番号は 1234-5678 です。急ぎません。", "ログインすると「セッションが無効」と出ます。今日中に使う必要があります。"] },
  { id: "email", name: "メールの仕分け（迷惑・フィッシング）", icon: "📥", origin: "Laya email preset（spam F1 0.99 / phishing 0.98）", desc: "受信箱を、種類・迷惑メール・フィッシング・緊急度・返信要否で仕分け、危険なものを隔離。",
    questions: [
      { type: "choice", id: "category", instructions: "このメールの種類", options: ["業務連絡", "請求・支払い", "営業・宣伝", "イベント・案内", "システム通知", "個人・私用", "その他"] },
      { type: "noul", id: "is_spam", instructions: "迷惑メール（無差別に送られた宣伝や詐欺）か" },
      { type: "noul", id: "is_phishing", instructions: "フィッシング（偽サイトへの誘導・認証情報や送金の要求・なりすまし）か" },
      { type: "score", id: "urgency", instructions: "返信や対応の緊急度", levels: L5 },
      { type: "noul", id: "needs_reply", instructions: "返信が必要か" }],
    routes: [
      { name: "🔴 隔離（フィッシング）", cls: "bad", when: (A, g) => A.is_phishing?.noul && A.is_phishing.confidence >= g },
      { name: "⚪ 迷惑メール", cls: "muted", when: (A, g) => A.is_spam?.noul && A.is_spam.confidence >= g },
      { name: "🟡 人が確認", cls: "warn", when: (A, g) => low(A, ["is_phishing", "is_spam", "category"], g) },
      { name: "🟠 今日返信", cls: "warn", when: A => A.needs_reply?.noul && A.urgency?.score >= 3 },
      { name: "🟢 受信箱", cls: "ok", when: () => true }],
    columns: ["category", "is_spam", "is_phishing", "urgency", "needs_reply"], sort: "urgency",
    examples: ["【重要】アカウントが制限されました。24 時間以内に下記リンクから本人確認をしてください。 http://mufg-secure-login.example.net", "来週の職員会議は 10 月 3 日 16:00 に変更になりました。資料は共有フォルダにあります。", "期間限定！今だけ 90% OFF、今すぐクリック！"] },
  { id: "moderation", name: "投稿のモデレーション", icon: "🛡", origin: "Laya moderation preset / TypeSafe guardrails cookbook", desc: "コミュニティや掲示板の投稿を、攻撃・嫌がらせ・脅し・スパム・深刻度で仕分けて、公開／非表示／要確認に。",
    questions: [
      { type: "noul", id: "toxic", instructions: "侮辱・差別・攻撃的な表現を含むか" },
      { type: "noul", id: "harassment", instructions: "特定の人物への嫌がらせや個人情報の暴露を含むか" },
      { type: "noul", id: "threat", instructions: "暴力や危害の脅しを含むか" },
      { type: "noul", id: "spam", instructions: "宣伝・スパム・無関係な投稿か" },
      { type: "score", id: "severity", instructions: "放置した場合の害の深刻さ", levels: L5 }],
    routes: [
      { name: "🔴 即対応（脅し）", cls: "bad", when: (A, g) => A.threat?.noul && A.threat.confidence >= g },
      { name: "🟠 非表示・確認", cls: "warn", when: A => A.severity?.score >= 3 || A.harassment?.noul },
      { name: "🟡 人が確認", cls: "warn", when: (A, g) => low(A, ["toxic", "harassment", "threat"], g) },
      { name: "⚪ スパム", cls: "muted", when: (A, g) => A.spam?.noul && A.spam.confidence >= g },
      { name: "🟢 公開", cls: "ok", when: () => true }],
    columns: ["toxic", "harassment", "threat", "spam", "severity"], sort: "severity",
    examples: ["こいつの住所知ってるぞ。次見かけたらただじゃおかない。", "この教材、うちの学校でも使ってます。設定でつまずいたので手順を共有します。", "副業で月 50 万！詳しくはプロフのリンクから"] },
  { id: "sns", name: "SNS 投稿の検証（拡散前）", icon: "🔎", origin: "本研究の主題（種類・具体性・拡散依頼・根拠との整合 → 信頼性）", desc: "災害や事件の投稿を、公的発表／報道／目撃／意見／宣伝に分け、具体性と拡散依頼から信頼性を出す。根拠を入れると整合も見る。",
    questions: [
      { type: "choice", id: "kind", instructions: "この情報の種類", options: ["公的機関の発表", "報道", "一般の投稿・目撃", "意見・感想", "宣伝・無関係"] },
      { type: "noul", id: "has_specifics", instructions: "日時・場所・数量など検証可能な具体情報が含まれているか" },
      { type: "noul", id: "asks_spread", instructions: "拡散や転送を呼びかける表現があるか" },
      { type: "score", id: "urgency", instructions: "今すぐ対応や確認が必要な度合い", levels: L5 },
      { type: "noul", id: "supported", instructions: "根拠（context）の内容と整合しているか", needsContext: true }],
    routes: [
      { name: "🔴 矛盾の疑い", cls: "bad", when: (A, g) => A.supported && A.supported.p_yes < 0.35 && A.supported.confidence >= g },
      { name: "🟠 拡散依頼・要確認", cls: "warn", when: A => A.asks_spread?.noul },
      { name: "🟡 人が確認", cls: "warn", when: (A, g) => low(A, ["kind"], g) || (A.kind?.choice === "一般の投稿・目撃" && A.urgency?.score >= 3) },
      { name: "🟢 一次情報・報道", cls: "ok", when: A => ["公的機関の発表", "報道"].includes(A.kind?.choice) },
      { name: "⚪ 意見・その他", cls: "muted", when: () => true }],
    columns: ["kind", "has_specifics", "asks_spread", "urgency", "supported"], sort: "urgency",
    examples: ["【拡散希望】午後 3 時ごろ志登茂川が氾濫して県道 10 号が冠水、車が流されています。市役所は何も発表していません。RT お願いします！", "津市役所【公式】: 15:10、栗真地区に高齢者等避難（警戒レベル 3）を発令しました。指定避難所は栗真小学校です。", "雨すごいね。今日は家でおとなしくしてよう。"] },
  { id: "scam", name: "詐欺・誘導のチェック", icon: "🚨", origin: "TypeSafe guardrails（hazard nouls + severity）/ Tiny-Jev agent-safety", desc: "届いた文が、急かし・個人情報や送金の要求・公的機関や企業のなりすまし・偽リンクを含むかを見て危険度を出す。",
    questions: [
      { type: "noul", id: "pressure", instructions: "期限や罰則で相手を急かす表現があるか" },
      { type: "noul", id: "asks_personal", instructions: "個人情報・認証情報・送金・ギフトカードなどを要求しているか" },
      { type: "noul", id: "impersonation", instructions: "公的機関・金融機関・有名企業・知人になりすましている疑いがあるか" },
      { type: "noul", id: "link_lure", instructions: "リンクや添付を開かせようとしているか" },
      { type: "score", id: "harm", instructions: "従った場合の被害の大きさ", levels: L5 }],
    routes: [
      { name: "🔴 危険（従わない）", cls: "bad", when: (A, g) => (A.asks_personal?.noul && A.asks_personal.confidence >= g) || A.harm?.score >= 3.5 },
      { name: "🟠 疑わしい", cls: "warn", when: A => [A.pressure, A.impersonation, A.link_lure].filter(x => x?.noul).length >= 2 },
      { name: "🟡 人が確認", cls: "warn", when: (A, g) => low(A, ["asks_personal", "impersonation"], g) },
      { name: "🟢 問題なし", cls: "ok", when: () => true }],
    columns: ["pressure", "asks_personal", "impersonation", "link_lure", "harm"], sort: "harm",
    examples: ["【税務署】還付金 23,400 円の受け取り手続きが未完了です。本日中に下記 URL からマイナンバーと口座を登録してください。", "お母さん、携帯が壊れて番号変わった。今日中に 30 万必要なんだけど振り込んでくれない？", "来月の同窓会の出欠を 10 日までに返信してください。"] },
  { id: "survey", name: "アンケート・自由記述の集計", icon: "📝", origin: "TypeSafe composite scoring / feature extraction", desc: "授業や行事の自由記述を、感情・話題・具体的な提案・個別対応の要否で集計する（教育現場向け）。",
    questions: [
      { type: "score", id: "sentiment", instructions: "回答の全体的な感情", levels: ["否定的", "やや否定的", "中立", "やや肯定的", "肯定的"] },
      { type: "choice", id: "topic", instructions: "主な話題", options: ["授業内容・進度", "教材・機材", "先生の説明・対応", "友人関係・雰囲気", "設備・環境", "進路・評価", "その他"] },
      { type: "noul", id: "suggestion", instructions: "具体的な改善提案を含むか" },
      { type: "noul", id: "followup", instructions: "個別に声をかけるべき内容（困りごと・不安・体調）を含むか" }],
    routes: [
      { name: "🔴 個別対応", cls: "bad", when: (A, g) => A.followup?.noul && A.followup.confidence >= g },
      { name: "🟠 改善提案", cls: "warn", when: A => A.suggestion?.noul },
      { name: "🟡 人が確認", cls: "warn", when: (A, g) => low(A, ["topic", "followup"], g) },
      { name: "🟢 集計のみ", cls: "ok", when: () => true }],
    columns: ["sentiment", "topic", "suggestion", "followup"], sort: "sentiment",
    examples: ["はんだ付けが難しかったけど、先生が個別に見てくれたのでできた。もう少し練習時間がほしい。", "パソコンが古くて動画編集が固まる。授業中ずっと待っていた。", "最近ずっと眠れなくて授業に集中できない。"] },
  { id: "news", name: "ニュース・記事の整理", icon: "📰", origin: "TypeSafe hierarchical classification / citation check", desc: "集めた記事を、テーマ・一次情報の有無・論調・信頼できる根拠の有無で整理する。",
    questions: [
      { type: "choice", id: "topic", instructions: "記事の主なテーマ", options: ["政治・行政", "経済・企業", "災害・防災", "事件・事故", "科学・技術", "教育", "生活・健康", "その他"] },
      { type: "noul", id: "primary", instructions: "公的機関や当事者の一次情報（発表・文書・会見）に基づいているか" },
      { type: "choice", id: "stance", instructions: "記事の論調", options: ["中立・事実報道", "肯定的", "批判的", "不明"] },
      { type: "noul", id: "named_source", instructions: "情報源が実名・組織名で明示されているか" }],
    routes: [
      { name: "🟢 一次情報あり", cls: "ok", when: A => A.primary?.noul && A.named_source?.noul },
      { name: "🟡 出典を確認", cls: "warn", when: (A, g) => !A.named_source?.noul || low(A, ["primary"], g) },
      { name: "⚪ 意見・解説", cls: "muted", when: () => true }],
    columns: ["topic", "primary", "stance", "named_source"], sort: "",
    examples: ["総務省は 26 日、生成 AI による偽情報対策の検討会報告書案を公表した。事業者にラベル付けを求める内容。", "AI がすべての仕事を奪うという説には根拠がない、と私は考える。"] },
  { id: "review", name: "レビュー・口コミの整理", icon: "⭐", origin: "TypeSafe use-case map（e-commerce listings / fake reviews）", desc: "口コミを、評価・観点・サクラの疑い・対応が必要な指摘で整理する。",
    questions: [
      { type: "score", id: "rating", instructions: "この口コミの評価", levels: ["1", "2", "3", "4", "5"] },
      { type: "choice", id: "aspect", instructions: "主に言及している観点", options: ["価格", "品質・性能", "対応・接客", "配送・納期", "使いやすさ", "その他"] },
      { type: "noul", id: "fake", instructions: "宣伝・サクラ・定型文の疑いがあるか（具体性がなく称賛のみ、不自然な言い回し）" },
      { type: "noul", id: "actionable", instructions: "運営側が対応すべき具体的な指摘を含むか" }],
    routes: [
      { name: "🔴 対応が必要", cls: "bad", when: A => A.actionable?.noul && A.rating?.score <= 2 },
      { name: "⚪ サクラの疑い", cls: "muted", when: (A, g) => A.fake?.noul && A.fake.confidence >= g },
      { name: "🟡 人が確認", cls: "warn", when: (A, g) => low(A, ["fake", "aspect"], g) },
      { name: "🟢 集計", cls: "ok", when: () => true }],
    columns: ["rating", "aspect", "fake", "actionable"], sort: "rating",
    examples: ["最高です！！買ってよかった！おすすめ！", "2 回目の使用で電源が入らなくなった。サポートに連絡したが 1 週間返事がない。", "値段の割に作りがしっかりしている。届くのも早かった。"] },
  { id: "router", name: "相談の振り分け（学校・窓口）", icon: "🧭", origin: "TypeSafe intent routing / Laya router preset", desc: "生徒・保護者・利用者からの相談を、分野・難しさ・専門家の要否・配慮の要否で振り分ける。",
    questions: [
      { type: "choice", id: "domain", instructions: "相談の分野", options: ["学習・授業", "進路・受験", "生活・健康", "人間関係・いじめ", "事務手続き・費用", "その他"] },
      { type: "score", id: "difficulty", instructions: "対応の難しさ（定型で答えられる 0 〜 専門的な判断が必要 5）", levels: L5 },
      { type: "noul", id: "needs_human", instructions: "担任・カウンセラー・専門家など人が直接対応すべきか" },
      { type: "noul", id: "sensitive", instructions: "配慮が必要な内容（心身の不調・家庭の事情・被害）を含むか" }],
    routes: [
      { name: "🔴 専門家へ", cls: "bad", when: (A, g) => (A.sensitive?.noul && A.sensitive.confidence >= g) || A.domain?.choice === "人間関係・いじめ" },
      { name: "🟠 担当者が対応", cls: "warn", when: A => A.needs_human?.noul || A.difficulty?.score >= 3 },
      { name: "🟡 人が確認", cls: "warn", when: (A, g) => low(A, ["domain", "sensitive"], g) },
      { name: "🟢 定型案内", cls: "ok", when: () => true }],
    columns: ["domain", "difficulty", "needs_human", "sensitive"], sort: "difficulty",
    examples: ["来年度の教科書代はいつまでに払えばいいですか。", "クラスの何人かに無視されていて学校に行きたくない。誰にも言わないでほしい。", "推薦入試の志望理由書を見てもらいたいです。"] },
];
export const presetOf = id => TRIAGE_PRESETS.find(p => p.id === id);
/* 経路を決める：ルールを上から評価 */
export function route(preset, answers, gateTh = 0.7) { const A = Object.fromEntries(answers.map(a => [a.id, a])); for (const r of preset.routes) { try { if (r.when(A, gateTh)) return r; } catch { } } return preset.routes[preset.routes.length - 1]; }

/* 「見極める」の標準質問：SNS 検証 ＋ 詐欺・誘導（fan-out で 1 回に問う） */
export const VERIFY_QUESTIONS = ctxLen => [
  ...presetOf("sns").questions.filter(q => !q.needsContext || ctxLen > 0),
  { type: "noul", id: "asks_personal", instructions: "個人情報・認証情報・送金などを要求しているか" },
  { type: "noul", id: "impersonation", instructions: "公的機関・企業・知人になりすましている疑いがあるか" },
];
/* 主張と根拠の照合（citation check）：根拠 1 つに対して主張がどう関係するか */
export const CITATION_Q = { type: "choice", id: "relation", instructions: "根拠（context）はこの主張に対してどういう関係か", options: ["支持している", "矛盾している", "関係ない・判断できない"] };
/* 同一性の判定（entity alignment） */
export const SAME_QS = [
  { type: "noul", id: "same", instructions: "A と B は同じ対象（同じ人物・組織・出来事・場所）を指しているか" },
  { type: "score", id: "similarity", instructions: "A と B の内容の一致の度合い", levels: ["別物", "一部が一致", "ほぼ一致", "同一"] },
];
/* 「聞く」のよく使う質問（文書に問う） */
export const ASK_QUICK = [
  { label: "要点はどこ？", text: "どの行: この文書の最も重要な主張" },
  { label: "日時・場所は？", text: "どの行: 日時や場所が書かれている" },
  { label: "根拠は？", text: "どの行: 主張の根拠（出典・数値・引用）" },
  { label: "信頼できる？", text: "? 公的機関や当事者の一次情報に基づいているか" },
  { label: "拡散依頼？", text: "? 拡散や転送を呼びかけているか" },
  { label: "緊急度", text: "度合い: 今すぐ対応が必要な度合い" },
  { label: "種類", text: "この情報の種類 | 公的機関の発表 | 報道 | 一般の投稿・目撃 | 意見・感想 | 宣伝・無関係" },
  { label: "返信要？", text: "? 返信が必要か" },
];
/* 既存 Jev の使い方例（ホームの「できること」） */
export const USE_CASES = [
  { icon: "🔎", name: "調べる", what: "話題を公開 API から集め、1 件ずつ JEV が種類・具体性・拡散依頼を判定 → 信頼性順に並べ、読む件数を減らす", jev: "RAG passage classification / re-ranking（TypeSafe cookbook）" },
  { icon: "⚖️", name: "見極める", what: "投稿・メール・主張を、10 の問いに 1 回で答えて信頼性と危険度を出す。根拠と照合し、確認すべきことを列挙", jev: "speculative fan-out ＋ guardrails ＋ citation check" },
  { icon: "🗂", name: "仕分ける", what: "問い合わせ・メール・投稿・アンケートを何十件でも同じ問いで仕分け、確信度で「自動／人が確認」に振り分け", jev: "intent routing ＋ confidence-gated routing（Laya presets）" },
  { icon: "💬", name: "聞く", what: "文書を置いて、はい／いいえ・選択・段階・「どの行に書いてある？」で問う。生成しないので、答えは常に確率つき", jev: "line-by-line semantic find（TypeSafe cookbook）/ Jevlet" },
];
