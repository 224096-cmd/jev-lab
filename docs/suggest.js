/* 検索の補完・検索プラン（キーワードを入れたら、何をどう探すかを自動で組み立てる）
   - suggest(q, history): 入力途中の候補（履歴・プリセット・Wikipedia の見出し候補）と「検索プラン」（話題の種類に合わせた dork 式・情報源）
   - plan(q): 種類（災害／噂・詐欺／技術・研究／組織・サイト／一般）を推定し、検索式のチップと推奨ソースを返す
   - dashboard(items, judged, trust): 収集結果の可視化（SVG）：種類の内訳・信頼性の分布・出典別件数・時系列 */
import { apiFetch } from "./quota.js";
import { buildFromLibrary, DORK_LIBRARY } from "./tools.js";
const enc = encodeURIComponent;
const DISASTER = /(大雨|豪雨|地震|津波|洪水|氾濫|冠水|浸水|土砂|火災|台風|警報|避難|停電|断水|噴火|大雪|竜巻|高潮|震度)/;
const RUMOR = /(デマ|噂|詐欺|偽|なりすまし|拡散|炎上|真偽|本当|嘘|フェイク|注意喚起)/;
const TECH = /^[\x20-\x7e]+$|(モデル|AI|LLM|ONNX|WebGPU|機械学習|論文|API|GitHub|Jev|typed)/i;
const PLACE = /(市|町|村|区|県|府|都|駅|川|山|港|島|湖|橋|空港|学校|病院)$/;
const ORG = /(株式会社|会社|大学|省|庁|市役所|県庁|銀行|協会|法人|\.(jp|com|net|org|go\.jp|lg\.jp))/;
export function kindOf(q) { if (DISASTER.test(q)) return "disaster"; if (RUMOR.test(q)) return "rumor"; if (ORG.test(q)) return "org"; if (TECH.test(q)) return "tech"; return "general"; }
export const KIND_LABEL = { disaster: "災害・地域", rumor: "噂・詐欺の検証", org: "組織・サイト", tech: "技術・研究", general: "一般の話題" };
const week = () => new Date(Date.now() - 7 * 864e5).toISOString().slice(0, 10);
/* 検索プラン：話題の種類に合わせて「集める情報源」「検索エンジンで開く式」「JEV に問う質問の重点」を組み立てる */
export function plan(q) {
  const k = kindOf(q); const place = q.replace(DISASTER, " ").replace(/\s+/g, " ").trim(); const P = { kind: k, label: KIND_LABEL[k], sources: [], dorks: [], focus: "" };
  if (k === "disaster") { P.sources = ["jma", "nominatim", "bluesky", "mastodon", "gdelt", "wikipedia"]; P.focus = "公的発表（気象庁・自治体）と一般投稿を分け、地理照合（標高・座標）と拡散依頼の有無で信頼性を判定"; P.dorks = [{ name: "自治体・気象庁の発表", q: `"${place}" (${q.replace(place, "").trim() || "災害"}) (site:go.jp OR site:lg.jp)` }, { name: "報道（直近 1 週間）", q: `"${q}" after:${week()} (site:nhk.or.jp OR site:asahi.com OR site:mainichi.jp OR site:yomiuri.co.jp)` }, { name: "SNS の投稿", q: `"${q}" (site:x.com OR site:bsky.app)` }, { name: "避難所・ハザード PDF", q: `"${place}" (避難所 OR ハザードマップ) filetype:pdf` }, { name: "動画", q: `"${q}" site:youtube.com` }]; }
  else if (k === "rumor") { P.sources = ["wikipedia", "wikidata", "bluesky", "mastodon", "gdelt", "hn"]; P.focus = "一次情報（公的発表・当事者）と拡散投稿を分け、根拠との整合・具体性・拡散依頼で判定。矛盾の疑いを要確認へ"; P.dorks = [{ name: "公的な否定・注意喚起", q: `"${q.replace(RUMOR, "").trim() || q}" (注意喚起 OR 事実無根 OR 否定) (site:go.jp OR site:lg.jp OR site:or.jp)` }, { name: "ファクトチェック", q: `"${q.replace(RUMOR, "").trim() || q}" (site:factcheckcenter.jp OR site:infact.press OR ファクトチェック)` }, { name: "初出を探す（古い順）", q: `"${q.replace(RUMOR, "").trim() || q}" before:${week()}` }, { name: "同じ画像・文面の使い回し", q: `"${q.replace(RUMOR, "").trim() || q}" (拡散希望 OR RT希望 OR 拡散お願い)` }]; }
  else if (k === "org") { P.sources = ["wikipedia", "wikidata", "gdelt", "bluesky", "github"]; P.focus = "公式サイト・登記・報道と、口コミ・投稿を分ける。ドメインは「調査」タブで素性を確認"; P.dorks = [{ name: "公式サイト以外の言及", q: `"${q}" -site:${q.replace(/^https?:\/\//, "").split("/")[0]}` }, { name: "法人情報・登記", q: `"${q}" (site:houjin-bangou.nta.go.jp OR site:info.gbiz.go.jp)` }, { name: "被害・注意喚起", q: `"${q}" (詐欺 OR 注意 OR 被害 OR 評判)` }, { name: "採用・IR・PDF", q: `"${q}" filetype:pdf` }]; }
  else if (k === "tech") { P.sources = ["hn", "github", "arxiv", "crossref", "semanticscholar", "stackexchange", "bluesky"]; P.focus = "一次情報（論文・コード・公式発表）と紹介記事・意見を分け、関連度で仕分け"; P.dorks = [{ name: "論文", q: `"${q}" (site:arxiv.org OR site:aclanthology.org OR site:openreview.net)` }, { name: "コード", q: `"${q}" (site:github.com OR site:huggingface.co)` }, { name: "技術記事（日本語）", q: `"${q}" (site:qiita.com OR site:zenn.dev OR site:note.com)` }, { name: "公式・ドキュメント", q: `"${q}" (docs OR documentation OR "release notes")` }]; }
  else { P.sources = ["wikipedia", "gdelt", "bluesky", "mastodon", "hn"]; P.focus = "公的発表・報道・投稿・意見を分けて信頼性順に"; P.dorks = [{ name: "報道", q: `"${q}" (site:nhk.or.jp OR site:asahi.com OR site:nikkei.com OR site:reuters.com)` }, { name: "公的発表", q: `"${q}" (site:go.jp OR site:lg.jp)` }, { name: "直近 1 週間", q: `"${q}" after:${week()}` }, { name: "SNS", q: `"${q}" (site:x.com OR site:bsky.app OR site:reddit.com)` }, { name: "PDF 資料", q: `"${q}" filetype:pdf` }]; }
  P.dorks.forEach(d => d.url = `https://www.google.com/search?q=${enc(d.q)}`); return P;
}
/* 入力途中の候補 */
export async function suggest(q, history = [], presets = []) {
  const s = q.trim(); const out = [];
  if (!s) { history.slice(0, 5).forEach(h => out.push({ kind: "履歴", text: h })); presets.slice(0, 4).forEach(p => out.push({ kind: "例", text: p.query, preset: p })); return out; }
  history.filter(h => h.includes(s) && h !== s).slice(0, 3).forEach(h => out.push({ kind: "履歴", text: h }));
  presets.filter(p => p.query.includes(s) || p.name.includes(s)).slice(0, 2).forEach(p => out.push({ kind: "例", text: p.query, preset: p }));
  if (s.length >= 2 && !/[|:"()]/.test(s)) { try { const lang = /^[\x20-\x7e]+$/.test(s) ? "en" : "ja"; const r = await apiFetch("wikipedia", `https://${lang}.wikipedia.org/w/api.php?action=opensearch&search=${enc(s)}&limit=5&namespace=0&format=json&origin=*`, { ttl: 86400 }); (r[1] || []).forEach((t, i) => { if (t !== s) out.push({ kind: "Wikipedia", text: t, desc: (r[2] || [])[i] || "" }); }); } catch { } }
  return out.slice(0, 10);
}
/* 調査タブの「次に調べる」（対象の種類ごと） */
export function nextSteps(R) { const v = R.target.value, d = v.replace(/^https?:\/\//, "").split("/")[0]; const L = DORK_LIBRARY; const pick = names => L.filter(x => names.includes(x.name)).map(x => ({ name: x.name, q: buildFromLibrary(x, d, d), url: `https://www.google.com/search?q=${enc(buildFromLibrary(x, d, d))}` }));
  if (["domain", "url", "email"].includes(R.type)) return [...pick(["同名サイトの重複", "被害・注意喚起の投稿", "サブドメイン（www 以外）", "ログイン画面"]), { name: "この話題を収集", collect: d }];
  if (["coords", "place"].includes(R.type)) return [{ name: "この地名で収集（災害・出来事）", collect: v }, { name: "自治体の PDF", url: `https://www.google.com/search?q=${enc(`"${v}" filetype:pdf (site:lg.jp OR site:go.jp)`)}` }];
  if (R.type === "company") return [...pick(["法人情報・登記", "被害・注意喚起の投稿", "プレスリリース"].filter(n => L.some(x => x.name === n))), { name: "この名前で収集", collect: v }];
  if (["username", "hashtag"].includes(R.type)) return [{ name: "投稿を収集（Bluesky・Mastodon）", collect: (R.type === "hashtag" ? "#" : "") + v }];
  return [{ name: "この語で収集", collect: v }];
}
/* ---------- 可視化（SVG） ---------- */
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
export function bars(pairs, { color = "var(--acc)", width = 320, max } = {}) { const mx = max || Math.max(1, ...pairs.map(p => p[1])); const h = pairs.length * 22 + 4; return `<svg viewBox="0 0 ${width} ${h}" width="100%" style="max-width:${width}px;display:block" role="img">${pairs.map(([k, v, c], i) => `<text x="0" y="${i * 22 + 15}" font-size="12" fill="currentColor">${esc(String(k).slice(0, 12))}</text><rect x="110" y="${i * 22 + 4}" width="${Math.max(2, (width - 150) * v / mx)}" height="14" rx="3" fill="${c || color}"/><text x="${115 + (width - 150) * v / mx}" y="${i * 22 + 15}" font-size="12" fill="currentColor">${v}</text>`).join("")}</svg>`; }
export function dashboard(items, judged, trust) { const ok = items.map((it, i) => i).filter(i => !items[i].error); if (!ok.length) return "";
  const kinds = {}; ok.forEach(i => { const k = judged?.[i]?.kind?.choice || "未判定"; kinds[k] = (kinds[k] || 0) + 1; }); const KC = { "公的機関の発表": "#15803d", "報道": "#0b6e99", "一般の投稿・目撃": "#b45309", "意見・感想": "#7c3aed", "宣伝・無関係": "#6b7280", "未判定": "#9aa3ad" };
  const bins = [0, 0, 0, 0, 0]; if (trust) ok.forEach(i => { const t = trust[i]?.score; if (t != null) bins[Math.min(4, Math.floor(t * 5))]++; });
  const srcs = {}; ok.forEach(i => { srcs[items[i].source] = (srcs[items[i].source] || 0) + 1; });
  return `<div class="dash"><div><b class="small">種類の内訳</b>${bars(Object.entries(kinds).sort((a, b) => b[1] - a[1]).map(([k, v]) => [k, v, KC[k]]))}</div>${trust ? `<div><b class="small">信頼性の分布</b>${bars(["0–20", "20–40", "40–60", "60–80", "80–100"].map((l, i) => [l, bins[i], i >= 3 ? "#15803d" : i === 2 ? "#b45309" : "#be123c"]))}</div>` : ""}<div><b class="small">出典別</b>${bars(Object.entries(srcs).sort((a, b) => b[1] - a[1]).slice(0, 6))}</div></div>`; }
