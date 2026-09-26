/* 調査ツール（動画で紹介される OSINT 手法のうち、公開・受動的・合法な範囲をブラウザで）
   - 検索演算子（Google dorks）: 検索 URL を組み立てて開くだけ。自動巡回はしない
   - Wayback Machine: archive.org の公開 API（CORS 可）
   - DNS: Google Public DNS の DoH JSON（CORS 可）
   - WHOIS/RDAP: rdap.org（CORS 可。.jp など未対応 TLD はリンクで案内）
   - サブドメイン: crt.sh の証明書透明性ログ（公開・受動的）
   - EXIF/GPS: 自分の端末で選んだ画像のメタデータを端末内で読む（送信しない）
   - 逆画像検索: 各サービスの検索ページを開くリンク
   実装しないもの: Shodan・HIBP など API キーが要るもの、ユーザー名の横断照会など個人を対象にするもの、
   robots.txt を無視した収集、ログインが必要なサービスの自動操作 */

const sleep = ms => new Promise(r => setTimeout(r, ms));
const enc = encodeURIComponent;

/* ---------- 検索演算子（dork）ビルダー ---------- */
export function buildDorks({ q = "", site = "", filetype = "", intitle = "", inurl = "", exclude = "", after = "", before = "", exact = false }) {
  const parts = [];
  if (q) parts.push(exact ? `"${q}"` : q);
  if (site) parts.push(`site:${site}`);
  if (filetype) parts.push(`filetype:${filetype}`);
  if (intitle) parts.push(`intitle:"${intitle}"`);
  if (inurl) parts.push(`inurl:${inurl}`);
  if (exclude) exclude.split(",").map(s => s.trim()).filter(Boolean).forEach(x => parts.push(`-${x}`));
  if (after) parts.push(`after:${after}`);
  if (before) parts.push(`before:${before}`);
  const query = parts.join(" ");
  return { query, links: [
    { name: "Google", url: `https://www.google.com/search?q=${enc(query)}` },
    { name: "Bing", url: `https://www.bing.com/search?q=${enc(query)}` },
    { name: "DuckDuckGo", url: `https://duckduckgo.com/?q=${enc(query)}` },
    { name: "Yahoo! JAPAN", url: `https://search.yahoo.co.jp/search?p=${enc(query)}` },
    { name: "Google ニュース", url: `https://news.google.com/search?q=${enc(query)}&hl=ja&gl=JP&ceid=JP:ja` },
    { name: "X（Twitter）検索", url: `https://x.com/search?q=${enc(query)}&f=live` },
    { name: "YouTube", url: `https://www.youtube.com/results?search_query=${enc(query)}` },
    { name: "Wikipedia", url: `https://ja.wikipedia.org/w/index.php?search=${enc(q)}` },
    { name: "Wayback（サイト）", url: site ? `https://web.archive.org/web/*/${site}*` : "" },
  ].filter(l => l.url) };
}
export const DORK_PRESETS = [
  { name: "自治体の公開 PDF", fill: { filetype: "pdf", site: "lg.jp" } },
  { name: "官公庁（go.jp）", fill: { site: "go.jp" } },
  { name: "公開ディレクトリ一覧（自サイト点検用）", fill: { intitle: "index of", exact: false } },
  { name: "報道機関（例: nhk.or.jp）", fill: { site: "nhk.or.jp" } },
  { name: "Excel / CSV の公開データ", fill: { filetype: "xlsx" } },
  { name: "期間指定（直近 1 年）", fill: { after: new Date(Date.now() - 365 * 864e5).toISOString().slice(0, 10) } },
];

/* ---------- Wayback Machine ---------- */
export async function wayback(url) {
  const years = [2010, 2015, 2018, 2020, 2022, 2024, 2026]; const out = [];
  for (const y of years) {
    try { const r = await (await fetch(`https://archive.org/wayback/available?url=${enc(url)}&timestamp=${y}0101`)).json(); const c = r.archived_snapshots?.closest; if (c?.available) out.push({ year: y, timestamp: c.timestamp, url: c.url }); } catch { }
    await sleep(250);
  }
  const uniq = []; const seen = new Set(); for (const s of out) if (!seen.has(s.timestamp)) { seen.add(s.timestamp); uniq.push(s); }
  return { snapshots: uniq, first: uniq.length ? uniq.reduce((a, b) => a.timestamp < b.timestamp ? a : b) : null, browse: `https://web.archive.org/web/*/${url}` };
}

/* ---------- DNS（DoH） ---------- */
export async function dns(domain, types = ["A", "AAAA", "MX", "NS", "TXT", "CNAME"]) {
  const out = {};
  for (const t of types) { try { const r = await (await fetch(`https://dns.google/resolve?name=${enc(domain)}&type=${t}`)).json(); out[t] = (r.Answer || []).map(a => a.data); } catch (e) { out[t] = ["error"]; } await sleep(120); }
  return out;
}

/* ---------- RDAP（WHOIS 後継） ---------- */
export async function rdap(domain) {
  try { const r = await fetch(`https://rdap.org/domain/${enc(domain)}`); if (!r.ok) return { error: `RDAP ${r.status}（.jp は未対応。JPRS WHOIS: https://whois.jprs.jp/ ）`, whois_link: "https://whois.jprs.jp/" }; const j = await r.json();
    const ev = Object.fromEntries((j.events || []).map(e => [e.eventAction, e.eventDate])); const reg = (j.entities || []).find(e => (e.roles || []).includes("registrar")); const name = reg?.vcardArray?.[1]?.find(x => x[0] === "fn")?.[3];
    return { handle: j.ldhName || domain, registered: ev.registration, expires: ev.expiration, updated: ev["last changed"], registrar: name, status: j.status, nameservers: (j.nameservers || []).map(n => n.ldhName), age_days: ev.registration ? Math.round((Date.now() - new Date(ev.registration)) / 864e5) : null }; }
  catch (e) { return { error: String(e) }; }
}

/* ---------- 証明書透明性ログ（サブドメイン） ---------- */
export async function crtsh(domain) {
  try { const r = await (await fetch(`https://crt.sh/?q=${enc("%." + domain)}&output=json`)).json(); const names = new Set(); for (const c of r) for (const n of String(c.name_value).split("\n")) if (!n.startsWith("*")) names.add(n.trim()); return { count: names.size, names: [...names].sort().slice(0, 200), issuers: [...new Set(r.map(c => c.issuer_name))].slice(0, 5) }; }
  catch (e) { return { error: String(e) }; }
}

/* ---------- ドメインの信頼性チェック（DNS + RDAP + Wayback + CT を 1 つの根拠にまとめる） ---------- */
export async function domainProfile(domain, onProgress = () => {}) {
  onProgress("DNS…"); const d = await dns(domain, ["A", "MX", "NS"]);
  onProgress("RDAP…"); const w = await rdap(domain);
  onProgress("Wayback…"); const wb = await wayback(domain);
  onProgress("証明書ログ…"); const ct = await crtsh(domain);
  onProgress("");
  const lines = [];
  lines.push(`${domain}: DNS A=${(d.A || []).length} 件, MX=${(d.MX || []).length} 件, NS=${(d.NS || []).join(",") || "なし"}`);
  if (w.registered) lines.push(`${domain}: 登録 ${w.registered.slice(0, 10)}（約${Math.round((w.age_days || 0) / 365)}年前）、レジストラ ${w.registrar || "?"}`); else if (w.error) lines.push(`${domain}: RDAP 情報なし（${w.error.slice(0, 40)}）`);
  if (wb.first) lines.push(`${domain}: Wayback 最古のスナップショット ${wb.first.timestamp.slice(0, 4)} 年、${wb.snapshots.length} 時点`); else lines.push(`${domain}: Wayback にスナップショットなし（新しい、または非公開）`);
  if (!ct.error) lines.push(`${domain}: 証明書ログ上のホスト名 ${ct.count} 件`);
  return { domain, dns: d, rdap: w, wayback: wb, crt: ct, context: lines };
}

/* ---------- EXIF / GPS（端末内で読む。送信しない） ---------- */
export async function readExif(file) {
  const buf = new DataView(await file.arrayBuffer()); const out = { file: file.name, size: file.size, type: file.type };
  if (buf.getUint16(0) !== 0xFFD8) { out.note = "JPEG ではありません（PNG/HEIC/WebP の EXIF は未対応）"; return out; }
  let off = 2;
  while (off < buf.byteLength - 4) {
    const marker = buf.getUint16(off); const len = buf.getUint16(off + 2); if (marker === 0xFFE1 && buf.getUint32(off + 4) === 0x45786966) { parseTiff(buf, off + 10, out); break; } if ((marker & 0xFF00) !== 0xFF00) break; off += 2 + len;
  }
  if (out.lat != null) out.map = `https://www.openstreetmap.org/?mlat=${out.lat}&mlon=${out.lon}#map=16/${out.lat}/${out.lon}`;
  if (!("Make" in out) && out.lat == null) out.note = "EXIF なし（SNS 経由の画像は通常メタデータが削除されている）";
  return out;
}
function parseTiff(v, start, out) {
  const le = v.getUint16(start) === 0x4949; const u16 = o => v.getUint16(o, le), u32 = o => v.getUint32(o, le);
  const ifd0 = start + u32(start + 4); const TAGS = { 0x10F: "Make", 0x110: "Model", 0x132: "DateTime", 0x131: "Software", 0x9003: "DateTimeOriginal", 0xA405: "FocalLength35mm" };
  const readIfd = (p, cb) => { if (p <= start || p >= v.byteLength - 2) return; const n = u16(p); for (let i = 0; i < n; i++) { const e = p + 2 + i * 12; if (e + 12 > v.byteLength) break; cb(u16(e), u16(e + 2), u32(e + 4), e + 8); } };
  const str = (cnt, vo) => { const o = cnt > 4 ? start + u32(vo) : vo; let s = ""; for (let i = 0; i < cnt - 1 && o + i < v.byteLength; i++) s += String.fromCharCode(v.getUint8(o + i)); return s; };
  const rat = (cnt, vo) => { const o = start + u32(vo); const a = []; for (let i = 0; i < cnt; i++) a.push(u32(o + i * 8) / (u32(o + i * 8 + 4) || 1)); return a; };
  let exifP = 0, gpsP = 0;
  readIfd(ifd0, (tag, type, cnt, vo) => { if (TAGS[tag] && type === 2) out[TAGS[tag]] = str(cnt, vo); if (tag === 0x8769) exifP = start + u32(vo); if (tag === 0x8825) gpsP = start + u32(vo); });
  if (exifP) readIfd(exifP, (tag, type, cnt, vo) => { if (TAGS[tag] && type === 2) out[TAGS[tag]] = str(cnt, vo); });
  if (gpsP) { const g = {}; readIfd(gpsP, (tag, type, cnt, vo) => { if (tag === 1 || tag === 3) g[tag] = String.fromCharCode(v.getUint8(vo)); if ((tag === 2 || tag === 4) && type === 5) g[tag] = rat(cnt, vo); });
    const dms = a => a ? a[0] + a[1] / 60 + a[2] / 3600 : null; if (g[2] && g[4]) { out.lat = +(dms(g[2]) * (g[1] === "S" ? -1 : 1)).toFixed(6); out.lon = +(dms(g[4]) * (g[3] === "W" ? -1 : 1)).toFixed(6); } }
}

/* ---------- 逆画像検索（画像 URL を各サービスで開く） ---------- */
export const reverseImageLinks = url => [
  { name: "Google Lens", url: `https://lens.google.com/uploadbyurl?url=${enc(url)}` },
  { name: "Bing Visual Search", url: `https://www.bing.com/images/search?view=detailv2&iss=sbi&q=imgurl:${enc(url)}` },
  { name: "Yandex", url: `https://yandex.com/images/search?rpt=imageview&url=${enc(url)}` },
  { name: "TinEye", url: `https://tineye.com/search?url=${enc(url)}` },
];

/* ---------- URL からドメインを取り出す ---------- */
export const domainOf = u => { try { return new URL(u.includes("://") ? u : "https://" + u).hostname.replace(/^www\./, ""); } catch { return u.trim(); } };

/* ================= v0.7 追加：端末内ツール（CyberChef / FotoForensics / FOCA / EFF Cover Your Tracks 相当の一部） ================= */
const b64enc = s => btoa(String.fromCharCode(...new TextEncoder().encode(s))), b64dec = s => new TextDecoder().decode(Uint8Array.from(atob(s.replace(/-/g, "+").replace(/_/g, "/")), c => c.charCodeAt(0)));
export const CODEC = {
  "Base64 エンコード": s => b64enc(s), "Base64 デコード": s => b64dec(s.trim()),
  "Hex エンコード": s => [...new TextEncoder().encode(s)].map(b => b.toString(16).padStart(2, "0")).join(" "), "Hex デコード": s => new TextDecoder().decode(Uint8Array.from(s.replace(/[^0-9a-f]/gi, "").match(/../g) || [], h => parseInt(h, 16))),
  "URL エンコード": s => encodeURIComponent(s), "URL デコード": s => decodeURIComponent(s.replace(/\+/g, " ")),
  "HTML エンティティ → 文字": s => { const t = document.createElement("textarea"); t.innerHTML = s; return t.value; }, "文字 → HTML エンティティ": s => s.replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])),
  "ROT13": s => s.replace(/[a-z]/gi, c => { const b = c <= "Z" ? 65 : 97; return String.fromCharCode(b + (c.charCodeAt(0) - b + 13) % 26); }),
  "UNIX 時刻 → 日時": s => { const n = +s.trim(); if (isNaN(n)) return "数字ではない"; const d = new Date(n > 1e12 ? n : n * 1000); return `${d.toISOString()} (UTC) / ${d.toLocaleString("ja-JP")} (端末)`; }, "日時 → UNIX 時刻": s => { const d = new Date(s); return isNaN(d) ? "日時として読めない" : String(Math.floor(d / 1000)); },
  "Unicode エスケープ → 文字": s => s.replace(/\\u([0-9a-f]{4})/gi, (_, h) => String.fromCharCode(parseInt(h, 16))), "全角 → 半角（英数）": s => s.replace(/[Ａ-Ｚａ-ｚ０-９]/g, c => String.fromCharCode(c.charCodeAt(0) - 0xFEE0)),
  "JWT を読む（署名は検証しない）": s => { const p = s.trim().split("."); if (p.length < 2) return "JWT ではない"; return JSON.stringify({ header: JSON.parse(b64dec(p[0])), payload: JSON.parse(b64dec(p[1])) }, null, 1); },
  "短縮 URL の展開（リンクを開く）": s => `https://unshorten.it/?u=${encodeURIComponent(s.trim())}`,
};
export function identifyString(s) { const t = s.trim(); const out = []; if (/^[0-9a-f]{32}$/i.test(t)) out.push("MD5 / NTLM の可能性（32 hex）"); if (/^[0-9a-f]{40}$/i.test(t)) out.push("SHA-1（40 hex）"); if (/^[0-9a-f]{64}$/i.test(t)) out.push("SHA-256（64 hex）"); if (/^\$2[aby]\$/.test(t)) out.push("bcrypt"); if (/^[A-Za-z0-9+/=]{16,}$/.test(t) && t.length % 4 === 0) out.push("Base64 らしい"); if (/^ey[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]*$/.test(t)) out.push("JWT"); if (/^\d{10}$|^\d{13}$/.test(t)) out.push("UNIX 時刻（秒 / ミリ秒）の可能性"); if (/^(https?:\/\/)?(bit\.ly|t\.co|tinyurl\.com|goo\.gl|is\.gd|ow\.ly|buff\.ly|x\.gd|00m\.in)\//i.test(t)) out.push("短縮 URL（展開してから開く）"); if (/^xn--/.test(t) || /\.xn--/.test(t)) out.push("Punycode（国際化ドメイン）"); return out.length ? out : ["特徴なし（プレーンテキスト）"]; }
export async function hashText(s, algo = "SHA-256") { const b = await crypto.subtle.digest(algo, new TextEncoder().encode(s)); return [...new Uint8Array(b)].map(x => x.toString(16).padStart(2, "0")).join(""); }
export async function hashFile(file, algo = "SHA-256") { const b = await crypto.subtle.digest(algo, await file.arrayBuffer()); return [...new Uint8Array(b)].map(x => x.toString(16).padStart(2, "0")).join(""); }

/* ---------- ELA（Error Level Analysis）: JPEG を品質 95% で再圧縮し差分を強調。編集された領域は差分が大きく出やすい（FotoForensics と同じ原理。断定はできない） ---------- */
export async function ela(file, quality = 0.95, scale = 20) {
  const img = await new Promise((ok, ng) => { const i = new Image(); i.onload = () => ok(i); i.onerror = ng; i.src = URL.createObjectURL(file); });
  const W = Math.min(img.width, 1400), H = Math.round(img.height * W / img.width); const c = document.createElement("canvas"); c.width = W; c.height = H; const g = c.getContext("2d"); g.drawImage(img, 0, 0, W, H); const a = g.getImageData(0, 0, W, H);
  const blob = await new Promise(r => c.toBlob(r, "image/jpeg", quality)); const img2 = await new Promise((ok, ng) => { const i = new Image(); i.onload = () => ok(i); i.onerror = ng; i.src = URL.createObjectURL(blob); }); g.drawImage(img2, 0, 0, W, H); const b = g.getImageData(0, 0, W, H);
  const out = g.createImageData(W, H); let mx = 0; for (let i = 0; i < a.data.length; i += 4) { for (let k = 0; k < 3; k++) { const d = Math.min(255, Math.abs(a.data[i + k] - b.data[i + k]) * scale); out.data[i + k] = d; if (d > mx) mx = d; } out.data[i + 3] = 255; } g.putImageData(out, 0, 0);
  return { canvas: c, width: W, height: H, max: mx, note: "明るい領域＝再圧縮で変化が大きい部分。一様に暗ければ一度きりの保存、局所的に明るければその部分だけ編集・貼り付けの可能性（縮小・再投稿でも変わるので断定はしない）" };
}

/* ---------- 文書メタデータ（PDF の /Info と XMP、Office の docProps/core.xml）。端末内、送信しない ---------- */
export async function docMeta(file) {
  const buf = new Uint8Array(await file.arrayBuffer()); const out = { file: file.name, size: file.size, type: file.type }; const head = new TextDecoder("latin1").decode(buf.subarray(0, 8));
  if (head.startsWith("%PDF")) { const txt = new TextDecoder("latin1").decode(buf); for (const k of ["Title", "Author", "Subject", "Keywords", "Creator", "Producer", "CreationDate", "ModDate"]) { const m = new RegExp("/" + k + "\\s*\\((.*?)(?<!\\\\)\\)").exec(txt); if (m) out[k] = m[1].slice(0, 120); } const xmp = /<x:xmpmeta[\s\S]*?<\/x:xmpmeta>/.exec(txt); if (xmp) { const x = xmp[0]; for (const [k, re] of [["xmp:CreatorTool", /<xmp:CreatorTool>(.*?)</], ["dc:creator", /<dc:creator>[\s\S]*?<rdf:li[^>]*>(.*?)</], ["xmp:CreateDate", /<xmp:CreateDate>(.*?)</], ["xmp:ModifyDate", /<xmp:ModifyDate>(.*?)</], ["pdf:Producer", /<pdf:Producer>(.*?)</]]) { const m = re.exec(x); if (m) out[k] = m[1]; } } out.kind = "PDF"; out.hint = "Author / Creator に個人名や社内のソフト名が残っていることがある（公開前に消す）"; return out; }
  if (buf[0] === 0x50 && buf[1] === 0x4B) { out.kind = "Office/ZIP"; try { const files = await unzipEntries(buf, ["docProps/core.xml", "docProps/app.xml"]); const core = files["docProps/core.xml"]; if (core) for (const [k, re] of [["creator", /<dc:creator>(.*?)</], ["lastModifiedBy", /<cp:lastModifiedBy>(.*?)</], ["created", /<dcterms:created[^>]*>(.*?)</], ["modified", /<dcterms:modified[^>]*>(.*?)</], ["title", /<dc:title>(.*?)</], ["revision", /<cp:revision>(.*?)</]]) { const m = re.exec(core); if (m) out[k] = m[1]; } const app = files["docProps/app.xml"]; if (app) for (const [k, re] of [["Application", /<Application>(.*?)</], ["Company", /<Company>(.*?)</], ["TotalTime(min)", /<TotalTime>(.*?)</], ["Template", /<Template>(.*?)</]]) { const m = re.exec(app); if (m) out[k] = m[1]; } out.hint = "creator / lastModifiedBy / Company は作成者の手がかり。TotalTime は編集時間（分）"; } catch (e) { out.error = String(e); } return out; }
  out.kind = "未対応（JPEG は EXIF、PDF、docx/xlsx/pptx に対応）"; return out;
}
async function unzipEntries(buf, wanted) { /* 中央ディレクトリを読み、必要な項目だけ DecompressionStream で展開 */ const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength); let eocd = -1; for (let i = buf.length - 22; i >= Math.max(0, buf.length - 66000); i--) if (dv.getUint32(i, true) === 0x06054b50) { eocd = i; break; } if (eocd < 0) throw new Error("ZIP の末尾が見つからない"); const n = dv.getUint16(eocd + 10, true), cd = dv.getUint32(eocd + 16, true); const out = {}; let p = cd; const td = new TextDecoder();
  for (let i = 0; i < n; i++) { if (dv.getUint32(p, true) !== 0x02014b50) break; const method = dv.getUint16(p + 10, true), csize = dv.getUint32(p + 20, true), nlen = dv.getUint16(p + 28, true), elen = dv.getUint16(p + 30, true), clen = dv.getUint16(p + 32, true), off = dv.getUint32(p + 42, true); const name = td.decode(buf.subarray(p + 46, p + 46 + nlen)); p += 46 + nlen + elen + clen; if (!wanted.includes(name)) continue; const lnlen = dv.getUint16(off + 26, true), lelen = dv.getUint16(off + 28, true); const data = buf.subarray(off + 30 + lnlen + lelen, off + 30 + lnlen + lelen + csize); if (method === 0) out[name] = td.decode(data); else if (method === 8) { const ds = new DecompressionStream("deflate-raw"); const w = ds.writable.getWriter(); w.write(data); w.close(); out[name] = await new Response(ds.readable).text(); } } return out; }

/* ---------- ブラウザの露出情報（自分の OpSec 確認）。何も送信しない ---------- */
export function fingerprint() { const c = document.createElement("canvas"); let gl = "", vendor = ""; try { const g = c.getContext("webgl"); const d = g.getExtension("WEBGL_debug_renderer_info"); gl = g.getParameter(d.UNMASKED_RENDERER_WEBGL); vendor = g.getParameter(d.UNMASKED_VENDOR_WEBGL); } catch { } return { "User-Agent": navigator.userAgent, 言語: navigator.languages.join(", "), タイムゾーン: Intl.DateTimeFormat().resolvedOptions().timeZone, 画面: `${screen.width}×${screen.height} @${devicePixelRatio}x、色深度 ${screen.colorDepth}`, "CPU コア": navigator.hardwareConcurrency, メモリ: navigator.deviceMemory ? navigator.deviceMemory + " GB" : "?", GPU: `${vendor} ${gl}`, プラットフォーム: navigator.platform, "Do Not Track": navigator.doNotTrack, Cookie: navigator.cookieEnabled ? "有効" : "無効", タッチ: navigator.maxTouchPoints, 注意: "これらの組み合わせで個体識別されうる（Cover Your Tracks で一意性を確認）。IP は本ページからは分からないが、収集先の各 API には端末の IP が渡る（VPN/Tor は自己責任・規約順守で）" }; }

/* ---------- 検索演算子の一覧（説明つき）と用途別プリセット（{q} は対象で置換） ---------- */
export const DORK_OPERATORS = [
  ["site:", "特定サイト・ドメイン内だけ", "site:city.tsu.lg.jp 避難所"], ["-", "語・サイトを除外", "津 大雨 -天気予報"], ["\"…\"", "完全一致", "\"志登茂川 氾濫\""], ["OR / |", "どちらか", "\"津市\" (氾濫 OR 冠水)"], ["*", "任意の語", "\"津市 * 避難指示\""], ["filetype: / ext:", "ファイル形式", "filetype:pdf 避難所 一覧"], ["intitle: / allintitle:", "タイトルに含む", "intitle:\"index of\" backup"], ["inurl: / allinurl:", "URL に含む", "inurl:admin login"], ["intext: / allintext:", "本文に含む", "intext:\"内部資料\""], ["inanchor:", "リンク文字列に含む", "inanchor:ダウンロード"], ["before: / after:", "期間（YYYY-MM-DD）", "after:2026-09-01 津 大雨"], ["..", "数値範囲", "震度 5..7"], ["AROUND(n)", "n 語以内に共起（Google）", "津市 AROUND(5) 冠水"], ["related:", "似たサイト（Google）", "related:jma.go.jp"], ["define:", "定義", "define:typed decision"], ["cache:", "（廃止）→ Wayback を使う", "—"], ["ip:（Bing）", "同一 IP のサイト", "ip:203.0.113.1"], ["contains:（Bing）", "指定形式へのリンクを含む", "contains:pdf"], ["!bang（DuckDuckGo）", "他サービスへ直接", "!w 津市"],
];
export const DORK_LIBRARY = [
  { cat: "組織の公開文書", name: "自治体・官公庁の PDF", q: "{q} filetype:pdf (site:lg.jp OR site:go.jp)" }, { cat: "組織の公開文書", name: "Excel / CSV の公開データ", q: "{q} (filetype:xlsx OR filetype:xls OR filetype:csv)" }, { cat: "組織の公開文書", name: "プレゼン資料", q: "{q} (filetype:pptx OR filetype:ppt)" },
  { cat: "サイトの構造", name: "サイト内の全ページ", q: "site:{q}" }, { cat: "サイトの構造", name: "サブドメイン（www 以外）", q: "site:{q} -site:www.{q}" }, { cat: "サイトの構造", name: "ログイン画面", q: "site:{q} (inurl:login OR inurl:signin OR intitle:ログイン)" }, { cat: "サイトの構造", name: "公開ディレクトリ一覧（自サイト点検）", q: "site:{q} intitle:\"index of\"" }, { cat: "サイトの構造", name: "設定・ログの露出（自サイト点検）", q: "site:{q} (ext:log OR ext:env OR ext:bak OR ext:sql OR ext:ini)" },
  { cat: "報道・発表", name: "報道機関の記事", q: "\"{q}\" (site:nhk.or.jp OR site:asahi.com OR site:mainichi.jp OR site:yomiuri.co.jp OR site:nikkei.com)" }, { cat: "報道・発表", name: "公的機関の発表", q: "\"{q}\" (site:go.jp OR site:lg.jp)" }, { cat: "報道・発表", name: "直近 1 週間", q: "\"{q}\" after:{week}" }, { cat: "報道・発表", name: "プレスリリース", q: "\"{q}\" (site:prtimes.jp OR intitle:プレスリリース)" },
  { cat: "SNS・掲示板", name: "X の投稿", q: "\"{q}\" site:x.com" }, { cat: "SNS・掲示板", name: "掲示板・Q&A", q: "\"{q}\" (site:5ch.net OR site:detail.chiebukuro.yahoo.co.jp OR site:reddit.com)" }, { cat: "SNS・掲示板", name: "動画", q: "\"{q}\" (site:youtube.com OR site:tiktok.com)" },
  { cat: "研究", name: "論文（PDF）", q: "\"{q}\" filetype:pdf (site:arxiv.org OR site:aclanthology.org OR site:jstage.jst.go.jp)" }, { cat: "研究", name: "GitHub のコード", q: "\"{q}\" site:github.com" }, { cat: "研究", name: "Hugging Face のモデル", q: "\"{q}\" site:huggingface.co" },
  { cat: "詐欺・なりすまし確認", name: "同名サイトの重複", q: "intitle:\"{q}\" -site:{d}" }, { cat: "詐欺・なりすまし確認", name: "被害・注意喚起の投稿", q: "\"{q}\" (詐欺 OR 注意 OR 偽 OR なりすまし)" },
];
export const buildFromLibrary = (item, q, d = "") => item.q.replace(/\{q\}/g, q).replace(/\{d\}/g, d || q).replace(/\{week\}/g, new Date(Date.now() - 7 * 864e5).toISOString().slice(0, 10));
