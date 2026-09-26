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
