/* 既存アプリとの連携（サーバー無し・無料で成立するものだけ）
   受け取る:
     - Web Share Target（Android の「共有」→ JEV Lab）: manifest.json の share_target → play.html?title=&text=&url=
     - URL パラメータ: play.html?text=…&tmpl=sns_verify&run=1 / ?q=津市 大雨 / ?target=jma.go.jp（他アプリ・Google フォーム・ショートカットから呼べる）
     - ブックマークレット（PC のブラウザで選択した文章を送る）、iOS ショートカット（共有シート → URL を開く）
     - Bluesky（アプリパスワードでログイン → 自分のタイムライン／通知を取り込んで仕分け）
     - Mastodon（アクセストークン → ホーム／通知）
     - Google スプレッドシート（「ウェブに公開」した CSV の URL）／CSV ファイル → 一括判断
   送る:
     - Discord Webhook（ブラウザから POST 可）、Slack Incoming Webhook（no-cors で送信）、OS の共有（navigator.share）、クリップボード
   認証情報は localStorage にだけ保存し、外部には各サービス以外に送らない */
const J = async (u, init) => { const r = await fetch(u, init); if (!r.ok) throw new Error(`${r.status} ${u.split("?")[0].split("/").slice(-1)[0]}: ${(await r.text()).slice(0, 120)}`); return r.json(); };
const item = (source, o) => ({ source, fetched_at: new Date().toISOString(), ...o });
const K = "jev.connect"; export const conf = { load() { try { return JSON.parse(localStorage.getItem(K) || "{}"); } catch { return {}; } }, save(c) { localStorage.setItem(K, JSON.stringify(c)); }, set(k, v) { const c = this.load(); c[k] = v; this.save(c); }, del(k) { const c = this.load(); delete c[k]; this.save(c); } };

/* ---------- 受け取り: URL パラメータ ---------- */
export function parseIncoming() { const p = new URLSearchParams(location.search); const o = {}; for (const k of ["text", "title", "url", "q", "target", "tmpl", "run", "tab", "ctx"]) if (p.has(k)) o[k] = p.get(k); if (!Object.keys(o).length) return null; history.replaceState(null, "", location.pathname); return o; }
export const appUrl = () => location.origin + location.pathname;
export const bookmarklet = () => `javascript:(function(){var t=String(window.getSelection&&window.getSelection().toString()||'').trim()||document.title;window.open('${appUrl()}?text='+encodeURIComponent(t.slice(0,2000))+'&url='+encodeURIComponent(location.href)+'&title='+encodeURIComponent(document.title.slice(0,120))+'&run=1','_blank')})()`;

/* ---------- Bluesky（AT Protocol、アプリパスワード） ---------- */
export const bluesky = {
  async login(identifier, appPassword) { const s = await J("https://bsky.social/xrpc/com.atproto.server.createSession", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ identifier, password: appPassword }) }); conf.set("bsky", { did: s.did, handle: s.handle, accessJwt: s.accessJwt, refreshJwt: s.refreshJwt, pds: "https://bsky.social" }); return s.handle; },
  async refresh() { const b = conf.load().bsky; if (!b) throw new Error("Bluesky 未ログイン"); try { const s = await J(`${b.pds}/xrpc/com.atproto.server.refreshSession`, { method: "POST", headers: { Authorization: "Bearer " + b.refreshJwt } }); conf.set("bsky", { ...b, accessJwt: s.accessJwt, refreshJwt: s.refreshJwt }); } catch (e) { throw new Error("Bluesky のセッションが切れました。もう一度ログインしてください"); } },
  logout() { conf.del("bsky"); }, get session() { return conf.load().bsky || null; },
  async _get(path, params) { let b = this.session; if (!b) throw new Error("Bluesky 未ログイン"); const u = `${b.pds}/xrpc/${path}?${new URLSearchParams(params)}`; try { return await J(u, { headers: { Authorization: "Bearer " + b.accessJwt } }); } catch (e) { if (/401|ExpiredToken/.test(String(e))) { await this.refresh(); b = this.session; return J(u, { headers: { Authorization: "Bearer " + b.accessJwt } }); } throw e; } },
  post2item(p, src) { return item(src, { title: `@${p.author?.handle}${p.author?.displayName ? "（" + p.author.displayName + "）" : ""}`, text: p.record?.text || "", time: p.record?.createdAt || p.indexedAt, url: `https://bsky.app/profile/${p.author?.handle}/post/${(p.uri || "").split("/").pop()}`, likes: p.likeCount, reposts: p.repostCount, lang: (p.record?.langs || [])[0] }); },
  async timeline(limit = 30) { const r = await this._get("app.bsky.feed.getTimeline", { limit }); return (r.feed || []).map(f => this.post2item(f.post, "bluesky:home")); },
  async notifications(limit = 30) { const r = await this._get("app.bsky.notification.listNotifications", { limit }); return (r.notifications || []).filter(n => n.record?.text).map(n => item("bluesky:notify", { title: `@${n.author?.handle}（${n.reason}）`, text: n.record?.text || "", time: n.indexedAt, url: `https://bsky.app/profile/${n.author?.handle}/post/${(n.uri || "").split("/").pop()}`, reason: n.reason })); },
  async search(q, limit = 25) { const r = await this._get("app.bsky.feed.searchPosts", { q, limit }); return (r.posts || []).map(p => this.post2item(p, "bluesky:search")); },
};

/* ---------- Mastodon（アクセストークン） ---------- */
export const mastodon = {
  save(instance, token) { conf.set("masto", { instance: instance.replace(/\/$/, "").replace(/^(?!https?:\/\/)/, "https://"), token }); }, logout() { conf.del("masto"); }, get session() { return conf.load().masto || null; },
  async _get(path, params = {}) { const m = this.session; if (!m) throw new Error("Mastodon 未設定"); return J(`${m.instance}${path}?${new URLSearchParams(params)}`, { headers: { Authorization: "Bearer " + m.token } }); },
  st2item(s, src) { return item(src, { title: `@${s.account?.acct}`, text: (s.content || "").replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim(), time: s.created_at, url: s.url, likes: s.favourites_count, reposts: s.reblogs_count, lang: s.language }); },
  async verify() { const a = await this._get("/api/v1/accounts/verify_credentials"); return a.acct; },
  async home(limit = 40) { return (await this._get("/api/v1/timelines/home", { limit })).map(s => this.st2item(s.reblog || s, "mastodon:home")); },
  async notifications(limit = 40) { return (await this._get("/api/v1/notifications", { limit })).filter(n => n.status).map(n => ({ ...this.st2item(n.status, "mastodon:notify"), title: `@${n.account?.acct}（${n.type}）` })); },
};

/* ---------- Google スプレッドシート（ウェブに公開した CSV）/ CSV テキスト ---------- */
export function parseCsv(text) { const rows = []; let row = [], cell = "", q = false; for (let i = 0; i < text.length; i++) { const c = text[i]; if (q) { if (c === '"' && text[i + 1] === '"') { cell += '"'; i++; } else if (c === '"') q = false; else cell += c; } else if (c === '"') q = true; else if (c === ",") { row.push(cell); cell = ""; } else if (c === "\n" || c === "\r") { if (c === "\r" && text[i + 1] === "\n") i++; row.push(cell); rows.push(row); row = []; cell = ""; } else cell += c; } if (cell || row.length) { row.push(cell); rows.push(row); } return rows.filter(r => r.some(x => x.trim())); }
export async function sheetCsv(url) { let u = url.trim(); const m = /docs\.google\.com\/spreadsheets\/d\/([\w-]+)/.exec(u); if (m && !/output=csv|format=csv/.test(u)) u = `https://docs.google.com/spreadsheets/d/${m[1]}/export?format=csv`; const r = await fetch(u); if (!r.ok) throw new Error(`HTTP ${r.status}（「ファイル → 共有 → ウェブに公開 → CSV」の URL、またはリンクを知っている全員が閲覧可にする）`); return parseCsv(await r.text()); }

/* ---------- 送る ---------- */
export async function sendDiscord(webhook, content) { const r = await fetch(webhook, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ content: content.slice(0, 1900) }) }); if (!r.ok && r.status !== 204) throw new Error("Discord " + r.status); return true; }
export async function sendSlack(webhook, text) { await fetch(webhook, { method: "POST", mode: "no-cors", headers: { "Content-Type": "text/plain" }, body: JSON.stringify({ text: text.slice(0, 3000) }) }); return true; }
export async function shareText(title, text, url) { if (navigator.share) { await navigator.share({ title, text, url }); return "share"; } await navigator.clipboard.writeText(text + (url ? "\n" + url : "")); return "clipboard"; }
