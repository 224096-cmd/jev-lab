/* 公開 API の「無料枠を超えない」仕組み
   1. 予算（budget）: API ごとに 1 日・1 時間・1 分の上限回数（無料枠より少なめ）と、文字数の上限（翻訳）を localStorage で数える。超えたら呼ばずにエラー（UI は「上限のためスキップ」と表示）
   2. 間隔（interval）: API ごとの最短呼び出し間隔（1 req/s 等の利用規約）を守って待つ
   3. キャッシュ（cache）: 同じ URL の応答を TTL 付きで IndexedDB に保存。キャッシュ命中は予算を消費しない（同じ対象を何度調べても無料枠を減らさない）
   すべての API 呼び出しは apiFetch() を通す */
import { store } from "./jev.js";

/* 無料枠（各サービスの公開情報に基づく控えめな値）。day/hour/min = 回数、chars = 1 日の文字数、interval = ms、ttl = キャッシュ秒 */
export const LIMITS = {
  hackertarget: { day: 40, interval: 2000, ttl: 7 * 86400, note: "無料 50 回/日（IP ごと）" },
  mymemory: { chars: 4500, interval: 1000, ttl: 30 * 86400, note: "無料 5,000 文字/日" },
  github: { min: 8, hour: 50, interval: 6500, ttl: 86400, note: "未認証 10 回/分（検索）・60 回/時" },
  nvd: { interval: 6500, day: 300, ttl: 86400, note: "鍵なし 5 回/30 秒" },
  nominatim: { interval: 1100, day: 500, ttl: 7 * 86400, note: "1 回/秒（利用規約）" },
  gdelt: { interval: 5000, day: 200, ttl: 600, note: "5 秒に 1 回" },
  semanticscholar: { interval: 1500, hour: 60, ttl: 6 * 3600, note: "鍵なしは低頻度" },
  crossref: { interval: 500, day: 500, ttl: 6 * 3600, note: "polite pool" },
  arxiv: { interval: 3100, day: 300, ttl: 6 * 3600, note: "3 秒に 1 回" },
  stackexchange: { interval: 500, day: 250, ttl: 6 * 3600, note: "匿名 300 回/日（IP ごと）" },
  ipinfo: { interval: 500, day: 800, ttl: 7 * 86400, note: "未認証 1,000 回/日" },
  otx: { interval: 1000, day: 200, ttl: 86400, note: "鍵なしの公開エンドポイント" },
  rdap: { interval: 700, day: 300, ttl: 7 * 86400 },
  crtsh: { interval: 3000, day: 100, ttl: 86400, note: "負荷が高いので控えめに" },
  mastodon: { interval: 1000, day: 300, ttl: 300, note: "300 回/5 分" },
  bluesky: { interval: 600, day: 1000, ttl: 300 },
  wayback: { interval: 1000, day: 500, ttl: 86400 },
  cdx: { interval: 3000, day: 100, ttl: 86400 },
  dns: { interval: 150, day: 3000, ttl: 3600 },
  mempool: { interval: 1000, day: 200, ttl: 3600 },
  blockscout: { interval: 1000, day: 200, ttl: 3600 },
  gravatar: { interval: 1000, day: 200, ttl: 7 * 86400 },
  pwned: { interval: 1600, day: 100, ttl: 0, note: "k-匿名性。キャッシュしない" },
  wikipedia: { interval: 500, day: 1000, ttl: 3600 },
  wikidata: { interval: 500, day: 1000, ttl: 3600 },
  jma: { interval: 1000, day: 500, ttl: 300, note: "5 分キャッシュ" },
  gsi: { interval: 200, day: 2000, ttl: 30 * 86400, note: "標高タイルは長期キャッシュ" },
  osm: { interval: 1000, day: 300, ttl: 86400 },
  hn: { interval: 500, day: 1000, ttl: 600 },
  hf: { interval: 1000, day: 300, ttl: 3600 },
};
const K = "jev.quota"; const load = () => { try { return JSON.parse(localStorage.getItem(K) || "{}"); } catch { return {}; } }; const save = q => { try { localStorage.setItem(K, JSON.stringify(q)); } catch { } };
const today = () => new Date().toISOString().slice(0, 10);
const lastCall = {};
export class QuotaError extends Error { constructor(api, why) { super(`${api}: ${why}`); this.api = api; this.quota = true; } }

/* 予算の状態（設定タブに表示） */
export function quotaStatus() { const q = load(); const now = Date.now(); return Object.entries(LIMITS).map(([api, L]) => { const s = q[api] || {}; const dayN = s.day === today() ? (s.n || 0) : 0; const chars = s.day === today() ? (s.chars || 0) : 0; const ts = (s.ts || []).filter(t => now - t < 3600e3); return { api, note: L.note || "", used_day: dayN, limit_day: L.day, used_hour: ts.length, limit_hour: L.hour, used_min: ts.filter(t => now - t < 60e3).length, limit_min: L.min, chars, limit_chars: L.chars, interval: L.interval, ttl: L.ttl, cache_hits: s.hits || 0 }; }); }
export function quotaReset() { localStorage.removeItem(K); }

function check(api, cost) { const L = LIMITS[api]; if (!L) return; const q = load(); const s = q[api] || {}; const now = Date.now(); if (s.day !== today()) { s.day = today(); s.n = 0; s.chars = 0; } s.ts = (s.ts || []).filter(t => now - t < 3600e3);
  if (L.day && s.n + 1 > L.day) throw new QuotaError(api, `1 日の上限 ${L.day} 回に達したためスキップ（明日リセット）`);
  if (L.hour && s.ts.length + 1 > L.hour) throw new QuotaError(api, `1 時間の上限 ${L.hour} 回に達したためスキップ`);
  if (L.min && s.ts.filter(t => now - t < 60e3).length + 1 > L.min) throw new QuotaError(api, `1 分の上限 ${L.min} 回。少し待ってから`);
  if (L.chars && s.chars + (cost || 0) > L.chars) throw new QuotaError(api, `1 日の文字数上限 ${L.chars} に達したためスキップ`);
  return { q, s }; }
function commit(api, cost, hit) { const q = load(); const s = q[api] || { day: today(), n: 0, chars: 0, ts: [] }; if (s.day !== today()) { s.day = today(); s.n = 0; s.chars = 0; } if (hit) s.hits = (s.hits || 0) + 1; else { s.n = (s.n || 0) + 1; s.chars = (s.chars || 0) + (cost || 0); s.ts = [...(s.ts || []), Date.now()].slice(-500); } q[api] = s; save(q); }
async function waitInterval(api) { const L = LIMITS[api]; if (!L?.interval) return; const t = lastCall[api] || 0; const wait = t + L.interval - Date.now(); if (wait > 0) await new Promise(r => setTimeout(r, wait)); lastCall[api] = Date.now(); }

/* 本体。parse: "json" | "text" | "head"（HEAD で ok だけ）| "buf" */
export async function apiFetch(api, url, { init, parse = "json", cost = 0, ttl, onProgress } = {}) {
  const L = LIMITS[api] || {}; const T = ttl ?? L.ttl ?? 0; const key = `cache:${api}:${url}`;
  if (T > 0 && parse !== "head") { try { const c = await store.cacheGet(key); if (c && Date.now() - c.t < T * 1000) { commit(api, 0, true); return c.v; } } catch { } }
  check(api, cost); await waitInterval(api);
  const r = await fetch(url, init); if (parse === "head") { commit(api, cost); return { ok: r.ok, status: r.status }; }
  if (!r.ok) { commit(api, cost); throw new Error(`${r.status} ${url.split("?")[0]}`); }
  const v = parse === "json" ? await r.json() : parse === "text" ? await r.text() : await r.arrayBuffer(); commit(api, cost);
  if (T > 0) { try { await store.cachePut(key, { t: Date.now(), v }); } catch { } }
  return v;
}
export async function cacheClear() { await store.cacheClear(); }
export const isQuota = e => !!e?.quota;
