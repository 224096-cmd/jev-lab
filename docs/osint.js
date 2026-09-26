/* OSINT（公開情報収集）モジュール — 合法・公開・ログイン不要の情報源のみ
   方針:
   - 利用するのは各運営者が公開 API として提供し CORS を許可しているものだけ（規約・レート制限を守る）
   - 対象は「出来事・地域・話題」。個人の特定・追跡・プロファイリングには使わない（UI に明記）
   - すべての取得結果に source / url / fetched_at を残し、証跡（evidence pack）として保存できる */

const UA_NOTE = "jev-lab (research, https://github.com/224096-cmd/jev-lab)";
const sleep = ms => new Promise(r => setTimeout(r, ms));
const item = (source, o) => ({ source, fetched_at: new Date().toISOString(), ...o });

export const SOURCES = {
  jma: {
    name: "気象庁 防災情報（警報・地震）", lang: "ja", kind: "official",
    desc: "www.jma.go.jp/bosai の公開 JSON。都道府県コードで警報、全国の最新地震一覧。",
    async run(q, opt) {
      const out = [];
      const code = opt.prefCode || "240000"; // 三重県
      try { const w = await (await fetch(`https://www.jma.go.jp/bosai/warning/data/warning/${code}.json`)).json();
        out.push(item("jma_warning", { title: `気象警報・注意報（${w.publishingOffice}）`, text: w.headlineText || "（発表なし）", time: w.reportDatetime, url: `https://www.jma.go.jp/bosai/warning/#area_type=offices&area_code=${code.slice(0, 2)}0000`, official: true })); } catch (e) { out.push(item("jma_warning", { error: String(e) })); }
      try { const qs = await (await fetch("https://www.jma.go.jp/bosai/quake/data/list.json")).json();
        for (const e of qs.slice(0, opt.limit || 5)) out.push(item("jma_quake", { title: `地震情報 ${e.anm || ""} M${e.mag || "?"} 最大震度${e.maxi || "?"}`, text: `${e.at || ""} ${e.anm || ""} 深さ${e.dep ?? "?"}km マグニチュード${e.mag ?? "?"} 最大震度${e.maxi ?? "?"}${e.ttl ? " / " + e.ttl : ""}`, time: e.at, url: "https://www.jma.go.jp/bosai/map.html#contents=earthquake_map", official: true })); } catch (e) { out.push(item("jma_quake", { error: String(e) })); }
      return out;
    },
  },
  wikipedia: {
    name: "Wikipedia（日本語）", lang: "ja", kind: "reference",
    desc: "検索 API（origin=*）。地名・施設・出来事の基礎情報を根拠（context）として使う。",
    async run(q, opt) {
      const r = await (await fetch(`https://ja.wikipedia.org/w/api.php?action=query&list=search&srsearch=${encodeURIComponent(q)}&srlimit=${opt.limit || 5}&format=json&origin=*`)).json();
      const out = [];
      for (const s of r.query.search) {
        let extract = "";
        try { const e = await (await fetch(`https://ja.wikipedia.org/w/api.php?action=query&prop=extracts&exintro=1&explaintext=1&titles=${encodeURIComponent(s.title)}&format=json&origin=*`)).json(); extract = Object.values(e.query.pages)[0].extract || ""; } catch { }
        out.push(item("wikipedia", { title: s.title, text: (extract || s.snippet.replace(/<[^>]+>/g, "")).slice(0, 600), time: s.timestamp, url: `https://ja.wikipedia.org/wiki/${encodeURIComponent(s.title)}` }));
        await sleep(300);
      }
      return out;
    },
  },
  wikidata: {
    name: "Wikidata（構造化データ）", lang: "multi", kind: "reference",
    desc: "エンティティ検索 → 説明文と座標。地名の実在確認に。",
    async run(q, opt) {
      const r = await (await fetch(`https://www.wikidata.org/w/api.php?action=wbsearchentities&search=${encodeURIComponent(q)}&language=ja&limit=${opt.limit || 5}&format=json&origin=*`)).json();
      return r.search.map(s => item("wikidata", { title: `${s.label} (${s.id})`, text: s.description || "", url: `https://www.wikidata.org/wiki/${s.id}` }));
    },
  },
  nominatim: {
    name: "OpenStreetMap Nominatim（地名→座標）", lang: "multi", kind: "geo",
    desc: "地名の実在と座標。1 秒 1 リクエストの利用規約。ハザード DB との照合の入口。",
    async run(q, opt) {
      const r = await (await fetch(`https://nominatim.openstreetmap.org/search?q=${encodeURIComponent(q)}&format=json&limit=${opt.limit || 3}&accept-language=ja`)).json();
      return r.map(p => item("nominatim", { title: p.display_name, text: `lat ${p.lat}, lon ${p.lon}, type ${p.type}, class ${p.class}`, lat: +p.lat, lon: +p.lon, url: `https://www.openstreetmap.org/${p.osm_type}/${p.osm_id}` }));
    },
  },
  bluesky: {
    name: "Bluesky 公開投稿", lang: "multi", kind: "social",
    desc: "public.api.bsky.app の公開検索（ログイン不要）。一般投稿の速報性・拡散表現の検証に。",
    async run(q, opt) {
      const r = await (await fetch(`https://public.api.bsky.app/xrpc/app.bsky.feed.searchPosts?q=${encodeURIComponent(q)}&limit=${opt.limit || 10}${opt.lang ? "&lang=" + opt.lang : ""}`)).json();
      return (r.posts || []).map(p => item("bluesky", { title: `@${p.author.handle}`, text: p.record?.text || "", time: p.record?.createdAt, url: `https://bsky.app/profile/${p.author.handle}/post/${p.uri.split("/").pop()}`, likes: p.likeCount, reposts: p.repostCount }));
    },
  },
  gdelt: {
    name: "GDELT（世界のニュース見出し）", lang: "multi", kind: "news",
    desc: "api.gdeltproject.org DOC API。5 秒に 1 回まで。英語・日本語の報道を横断。",
    async run(q, opt) {
      const r = await (await fetch(`https://api.gdeltproject.org/api/v2/doc/doc?query=${encodeURIComponent(q)}&mode=artlist&maxrecords=${opt.limit || 10}&format=json${opt.lang === "ja" ? "&sourcelang=japanese" : ""}`)).json();
      return (r.articles || []).map(a => item("gdelt", { title: a.title, text: a.title, time: a.seendate, url: a.url, domain: a.domain, lang: a.language }));
    },
  },
  hn: {
    name: "Hacker News（技術系・英語）", lang: "en", kind: "social",
    desc: "hn.algolia.com 検索。英語モデル（open-jev 等）の比較用。",
    async run(q, opt) {
      const r = await (await fetch(`https://hn.algolia.com/api/v1/search?query=${encodeURIComponent(q)}&hitsPerPage=${opt.limit || 10}`)).json();
      return r.hits.map(h => item("hn", { title: h.title || h.story_title || "", text: (h.story_text || h.comment_text || h.title || "").replace(/<[^>]+>/g, "").slice(0, 600), time: h.created_at, url: h.url || `https://news.ycombinator.com/item?id=${h.objectID}`, points: h.points }));
    },
  },
};

/* 国土地理院 標高タイル（PNG DEM, z=14）から標高[m] を読む。CORS 許可あり */
export async function elevation(lat, lon, z = 14) {
  const n = 2 ** z, x = (lon + 180) / 360 * n, latr = lat * Math.PI / 180, y = (1 - Math.log(Math.tan(latr) + 1 / Math.cos(latr)) / Math.PI) / 2 * n;
  const tx = Math.floor(x), ty = Math.floor(y), px = Math.floor((x - tx) * 256), py = Math.floor((y - ty) * 256);
  for (const src of ["dem_png", "dem5a_png"]) {
    try { const img = await new Promise((ok, ng) => { const im = new Image(); im.crossOrigin = "anonymous"; im.onload = () => ok(im); im.onerror = ng; im.src = `https://cyberjapandata.gsi.go.jp/xyz/${src}/${z}/${tx}/${ty}.png`; });
      const c = document.createElement("canvas"); c.width = c.height = 256; const g = c.getContext("2d"); g.drawImage(img, 0, 0); const [r, gg, b] = g.getImageData(px, py, 1, 1).data;
      let v = r * 65536 + gg * 256 + b; if (v === 8388608) continue; if (v > 8388608) v -= 16777216; return { elevation_m: v * 0.01, source: `GSI ${src} z${z}` }; } catch { }
  }
  return null;
}

/* 地理照合: テキスト中の地名候補（Nominatim 結果）に標高を付け、「水没」「浸水」等の主張と突き合わせる材料にする */
export async function geoContext(items) {
  const out = [];
  for (const it of items) if (it.source === "nominatim" && it.lat != null) { const e = await elevation(it.lat, it.lon); if (e) { it.elevation_m = e.elevation_m; out.push(`${it.title.split(",")[0]}: 標高 約${e.elevation_m.toFixed(0)}m（${e.source}）、座標 ${it.lat.toFixed(4)}, ${it.lon.toFixed(4)}`); } await sleep(300); }
  return out;
}

/* 説明可能な信頼性スコア（0〜1）。重みは編集可能で、内訳を返す（JEV-Rule/Score の考え方） */
export const TRUST_WEIGHTS_DEFAULT = { official: 0.35, supported: 0.25, has_specifics: 0.15, kind_report: 0.10, no_spread: 0.10, relevant: 0.05 };
export function trustScore(item, j, w = TRUST_WEIGHTS_DEFAULT) {
  if (!j) return null;
  const parts = { official: item.official ? 1 : 0, supported: j.supported?.p_yes ?? 0.5, has_specifics: j.has_specifics?.p_yes ?? 0.5, kind_report: ({ "公的機関の発表": 1, "報道": 0.8, "一般の投稿・目撃": 0.5, "意見・感想": 0.2, "宣伝・無関係": 0 })[j.kind?.choice] ?? 0.5, no_spread: 1 - (j.asks_spread?.p_yes ?? 0.5), relevant: j.relevant?.p_yes ?? 0.5 };
  const tot = Object.values(w).reduce((a, b) => a + b, 0) || 1; const score = Object.entries(w).reduce((s, [k, wk]) => s + wk * parts[k], 0) / tot;
  return { score, parts, weights: w };
}

/* OSINT トリアージ用の標準質問セット（JEV に一括で問う） */
export const OSINT_QUESTIONS = (query) => [
  { type: "noul", id: "relevant", instructions: `この情報は「${query}」に関係しているか` },
  { type: "choice", id: "kind", instructions: "この情報の種類", options: ["公的機関の発表", "報道", "一般の投稿・目撃", "意見・感想", "宣伝・無関係"] },
  { type: "score", id: "urgency", instructions: "今すぐ対応や確認が必要な度合い", levels: ["0", "1", "2", "3", "4", "5"] },
  { type: "noul", id: "has_specifics", instructions: "日時・場所・数量など検証可能な具体情報が含まれているか" },
  { type: "noul", id: "asks_spread", instructions: "拡散や転送を呼びかける表現があるか" },
  { type: "noul", id: "supported", instructions: "根拠（context）の内容と整合しているか" },
];

/* 収集の実行。sources: キー配列、opt: {limit, prefCode, lang}。進捗コールバックつき */
export async function collect(query, sources, opt = {}, onProgress = () => {}) {
  const all = [];
  for (const k of sources) {
    const s = SOURCES[k]; if (!s) continue;
    onProgress(`${s.name} を取得中…`);
    try { const items = await s.run(query, opt); all.push(...items); }
    catch (e) { all.push(item(k, { error: String(e) })); }
    if (k === "gdelt") await sleep(5000); else await sleep(1000);
  }
  onProgress("");
  return all;
}

export function evidencePack(query, items, judged) {
  return { query, collected_at: new Date().toISOString(), policy: "公開 API のみ・ログイン不要・個人の特定目的では使用しない", user_agent_note: UA_NOTE, items: items.map((it, i) => ({ ...it, judgement: judged?.[i] || null })) };
}
