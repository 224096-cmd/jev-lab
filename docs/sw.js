/* アプリ本体・CDN ライブラリ・読み込んだモデルをキャッシュし、機内モードでも play.html が動くようにする */
const APP = "jev-lab-app-v20", LIB = "jev-lab-lib-v20", MODELS = "jev-lab-models-v20";
const SHELL = ["./", "./index.html", "./play.html", "./jev.js", "./app.js", "./lab.js", "./ui.js", "./presets.js", "./extras.js", "./sources_more.js", "./params.js", "./app.css", "./templates.js", "./osint.js", "./tools.js", "./recon.js", "./quota.js", "./connect.js", "./suggest.js", "./bench/index.json", "./bench/jevbench_ja_small.jsonl", "./bench/survey_example.jsonl", "./theory.html", "./models/registry.json", "./bench/sample_ja.jsonl", "./style.css", "./manifest.json", "./icon.svg", "./models/index.json", "./results/index.json"];
// 端末内推論に必要な CDN ファイル（play.html が実際に読む 4 つ）。install 時に先読みしておく
const CDN = [
  "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.5.2/dist/transformers.min.js",
  "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.webgpu.min.mjs",
  "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort-wasm-simd-threaded.jsep.mjs",
  "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort-wasm-simd-threaded.jsep.wasm",
  "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js",
  "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css",
];
self.addEventListener("install", e => {
  e.waitUntil((async () => {
    const app = await caches.open(APP); await app.addAll(SHELL);
    const lib = await caches.open(LIB);
    await Promise.all(CDN.map(u => lib.add(u).catch(() => {})));   // 1 つ失敗しても install は続ける
    await self.skipWaiting();
  })());
});
self.addEventListener("activate", e => {
  e.waitUntil((async () => {
    for (const k of await caches.keys()) if (![APP, LIB, MODELS].includes(k)) await caches.delete(k);
    await clients.claim();
  })());
});
self.addEventListener("fetch", e => {
  if (e.request.method !== "GET") return;
  const u = new URL(e.request.url);
  const isModel = u.pathname.includes("/models/") && !u.pathname.endsWith("index.json") && !u.pathname.endsWith("registry.json");
  const isCdn = u.hostname === "cdn.jsdelivr.net";
  if (isModel) return;      // モデルファイルはアプリが IndexedDB に保存する（Cache API と二重に持たない。v3.1 修正）
  if (isCdn) {              // ライブラリ: cache-first
    e.respondWith(caches.open(LIB).then(async c => (await c.match(e.request)) || fetch(e.request).then(r => { if (r.ok) c.put(e.request, r.clone()); return r; })));
    return;
  }
  // アプリ本体・結果 JSON: network-first、失敗時 cache
  e.respondWith(fetch(e.request).then(r => { if (r.ok) caches.open(APP).then(c => c.put(e.request, r.clone())); return r; }).catch(() => caches.match(e.request)));
});
// ページからの問い合わせ: 指定モデルがオフライン実行可能か（CDN 4 点 + モデル 4 点が全部キャッシュ済みか）
self.addEventListener("message", async e => {
  if (e.data?.type !== "offline-check") return;
  const need = CDN.slice(0, 4); const lib = await caches.open(LIB); const missing = [];
  for (const u of need) if (!(await lib.match(u))) missing.push(u.split("/").pop());
  e.source.postMessage({ type: "offline-status", model: e.data.model, ready: missing.length === 0, missing });
});
