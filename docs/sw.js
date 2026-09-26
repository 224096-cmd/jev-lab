/* アプリ本体と読み込んだモデルをキャッシュし、機内モードでも play.html が動くようにする */
const APP = "jev-lab-app-v1", MODELS = "jev-lab-models-v1";
const SHELL = ["./", "./index.html", "./play.html", "./jev.js", "./style.css", "./manifest.json", "./icon.svg", "./models/index.json"];
self.addEventListener("install", e => { e.waitUntil(caches.open(APP).then(c => c.addAll(SHELL)).then(() => self.skipWaiting())); });
self.addEventListener("activate", e => { e.waitUntil(clients.claim()); });
self.addEventListener("fetch", e => {
  const u = new URL(e.request.url);
  if (e.request.method !== "GET") return;
  // モデル・トークナイザ・CDN ライブラリ: cache-first（大きい・変わらない）
  if (u.pathname.includes("/models/") || u.hostname === "cdn.jsdelivr.net") {
    e.respondWith(caches.open(MODELS).then(async c => (await c.match(e.request)) || fetch(e.request).then(r => { if (r.ok) c.put(e.request, r.clone()); return r; })));
    return;
  }
  // アプリ本体・結果 JSON: network-first、失敗時 cache
  e.respondWith(fetch(e.request).then(r => { if (r.ok) caches.open(APP).then(c => c.put(e.request, r.clone())); return r; }).catch(() => caches.match(e.request)));
});
