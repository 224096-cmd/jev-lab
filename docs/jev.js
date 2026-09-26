/* JEV-JA 端末内推論エンジン v2
   - モデルファイルは IndexedDB に保存（Cache API より消されにくい）。以後はネット不要
   - decide() は答えだけでなく、途中量（トークン列・スパン・u/v ベクトル・logit・類似度）も返す
   - span / marker 両方の読み出しを 1 forward で計算し、温度 T はページ側で再適用できる
   Python 側 jev_lab/adapters/jev_ja.py と同じ手順 */
import { PreTrainedTokenizer } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.5.2/dist/transformers.min.js";
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.webgpu.min.mjs";
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";

/* ---------------- IndexedDB ---------------- */
export const store = {
  db: null,
  async open() { if (this.db) return this.db; return this.db = await new Promise((ok, ng) => { const r = indexedDB.open("jev-lab", 2); r.onupgradeneeded = () => { const db = r.result; if (!db.objectStoreNames.contains("files")) db.createObjectStore("files"); if (!db.objectStoreNames.contains("experiments")) db.createObjectStore("experiments", { keyPath: "id" }); }; r.onsuccess = () => ok(r.result); r.onerror = () => ng(r.error); }); },
  async get(k) { const db = await this.open(); return new Promise((ok, ng) => { const r = db.transaction("files").objectStore("files").get(k); r.onsuccess = () => ok(r.result); r.onerror = () => ng(r.error); }); },
  async put(k, v) { const db = await this.open(); return new Promise((ok, ng) => { const r = db.transaction("files", "readwrite").objectStore("files").put(v, k); r.onsuccess = () => ok(); r.onerror = () => ng(r.error); }); },
  async del(k) { const db = await this.open(); return new Promise((ok, ng) => { const r = db.transaction("files", "readwrite").objectStore("files").delete(k); r.onsuccess = () => ok(); r.onerror = () => ng(r.error); }); },
  async keys() { const db = await this.open(); return new Promise((ok, ng) => { const r = db.transaction("files").objectStore("files").getAllKeys(); r.onsuccess = () => ok(r.result); r.onerror = () => ng(r.error); }); },
  /* 実験ノート */
  async expPut(e) { const db = await this.open(); return new Promise((ok, ng) => { const r = db.transaction("experiments", "readwrite").objectStore("experiments").put(e); r.onsuccess = () => ok(); r.onerror = () => ng(r.error); }); },
  async expAll() { const db = await this.open(); return new Promise((ok, ng) => { const r = db.transaction("experiments").objectStore("experiments").getAll(); r.onsuccess = () => ok(r.result); r.onerror = () => ng(r.error); }); },
  async expDel(id) { const db = await this.open(); return new Promise((ok, ng) => { const r = db.transaction("experiments", "readwrite").objectStore("experiments").delete(id); r.onsuccess = () => ok(); r.onerror = () => ng(r.error); }); },
};

const FILES = ["config.json", "tokenizer.json", "tokenizer_config.json", "head.onnx", "encoder.onnx"];

async function fetchBuf(url, onProgress) {
  const r = await fetch(url); if (!r.ok) throw new Error(`${r.status} ${url}`);
  const total = +r.headers.get("content-length") || 0; const reader = r.body.getReader(); const chunks = []; let got = 0;
  for (;;) { const { done, value } = await reader.read(); if (done) break; chunks.push(value); got += value.length; onProgress?.(got, total); }
  const out = new Uint8Array(got); let o = 0; for (const c of chunks) { out.set(c, o); o += c.length; } return out.buffer;
}

export async function isStored(name) { const ks = new Set(await store.keys()); return FILES.every(f => ks.has(`${name}/${f}`)); }
export async function removeStored(name) { for (const f of FILES) await store.del(`${name}/${f}`); }
export async function storedBytes() { let n = 0; for (const k of await store.keys()) { const v = await store.get(k); n += v?.byteLength || 0; } return n; }

/* ---------------- 数学ユーティリティ ---------------- */
export const softmax = (z, T = 1) => { const m = Math.max(...z); const e = z.map(x => Math.exp((x - m) / T)); const s = e.reduce((a, b) => a + b, 0); return e.map(x => x / s); };
export const cosine = (a, b) => { let d = 0, na = 0, nb = 0; for (let i = 0; i < a.length; i++) { d += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; } return d / Math.sqrt(na * nb || 1); };
/* 2 次元 PCA（べき乗法）。vectors: Float32Array[] → [[x,y],...] */
export function pca2(vecs) {
  const n = vecs.length, d = vecs[0].length, mu = new Float64Array(d);
  for (const v of vecs) for (let i = 0; i < d; i++) mu[i] += v[i] / n;
  const X = vecs.map(v => Float64Array.from(v, (x, i) => x - mu[i]));
  const comps = [];
  for (let c = 0; c < 2; c++) {
    let w = Float64Array.from({ length: d }, () => Math.random() - 0.5);
    for (let it = 0; it < 30; it++) {
      const nw = new Float64Array(d);
      for (const x of X) { let s = 0; for (let i = 0; i < d; i++) s += x[i] * w[i]; for (let i = 0; i < d; i++) nw[i] += s * x[i]; }
      for (const p of comps) { let s = 0; for (let i = 0; i < d; i++) s += nw[i] * p[i]; for (let i = 0; i < d; i++) nw[i] -= s * p[i]; }
      const nrm = Math.sqrt(nw.reduce((a, b) => a + b * b, 0)) || 1; w = nw.map(x => x / nrm);
    }
    comps.push(w);
  }
  return X.map(x => comps.map(w => { let s = 0; for (let i = 0; i < d; i++) s += x[i] * w[i]; return s; }));
}

/* ---------------- モデル ---------------- */
export class JevJa {
  constructor(name) { this.name = name; this.base = new URL(`./models/${name}/`, import.meta.url).href; }
  static optionsOf(q) { return q.type === "choice" ? q.options : q.type === "score" ? q.levels : ["yes", "no"]; }

  async load(onProgress = () => {}) {
    const bufs = {};
    for (const f of FILES) {
      const key = `${this.name}/${f}`;
      let b = await store.get(key);
      if (!b) {
        b = await fetchBuf(this.base + f, (got, total) => onProgress(`${f} をダウンロード中 ${(got / 1e6).toFixed(1)}${total ? " / " + (total / 1e6).toFixed(1) : ""} MB`));
        await store.put(key, b);
      } else onProgress(`${f} を端末内ストレージから読込`);
      bufs[f] = b;
    }
    const txt = b => new TextDecoder().decode(b);
    this.cfg = JSON.parse(txt(bufs["config.json"]));
    this.tok = new PreTrainedTokenizer(JSON.parse(txt(bufs["tokenizer.json"])), JSON.parse(txt(bufs["tokenizer_config.json"])));
    const ep = (this.cfg.prefer_webgpu && navigator.gpu) ? ["webgpu", "wasm"] : ["wasm"];
    onProgress("ONNX セッションを作成中");
    this.enc = await ort.InferenceSession.create(new Uint8Array(bufs["encoder.onnx"]), { executionProviders: ep });
    this.head = await ort.InferenceSession.create(new Uint8Array(bufs["head.onnx"]), { executionProviders: ["wasm"] });
    this.backend = ep[0]; this.vocab = this.tok.model?.vocab; onProgress("");
    return this;
  }

  _raw(text) { return Array.from(this.tok(text, { add_special_tokens: false }).input_ids.data, Number); }
  /* Python(tokenizers) と同じ byte fallback: 語彙に無い文字は UTF-8 バイト列 <0xXX> に */
  ids(text) {
    const unk = this.cfg.special_ids.unk, B = this.cfg.byte_ids; let out = this._raw(text);
    if (!out.includes(unk) || !B) return out;
    this._unkCache ??= new Map();
    const isUnk = ch => { if (!this._unkCache.has(ch)) this._unkCache.set(ch, this._raw(ch).includes(unk)); return this._unkCache.get(ch); };
    out = []; let seg = "";
    for (const ch of text) { if (isUnk(ch)) { if (seg) { out.push(...this._raw(seg)); seg = ""; } for (const b of new TextEncoder().encode(ch)) out.push(B[b]); } else seg += ch; }
    if (seg) out.push(...this._raw(seg)); return out;
  }
  tokenText(id) { const S = this.cfg.special_ids; for (const [k, v] of Object.entries(S)) if (v === id) return `[${k.toUpperCase()}]`; return (this.vocab?.[id] ?? this.tok.decode([id])).replace(/▁/g, "␣"); }

  buildState(state, context) { return (!context || !context.length) ? state : "[根拠]\n" + context.map(c => "- " + c).join("\n") + "\n[状況]\n" + state; }

  encode(state, questions) {
    const S = this.cfg.special_ids; const ids = [S.cls, S.state]; const roles = ["cls", "state"];
    const push = (arr, role) => { for (const i of arr) { ids.push(i); roles.push(role); } };
    push(this.ids(state), "state");
    const spans = {}, marker = {};
    questions.forEach((q, j) => {
      ids.push(S.q); roles.push("q"); let s = ids.length; push(this.ids(q.instructions), `instr${j}`); spans[`q${j}`] = [s, ids.length];
      JevJa.optionsOf(q).forEach((o, k) => { marker[`o${j}_${k}`] = ids.length; ids.push(S.opt); roles.push("opt"); s = ids.length; push(this.ids(o), `opt${j}_${k}`); spans[`o${j}_${k}`] = [s, ids.length]; });
    });
    ids.push(S.sep); roles.push("sep");
    if (ids.length > this.cfg.max_length) throw new Error(`入力長 ${ids.length} が max_length ${this.cfg.max_length} を超えました`);
    return { ids, roles, spans, marker };
  }

  /* 全部入りの推論。opts: {pool:"span"|"marker", T} は出力に反映するだけで、両方の logit を返す */
  async analyze(state, questions, context, opts = {}) {
    const t0 = performance.now();
    const full = this.buildState(state, context);
    const { ids, roles, spans, marker } = this.encode(full, questions);
    const L = ids.length, d = this.cfg.hidden;
    const H = (await this.enc.run({ input_ids: new ort.Tensor("int64", BigInt64Array.from(ids, x => BigInt(x)), [1, L]) })).last_hidden_state.data;
    const tEnc = performance.now();
    const mean = ([s, e]) => { const v = new Float32Array(d); for (let t = s; t < e; t++) for (let i = 0; i < d; i++) v[i] += H[t * d + i]; for (let i = 0; i < d; i++) v[i] /= (e - s); return v; };
    const row = t => H.slice(t * d, (t + 1) * d);
    const runHead = async (u, V, K) => Array.from((await this.head.run({ u: new ort.Tensor("float32", u, [d]), v: new ort.Tensor("float32", V, [K, d]) })).logits.data);
    const qres = [];
    for (let j = 0; j < questions.length; j++) {
      const q = questions[j], opts_ = JevJa.optionsOf(q), K = opts_.length;
      const u = mean(spans[`q${j}`]);
      const Vs = opts_.map((_, k) => mean(spans[`o${j}_${k}`])), Vm = opts_.map((_, k) => row(marker[`o${j}_${k}`]));
      const flat = arr => { const F = new Float32Array(K * d); arr.forEach((v, k) => F.set(v, k * d)); return F; };
      qres.push({ q, labels: opts_, u, Vs, Vm, z_span: await runHead(u, flat(Vs), K), z_marker: await runHead(u, flat(Vm), K), cos_span: Vs.map(v => cosine(u, v)), cos_marker: Vm.map(v => cosine(u, v)) });
    }
    const tHead = performance.now();
    const res = { model: this.name, backend: this.backend, probability_kind: "native", tokens: ids.map((id, i) => ({ id, text: this.tokenText(id), role: roles[i] })), spans, marker, n_tokens: L, timing: { encoder_ms: tEnc - t0, head_ms: tHead - tEnc, total_ms: tHead - t0 }, questions: qres, input: { state, context, questions } };
    res.answers = JevJa.answersFrom(res, opts);
    res.latency_ms = res.timing.total_ms;
    return res;
  }

  /* 途中量から答えを作り直す（T や pool を変えたとき再推論不要） */
  static answersFrom(res, { pool = "span", T = 1, method = "head", cosScale = 20 } = {}) {
    return res.questions.map(({ q, labels, z_span, z_marker, cos_span, cos_marker }) => {
      const z = method === "cos" ? (pool === "marker" ? cos_marker : cos_span).map(c => c * cosScale) : (pool === "marker" ? z_marker : z_span);
      const p = softmax(z, T), am = p.indexOf(Math.max(...p));
      const a = { id: q.id, type: q.type, distribution: { labels, probabilities: p }, logits: z, confidence: p[am] };
      if (q.type === "choice") a.choice = labels[am];
      else if (q.type === "score") { const vals = q.values || labels.map((_, i) => i); a.score = p.reduce((s, pi, i) => s + pi * vals[i], 0); a.level = labels[am]; }
      else { a.p_yes = p[0]; a.noul = p[0] >= 0.5; }
      return a;
    });
  }

  /* 文ベクトル（[STATE] 以降の state トークン平均）。類似・重複検出用。質問なしで 1 forward */
  async embed(text) {
    const S = this.cfg.special_ids; const body = this.ids(text).slice(0, this.cfg.max_length - 4); const ids = [S.cls, S.state, ...body, S.sep];
    const H = (await this.enc.run({ input_ids: new ort.Tensor("int64", BigInt64Array.from(ids, x => BigInt(x)), [1, ids.length]) })).last_hidden_state.data;
    const d = this.cfg.hidden, v = new Float32Array(d); for (let t = 2; t < ids.length - 1; t++) for (let i = 0; i < d; i++) v[i] += H[t * d + i]; const n = Math.sqrt(v.reduce((a, b) => a + b * b, 0)) || 1; for (let i = 0; i < d; i++) v[i] /= n; return v;
  }
  async decide(state, questions, context, opts) { const r = await this.analyze(state, questions, context, opts); return { model: r.model, backend: r.backend, probability_kind: "native", latency_ms: r.latency_ms, tokens: r.n_tokens, answers: r.answers, _raw: r }; }
}

/* 貪欲クラスタリング（cos >= th で同一クラスタ）。コピペ投稿・重複報道の検出 */
export function clusterByCos(vecs, th = 0.9) { const ids = new Array(vecs.length).fill(-1); const reps = []; for (let i = 0; i < vecs.length; i++) { let best = -1, bs = th; for (let c = 0; c < reps.length; c++) { const sc = cosine(vecs[i], vecs[reps[c]]); if (sc >= bs) { bs = sc; best = c; } } if (best < 0) { reps.push(i); ids[i] = reps.length - 1; } else ids[i] = best; } return ids; }

/* ---------------- 評価指標（Python bench/metrics.py と同じ定義） ---------------- */
export function metrics(rows, latencies) {
  const n = rows.length; if (!n) return {};
  const acc = rows.filter(r => r.correct).length / n, brier = rows.reduce((s, r) => s + r.brier, 0) / n;
  const bins = Array.from({ length: 10 }, (_, b) => ({ bin: `${(b / 10).toFixed(1)}-${((b + 1) / 10).toFixed(1)}`, n: 0, conf: 0, acc: 0 }));
  for (const r of rows) { const b = Math.min(9, Math.floor(r.conf * 10)); bins[b].n++; bins[b].conf += r.conf; bins[b].acc += r.correct ? 1 : 0; }
  let ece = 0; for (const b of bins) if (b.n) { b.conf /= b.n; b.acc /= b.n; ece += b.n / n * Math.abs(b.conf - b.acc); }
  const per = {}; for (const r of rows) { (per[r.family] ??= { n: 0, c: 0, mae: 0, ms: 0 }); per[r.family].n++; per[r.family].c += r.correct ? 1 : 0; if (r.mae != null) { per[r.family].mae += r.mae; per[r.family].ms++; } }
  const per_family = Object.fromEntries(Object.entries(per).map(([k, v]) => [k, { n: v.n, accuracy: v.c / v.n, ...(v.ms ? { mae: v.mae / v.ms } : {}) }]));
  const s = [...latencies].sort((a, b) => a - b), pct = p => s.length ? s[Math.min(s.length - 1, Math.floor((s.length - 1) * p))] : NaN;
  return { n_questions: n, accuracy: acc, brier, ece, ece_table: bins.map(b => ({ bin: b.bin, n: b.n, confidence: b.conf, accuracy: b.acc })), latency_p50_ms: pct(0.5), latency_p95_ms: pct(0.95), invalid_rate: 0, per_family };
}
