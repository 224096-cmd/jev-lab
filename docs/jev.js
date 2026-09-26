/* JEV-JA 端末内推論エンジン v2
   - モデルファイルは IndexedDB に保存（Cache API より消されにくい）。以後はネット不要
   - decide() は答えだけでなく、途中量（トークン列・スパン・u/v ベクトル・logit・類似度）も返す
   - span / marker 両方の読み出しを 1 forward で計算し、温度 T はページ側で再適用できる
   Python 側 jev_lab/adapters/jev_ja.py と同じ手順 */
import { PreTrainedTokenizer, AutoTokenizer, AutoModel, Tensor as HfTensor, env as hfenv } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.5.2/dist/transformers.min.js";
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.webgpu.min.mjs";
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
ort.env.wasm.proxy = true;                                   // 推論を Worker で実行し UI を止めない
ort.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 1);

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

const FILES = ["config.json", "tokenizer.json", "tokenizer_config.json", "head.onnx", "head.bin", "encoder.onnx"];

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

/* ---------------- 判断ヘッド（純 JS 実装。端末内で再学習できる） ----------------
   z = w2ᵀ tanh(W1 [u;v;u⊙v] + b1) + b2   W1: hid×3d */
export class HeadJS {
  /* version 2: LayerNorm(u), LayerNorm(v) → [u;v;u⊙v] → MLP。version 1 は正規化なし（旧チェックポイント） */
  constructor(d, hid, buf, version = 2) {
    this.d = d; this.hid = hid; this.version = version; const f = buf ? new Float32Array(buf) : null; let o = 0;
    const take = n => { const a = f ? f.slice(o, o + n) : new Float32Array(n); o += n; return a; };
    if (version >= 2) { this.gu = take(d); this.bu = take(d); this.gv = take(d); this.bv = take(d); if (!f) { this.gu.fill(1); this.gv.fill(1); } }
    this.W1 = take(hid * 3 * d); this.b1 = take(hid); this.w2 = take(hid); this.b2 = f ? take(1)[0] : 0;
    this.origin = "export";
  }
  static fromJSON(j) { const h = new HeadJS(j.d, j.hid, null, j.version ?? 1); for (const k of ["gu", "bu", "gv", "bv", "W1", "b1", "w2"]) if (j[k]) h[k] = Float32Array.from(j[k]); h.b2 = j.b2; h.origin = j.origin || "custom"; h.trained_on = j.trained_on; return h; }
  toJSON() { const j = { version: this.version, d: this.d, hid: this.hid, W1: Array.from(this.W1), b1: Array.from(this.b1), w2: Array.from(this.w2), b2: this.b2, origin: this.origin, trained_on: this.trained_on }; if (this.version >= 2) for (const k of ["gu", "bu", "gv", "bv"]) j[k] = Array.from(this[k]); return j; }
  clone() { return HeadJS.fromJSON(this.toJSON()); }
  static ln(x, g, b) { const n = x.length; let m = 0; for (let i = 0; i < n; i++) m += x[i]; m /= n; let v = 0; for (let i = 0; i < n; i++) v += (x[i] - m) ** 2; v /= n; const s = 1 / Math.sqrt(v + 1e-5); const y = new Float32Array(n); for (let i = 0; i < n; i++) y[i] = (x[i] - m) * s * g[i] + b[i]; return y; }
  _x(u, v) { const d = this.d; if (this.version >= 2) { u = HeadJS.ln(u, this.gu, this.bu); v = HeadJS.ln(v, this.gv, this.bv); } const x = new Float32Array(3 * d); for (let i = 0; i < d; i++) { x[i] = u[i]; x[d + i] = v[i]; x[2 * d + i] = u[i] * v[i]; } return x; }
  _fwd(x) { const { hid, d } = this, D = 3 * d, h = new Float32Array(hid); for (let j = 0; j < hid; j++) { let s = this.b1[j]; const off = j * D; for (let i = 0; i < D; i++) s += this.W1[off + i] * x[i]; h[j] = Math.tanh(s); } let z = this.b2; for (let j = 0; j < hid; j++) z += this.w2[j] * h[j]; return { z, h }; }
  logits(u, V) { return V.map(v => this._fwd(this._x(u, v)).z); }
  /* 学習（LN のパラメータは固定、W1/b1/w2/b2 を Adam で更新）。examples = [{u, V, y}] */
  train(examples, { epochs = 8, lr = 1e-3, lam = 1.0, onEpoch = () => {} } = {}) {
    const { hid, d } = this, D = 3 * d; const P = { W1: this.W1, b1: this.b1, w2: this.w2 }; const m = {}, v = {}; for (const k in P) { m[k] = new Float32Array(P[k].length); v[k] = new Float32Array(P[k].length); } let mb2 = 0, vb2 = 0, t = 0; const b1 = 0.9, b2 = 0.999, eps = 1e-8;
    const pre = examples.map(ex => ({ xs: ex.V.map(vv => this._x(ex.u, vv)), y: ex.y }));   // LN 済み入力をキャッシュ
    for (let ep = 0; ep < epochs; ep++) {
      let tot = 0, correct = 0; const order = pre.map((_, i) => i).sort(() => Math.random() - 0.5);
      for (const idx of order) {
        const ex = pre[idx], K = ex.xs.length, fw = ex.xs.map(x => this._fwd(x)); const z = fw.map(f => f.z), p = softmax(z, 1);
        const am = p.indexOf(Math.max(...p)); if (am === ex.y) correct++; tot += -Math.log(p[ex.y] + 1e-9) + lam * p.reduce((s, pk, k) => s + (pk - (k === ex.y ? 1 : 0)) ** 2, 0);
        const g = new Float32Array(K); const q = p.map((pk, k) => pk - (k === ex.y ? 1 : 0)); const dot = q.reduce((s2, qj, j) => s2 + qj * p[j], 0); for (let k = 0; k < K; k++) g[k] = q[k] + lam * 2 * p[k] * (q[k] - dot);
        const gW1 = new Float32Array(P.W1.length), gb1 = new Float32Array(hid), gw2 = new Float32Array(hid); let gb2 = 0;
        for (let k = 0; k < K; k++) { const gz = g[k]; if (!gz) continue; const { h } = fw[k], x = ex.xs[k]; gb2 += gz; for (let j = 0; j < hid; j++) { gw2[j] += gz * h[j]; const gh = gz * this.w2[j] * (1 - h[j] * h[j]); if (!gh) continue; gb1[j] += gh; const off = j * D; for (let i = 0; i < D; i++) gW1[off + i] += gh * x[i]; } }
        t++; const step = (param, grad, mm, vv2) => { const c1 = 1 - b1 ** t, c2 = 1 - b2 ** t; for (let i = 0; i < param.length; i++) { mm[i] = b1 * mm[i] + (1 - b1) * grad[i]; vv2[i] = b2 * vv2[i] + (1 - b2) * grad[i] * grad[i]; param[i] -= lr * (mm[i] / c1) / (Math.sqrt(vv2[i] / c2) + eps); } };
        step(P.W1, gW1, m.W1, v.W1); step(P.b1, gb1, m.b1, v.b1); step(P.w2, gw2, m.w2, v.w2); mb2 = b1 * mb2 + (1 - b1) * gb2; vb2 = b2 * vb2 + (1 - b2) * gb2 * gb2; this.b2 -= lr * (mb2 / (1 - b1 ** t)) / (Math.sqrt(vb2 / (1 - b2 ** t)) + eps);
      }
      onEpoch(ep, tot / examples.length, correct / examples.length);
    }
    this.origin = "custom"; return this;
  }
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
    this.headExport = new HeadJS(this.cfg.hidden, this.cfg.head?.hidden || 512, bufs["head.bin"], this.cfg.head?.version || 1);
    const custom = await store.get(`headjs:${this.name}`); this.headCustom = custom ? HeadJS.fromJSON(custom) : null;
    this.backend = ep[0]; this.vocab = this.tok.model?.vocab; this.kind = "jev_ja"; onProgress("");
    return this;
  }
  /* 端末内で学習したヘッドを使うか（null なら ONNX の書き出し時ヘッド） */
  get activeHead() { return this.headCustom; }
  async saveCustomHead(h, trained_on) { h.trained_on = trained_on; this.headCustom = h; await store.put(`headjs:${this.name}`, h.toJSON()); }
  async resetCustomHead() { this.headCustom = null; await store.del(`headjs:${this.name}`); }

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
    const runHead = async (u, V, K) => this.headCustom ? this.headCustom.logits(u, Array.from({ length: K }, (_, k) => V.subarray(k * d, (k + 1) * d))) : Array.from((await this.head.run({ u: new ort.Tensor("float32", u, [d]), v: new ort.Tensor("float32", V, [K, d]) })).logits.data);
    const qres = [];
    for (let j = 0; j < questions.length; j++) {
      const q = questions[j], opts_ = JevJa.optionsOf(q), K = opts_.length;
      const u = mean(spans[`q${j}`]);
      const Vs = opts_.map((_, k) => mean(spans[`o${j}_${k}`])), Vm = opts_.map((_, k) => row(marker[`o${j}_${k}`]));
      const flat = arr => { const F = new Float32Array(K * d); arr.forEach((v, k) => F.set(v, k * d)); return F; };
      qres.push({ q, labels: opts_, u, Vs, Vm, z_span: await runHead(u, flat(Vs), K), z_marker: await runHead(u, flat(Vm), K), cos_span: Vs.map(v => cosine(u, v)), cos_marker: Vm.map(v => cosine(u, v)) });
    }
    const tHead = performance.now();
    const res = { model: this.name, backend: this.backend, probability_kind: "native", head: this.headCustom ? "custom" : "export", tokens: ids.map((id, i) => ({ id, text: this.tokenText(id), role: roles[i] })), spans, marker, n_tokens: L, timing: { encoder_ms: tEnc - t0, head_ms: tHead - tEnc, total_ms: tHead - t0 }, questions: qres, input: { state, context, questions } };
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


/* ================= 既存の Jev 級モデル =================
   kind:
     "crossenc"      : 日本語 Jev cross-encoder（例 argos1111/modernbert-ja-310m-jev）。docs/models/<name>/ に ONNX（int8、分割）を置く。
                       「質問: …\n状況: …」と候補のペアを 1 本の系列で読み 1 スコア → 候補間 softmax。Noul は候補 ["true","false"]
     "open-jev-onnx" : Kotoba Labs open-jev の Transformers.js 版（HF から取得）。state + 全質問を 1 系列、(質問,選択肢) ペアごとに logit */
export class CrossEncJev {
  constructor(entry) { this.name = entry.name; this.cfg = entry; this.kind = "crossenc"; this.probability_kind = "native"; this.base = new URL(`./models/${entry.name}/`, import.meta.url).href; }
  async load(onProgress = () => {}) {
    const getf = async (f) => { const key = `${this.name}/${f}`; let b = await store.get(key); if (!b) { b = await fetchBuf(this.base + f, (got, total) => onProgress(`${f} ${(got / 1e6).toFixed(1)}${total ? " / " + (total / 1e6).toFixed(1) : ""} MB`)); await store.put(key, b); } return b; };
    const txt = b => new TextDecoder().decode(b);
    this.cfg = { ...this.cfg, ...JSON.parse(txt(await getf("config.json"))) };
    this.tok = new PreTrainedTokenizer(JSON.parse(txt(await getf("tokenizer.json"))), JSON.parse(txt(await getf("tokenizer_config.json"))));
    const ext = []; for (const f of (this.cfg.external_data || [])) ext.push({ path: f, data: new Uint8Array(await getf(f)) });
    onProgress("ONNX セッションを作成中"); this.sess = await ort.InferenceSession.create(new Uint8Array(await getf("model.onnx")), { executionProviders: ["wasm"], externalData: ext });
    this.backend = "wasm"; this.padId = this.tok.model?.tokens_to_ids?.get(this.tok.pad_token) ?? 3; onProgress(""); return this;
  }
  ids(t) { return Array.from(this.tok(t, { add_special_tokens: false }).input_ids.data, Number); }
  async decide(state, questions, context, opts = {}) {
    const t0 = performance.now(); const full = (!context || !context.length) ? state : "[根拠]\n" + context.map(c => "- " + c).join("\n") + "\n[状況]\n" + state; const answers = []; const { bos_id: B, eos_id: E, max_length: ML } = this.cfg;
    for (const [qi, q] of questions.entries()) { opts.onProgress?.(`質問 ${qi + 1}/${questions.length} を判定中（${this.name}）`);
      const labels = JevJa.optionsOf(q); const cands = q.type === "noul" ? this.cfg.noul_options : labels; const a = this.ids(this.cfg.prompt.replace("{question}", q.instructions).replace("{state}", full));
      const seqs = cands.map(c => { const b = this.ids(c); const A = a.slice(0, Math.max(8, ML - b.length - 4)); return [B, ...A, E, B, ...b, E]; }); const L = Math.max(...seqs.map(s => s.length));
      const ids = new BigInt64Array(seqs.length * L), mask = new BigInt64Array(seqs.length * L); seqs.forEach((s, i) => s.forEach((x, j) => { ids[i * L + j] = BigInt(x); mask[i * L + j] = 1n; })); for (let i = 0; i < seqs.length; i++) for (let j = seqs[i].length; j < L; j++) ids[i * L + j] = BigInt(this.padId);
      const out = await this.sess.run({ input_ids: new ort.Tensor("int64", ids, [seqs.length, L]), attention_mask: new ort.Tensor("int64", mask, [seqs.length, L]) }); const z = Array.from(out.logits.data).filter((_, i) => out.logits.dims[1] ? i % out.logits.dims[1] === 0 : true);
      const p = softmax(z, opts.T || 1); answers.push(mkAnswer(q, labels, p, z));
    }
    return { model: this.name, backend: this.backend, probability_kind: "native", latency_ms: performance.now() - t0, tokens: null, answers };
  }
}
export class OpenJevOnnx {
  constructor(entry) { this.name = entry.name; this.cfg = entry; this.kind = "open-jev-onnx"; this.probability_kind = "native"; }
  async load(onProgress = () => {}) {
    hfenv.allowRemoteModels = true; hfenv.allowLocalModels = false; hfenv.useBrowserCache = true; const gpu = !!navigator.gpu; const dtype = gpu ? (this.cfg.dtype_webgpu || "q4f16") : (this.cfg.dtype || "q4");
    const pc = p => { if (p.status === "progress") onProgress(`${p.file.split("/").pop()} ${(p.loaded / 1e6).toFixed(0)}/${(p.total / 1e6).toFixed(0)} MB`); };
    this.tok = await AutoTokenizer.from_pretrained(this.cfg.hf_id, { progress_callback: pc }); this.model = await AutoModel.from_pretrained(this.cfg.hf_id, { dtype, device: gpu ? "webgpu" : "wasm", progress_callback: pc });
    hfenv.allowLocalModels = true; this.backend = gpu ? "webgpu" : "wasm"; const e = t => Array.from(this.tok(t, { add_special_tokens: false }).input_ids.data, Number); [this.CLS, this.SEP, this.STATE, this.Q, this.OPT] = ["[CLS]", "[SEP]", "[STATE]", "[Q]", "[OPT]"].map(m => e(m)[0]); this.enc = e; onProgress(""); return this;
  }
  async decide(state, questions, context, opts = {}) {
    const t0 = performance.now(); const full = (!context || !context.length) ? state : context.join("\n") + "\n" + state;
    const tokens = [this.CLS, this.STATE, ...this.enc(full).slice(0, 256)]; const seg = tokens.map(() => -1); const pairQ = [], pairOpt = [], groups = []; const totalPairs = questions.reduce((n, q) => n + JevJa.optionsOf(q).length, 0);
    questions.forEach((q, qi) => { const text = this.enc(q.instructions); tokens.push(this.Q, ...text); seg.push(-1, ...text.map(() => totalPairs + qi)); const opts_ = q.type === "noul" ? ["no", "yes"] : JevJa.optionsOf(q); groups.push(opts_.map(o => { const ot = this.enc(o); tokens.push(this.OPT, ...ot); seg.push(-1, ...ot.map(() => pairOpt.length)); pairQ.push(totalPairs + qi); pairOpt.push(pairOpt.length); return pairOpt.length - 1; })); });
    tokens.push(this.SEP); seg.push(-1); const i64 = (v, dims) => new HfTensor("int64", BigInt64Array.from(v, BigInt), dims);
    const { logits } = await this.model({ input_ids: i64(tokens, [1, tokens.length]), attention_mask: i64(tokens.map(() => 1), [1, tokens.length]), seg: i64(seg, [1, seg.length]), pair_q: i64(pairQ, [1, pairQ.length]), pair_opt: i64(pairOpt, [1, pairOpt.length]) });
    const scores = Array.from(logits.to("float32").data); const T = (opts.T || 1) * (this.cfg.temperature || 1.05);
    const answers = questions.map((q, i) => { const labels = JevJa.optionsOf(q); let z = groups[i].map(p => scores[p]); if (q.type === "noul") z = [z[1], z[0]]; /* [no,yes] → [yes,no] */ const p = softmax(z, T); return mkAnswer(q, labels, p, z); });
    return { model: this.name, backend: this.backend, probability_kind: "native", latency_ms: performance.now() - t0, tokens: tokens.length, answers };
  }
}
function mkAnswer(q, labels, p, z) { const am = p.indexOf(Math.max(...p)); const a = { id: q.id, type: q.type, distribution: { labels, probabilities: p }, logits: z, confidence: p[am] }; if (q.type === "choice") a.choice = labels[am]; else if (q.type === "score") { const vals = q.values || labels.map((_, i) => i); a.score = p.reduce((s, pi, i) => s + pi * vals[i], 0); a.level = labels[am]; } else { a.p_yes = p[0]; a.noul = p[0] >= 0.5; } return a; }
export async function isStoredAny(name, files) { const ks = new Set(await store.keys()); return files.every(f => ks.has(`${name}/${f}`)); }

/* registry + index からモデル一覧を作り、名前でロードする */
export async function listAllModels() {
  const out = [];
  try { const idx = await (await fetch(new URL("./models/index.json", import.meta.url))).json(); for (const m of idx.models) out.push({ ...m, kind: m.kind || "jev_ja", group: m.kind === "crossenc" ? "既存 Jev（日本語・端末内）" : "JEV-JA（自作・端末内・学習可）" }); } catch { }
  try { const reg = await (await fetch(new URL("./models/registry.json", import.meta.url))).json(); for (const m of reg.models) out.push({ ...m, group: m.group || "既存 Jev（HF から取得）" }); } catch { }
  try { const custom = await store.get("registry:custom"); if (custom) for (const m of custom) out.push({ ...m, group: "追加したモデル（この端末）" }); } catch { }
  return out;
}
export async function loadModelByName(name, onProgress) {
  const all = await listAllModels(); const e = all.find(m => m.name === name); if (!e) throw new Error("unknown model: " + name);
  if (e.kind === "jev_ja") return new JevJa(name).load(onProgress);
  if (e.kind === "crossenc") return new CrossEncJev(e).load(onProgress);
  if (e.kind === "open-jev-onnx") return new OpenJevOnnx(e).load(onProgress);
  throw new Error("unsupported kind: " + e.kind);
}
