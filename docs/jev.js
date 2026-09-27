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
  async open() { if (this.db) return this.db; return this.db = await new Promise((ok, ng) => { const r = indexedDB.open("jev-lab", 3); r.onupgradeneeded = () => { const db = r.result; if (!db.objectStoreNames.contains("files")) db.createObjectStore("files"); if (!db.objectStoreNames.contains("experiments")) db.createObjectStore("experiments", { keyPath: "id" }); if (!db.objectStoreNames.contains("cache")) db.createObjectStore("cache"); }; r.onsuccess = () => ok(r.result); r.onerror = () => ng(r.error); }); },
  async get(k) { const db = await this.open(); return new Promise((ok, ng) => { const r = db.transaction("files").objectStore("files").get(k); r.onsuccess = () => ok(r.result); r.onerror = () => ng(r.error); }); },
  async put(k, v) { const db = await this.open(); return new Promise((ok, ng) => { const r = db.transaction("files", "readwrite").objectStore("files").put(v, k); r.onsuccess = () => ok(); r.onerror = () => ng(r.error); }); },
  async del(k) { const db = await this.open(); return new Promise((ok, ng) => { const r = db.transaction("files", "readwrite").objectStore("files").delete(k); r.onsuccess = () => ok(); r.onerror = () => ng(r.error); }); },
  async keys() { const db = await this.open(); return new Promise((ok, ng) => { const r = db.transaction("files").objectStore("files").getAllKeys(); r.onsuccess = () => ok(r.result); r.onerror = () => ng(r.error); }); },
  /* API 応答キャッシュ（quota.js） */
  async cacheGet(k) { const db = await this.open(); return new Promise((ok, ng) => { const r = db.transaction("cache").objectStore("cache").get(k); r.onsuccess = () => ok(r.result); r.onerror = () => ng(r.error); }); },
  async cachePut(k, v) { const db = await this.open(); return new Promise((ok, ng) => { const r = db.transaction("cache", "readwrite").objectStore("cache").put(v, k); r.onsuccess = () => ok(); r.onerror = () => ng(r.error); }); },
  async cacheClear() { const db = await this.open(); return new Promise((ok, ng) => { const r = db.transaction("cache", "readwrite").objectStore("cache").clear(); r.onsuccess = () => ok(); r.onerror = () => ng(r.error); }); },
  async cacheCount() { const db = await this.open(); return new Promise((ok, ng) => { const r = db.transaction("cache").objectStore("cache").count(); r.onsuccess = () => ok(r.result); r.onerror = () => ng(r.error); }); },
  /* 実験ノート */
  async expPut(e) { const db = await this.open(); return new Promise((ok, ng) => { const r = db.transaction("experiments", "readwrite").objectStore("experiments").put(e); r.onsuccess = () => ok(); r.onerror = () => ng(r.error); }); },
  async expAll() { const db = await this.open(); return new Promise((ok, ng) => { const r = db.transaction("experiments").objectStore("experiments").getAll(); r.onsuccess = () => ok(r.result); r.onerror = () => ng(r.error); }); },
  async expDel(id) { const db = await this.open(); return new Promise((ok, ng) => { const r = db.transaction("experiments", "readwrite").objectStore("experiments").delete(id); r.onsuccess = () => ok(); r.onerror = () => ng(r.error); }); },
};

const FILES = ["config.json", "tokenizer.json", "tokenizer_config.json", "head.onnx", "head.bin", "encoder.onnx"];

async function fetchBuf(url, onProgress, tries = 3) {
  let lastErr;
  for (let t = 1; t <= tries; t++) {
    try {
      const r = await fetch(url, { cache: t === 1 ? "default" : "reload" }); if (!r.ok) throw new Error(`HTTP ${r.status}: ${url.split("/").slice(-2).join("/")}`);
      /* GitHub Pages は gzip で返すので Content-Length は圧縮後のサイズ。長さ検証は非圧縮のときだけ行う（v3.1 修正：ここで全モデルが「途中で切れた」扱いになっていた） */
      const enc = (r.headers.get("content-encoding") || "").toLowerCase(); const total = (!enc || enc === "identity") ? (+r.headers.get("content-length") || 0) : 0; const reader = r.body.getReader(); const chunks = []; let got = 0;
      for (;;) { const { done, value } = await reader.read(); if (done) break; chunks.push(value); got += value.length; onProgress?.(got, total); }
      if (total && got < total) throw new Error(`ダウンロードが途中で切れました（${got}/${total} bytes）`);
      const out = new Uint8Array(got); let o = 0; for (const c of chunks) { out.set(c, o); o += c.length; } return out.buffer;
    } catch (e) { lastErr = e; if (t < tries) { onProgress?.(0, 0, `再試行 ${t}/${tries - 1}: ${e.message}`); await new Promise(r => setTimeout(r, 1500 * t)); } }
  }
  throw lastErr;
}
/* IndexedDB に無ければ取得して保存（完全に取れたものだけ保存）。壊れたキャッシュ（0 byte）は捨てる */
export async function getModelFile(name, f, base, onProgress) {
  const key = `${name}/${f}`; let b = null; try { b = await store.get(key); } catch { }
  if (b && b.byteLength > 0) { onProgress(`${f} を端末内ストレージから読込`); return b; }
  b = await fetchBuf(base + f, (got, total, msg) => onProgress(msg || `${f} をダウンロード中 ${(got / 1e6).toFixed(1)}${total ? " / " + (total / 1e6).toFixed(1) : ""} MB`));
  if (!b.byteLength) throw new Error(`${f} が空です（配置ミスか未 push）`);
  try { await store.put(key, b); } catch (e) { onProgress(`${f}: 端末内に保存できず（容量不足？）。今回だけメモリで使用`); }
  return b;
}
/* ONNX セッション作成。Worker（proxy）で失敗したらメインスレッドで再試行 */
export async function createSession(buf, opts) {
  try { return await ort.InferenceSession.create(buf, opts); }
  catch (e) { if (ort.env.wasm.proxy) { console.warn("proxy 失敗 → メインスレッドで再試行", e); ort.env.wasm.proxy = false; try { return await ort.InferenceSession.create(buf, opts); } catch (e2) { throw friendly(e2); } } throw friendly(e); }
}
const friendly = e => { const m = String(e?.message || e); if (/memory|allocation|out of memory|RangeError/i.test(m)) return new Error("メモリ不足でモデルを展開できません。小さいモデル（jev_ja_30m / jevlet_33m）を選ぶか、他のタブを閉じてください。" + " [" + m.slice(0, 80) + "]"); if (/wasm|WebAssembly|fetch|Failed to load/i.test(m)) return new Error("実行エンジン（ONNX Runtime）の読み込みに失敗しました。通信状態を確認して「読み込む」をもう一度押してください。 [" + m.slice(0, 80) + "]"); return e; };

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
    for (const f of FILES) bufs[f] = await getModelFile(this.name, f, this.base, onProgress);
    const txt = b => new TextDecoder().decode(b);
    this.cfg = JSON.parse(txt(bufs["config.json"]));
    this.tok = new PreTrainedTokenizer(JSON.parse(txt(bufs["tokenizer.json"])), JSON.parse(txt(bufs["tokenizer_config.json"])));
    const ep = (this.cfg.prefer_webgpu && navigator.gpu) ? ["webgpu", "wasm"] : ["wasm"];
    onProgress("ONNX セッションを作成中");
    this.enc = await createSession(new Uint8Array(bufs["encoder.onnx"]), { executionProviders: ep });
    this.head = await createSession(new Uint8Array(bufs["head.onnx"]), { executionProviders: ["wasm"] });
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
    const runHead = async (u, V, K) => this.headCustom ? this.headCustom.logits(u, Array.from({ length: K }, (_, k) => V.subarray(k * d, (k + 1) * d))) : Array.from((await this.head.run({ u: new ort.Tensor("float32", u.slice(), [d]), v: new ort.Tensor("float32", V, [K, d]) })).logits.data)   /* proxy(Worker) では入力バッファが転送されるので複製を渡す */;
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
    const getf = f => getModelFile(this.name, f, this.base, onProgress);
    const txt = b => new TextDecoder().decode(b);
    this.cfg = { ...this.cfg, ...JSON.parse(txt(await getf("config.json"))) };
    this.tok = new PreTrainedTokenizer(JSON.parse(txt(await getf("tokenizer.json"))), JSON.parse(txt(await getf("tokenizer_config.json"))));
    const ext = []; for (const f of (this.cfg.external_data || [])) ext.push({ path: f, data: new Uint8Array(await getf(f)) });
    onProgress("ONNX セッションを作成中"); this.sess = await createSession(new Uint8Array(await getf("model.onnx")), { executionProviders: ["wasm"], externalData: ext });
    this.backend = "wasm"; this.padId = this.tok.model?.tokens_to_ids?.get(this.tok.pad_token) ?? 3; onProgress(""); return this;
  }
  ids(t) { return Array.from(this.tok(t, { add_special_tokens: false }).input_ids.data, Number); }
  async decide(state, questions, context, opts = {}) {
    const t0 = performance.now(); const full = (!context || !context.length) ? state : "[根拠]\n" + context.map(c => "- " + c).join("\n") + "\n[状況]\n" + state; const answers = []; const { bos_id: B, eos_id: E, max_length: ML } = this.cfg;
    const C = this.cfg.cls_id ?? B, S = this.cfg.sep_id ?? E; const decisionFirst = this.cfg.pair_order === "decision_first";
    for (const [qi, q] of questions.entries()) { opts.onProgress?.(`質問 ${qi + 1}/${questions.length} を判定中（${this.name}）`);
      const labels = JevJa.optionsOf(q); const cands = q.type === "noul" ? this.cfg.noul_options : labels; const render = c => this.cfg.prompt.replace("{type}", q.type).replace("{question}", q.instructions).replace("{option}", c).replace("{options}", cands.join(" ; ")).replace("{state}", full);
      let seqs; if (decisionFirst) { const t = this.ids(full); seqs = cands.map(c => { const d = this.ids(render(c)).slice(0, 200); const T = t.slice(0, Math.max(8, ML - d.length - 4)); return [C, ...d, S, ...T, S]; }); } else { const a = this.ids(render("")); seqs = cands.map(c => { const b = this.ids(c); const A = a.slice(0, Math.max(8, ML - b.length - 4)); return [B, ...A, E, B, ...b, E]; }); } const L = Math.max(...seqs.map(s => s.length));
      const ids = new BigInt64Array(seqs.length * L), mask = new BigInt64Array(seqs.length * L); seqs.forEach((s, i) => s.forEach((x, j) => { ids[i * L + j] = BigInt(x); mask[i * L + j] = 1n; })); for (let i = 0; i < seqs.length; i++) for (let j = seqs[i].length; j < L; j++) ids[i * L + j] = BigInt(this.padId);
      const out = await this.sess.run({ input_ids: new ort.Tensor("int64", ids, [seqs.length, L]), attention_mask: new ort.Tensor("int64", mask, [seqs.length, L]) }); const z = Array.from(out.logits.data).filter((_, i) => out.logits.dims[1] ? i % out.logits.dims[1] === 0 : true);
      let p; if (this.cfg.score === "sigmoid") { const T = opts.T || 1; const sg = z.map(v => 1 / (1 + Math.exp(-v / T))); const sum = sg.reduce((a, b) => a + b, 0) || 1; p = sg.map(v => v / sum); } else p = softmax(z, opts.T || 1); answers.push(mkAnswer(q, labels, p, z));
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

/* "laya": Laya（convaiinnovations/laya-multilingual 等）。docs/models/<name>/ に export_laya.py で置いた int8 ONNX。
   系列 [CLS] "{type} question: {ins}" [SEP] [MASK] opt0 [MASK] opt1 … [SEP] state [SEP] → [MASK] 位置ごとに 1 logit（laya.common.build_sequence と同じ） */
export class LayaJev {
  constructor(entry) { this.name = entry.name; this.cfg = entry; this.kind = "laya"; this.probability_kind = "native"; this.base = new URL(`./models/${entry.name}/`, import.meta.url).href; }
  async load(onProgress = () => {}) {
    const getf = f => getModelFile(this.name, f, this.base, onProgress);
    const txt = b => new TextDecoder().decode(b);
    this.cfg = { ...this.cfg, ...JSON.parse(txt(await getf("config.json"))) };
    onProgress("tokenizer を準備中"); const tj = JSON.parse(txt(await getf("tokenizer.json")));
    // Python(tokenizers) の Metaspace prepend_scheme:"always" を Transformers.js で再現するには add_prefix_space が要る（無いと先頭の ▁ が付かず id がずれる）
    const fixMeta = pt => { if (!pt) return; if (pt.type === "Metaspace" && pt.add_prefix_space == null) pt.add_prefix_space = (pt.prepend_scheme ?? "always") !== "never"; (pt.pretokenizers || []).forEach(fixMeta); }; fixMeta(tj.pre_tokenizer);
    this.tok = new PreTrainedTokenizer(tj, JSON.parse(txt(await getf("tokenizer_config.json"))));
    const ext = []; for (const f of (this.cfg.external_data || [])) ext.push({ path: f, data: new Uint8Array(await getf(f)) });
    onProgress("ONNX セッションを作成中"); this.sess = await createSession(new Uint8Array(await getf("model.onnx")), { executionProviders: ["wasm"], externalData: ext });
    this.backend = "wasm"; onProgress(""); return this;
  }
  ids(t, max) { const o = { add_special_tokens: false }; if (max) { o.truncation = true; o.max_length = max; } return Array.from(this.tok(t, o).input_ids.data, Number).slice(0, max || 1e9); }
  /* Python の render_options と同じ文字列 */
  static renderOptions(q) { if (q.type === "choice") return q.options; if (q.type === "score") return q.levels.map((c, i) => `level ${i}: ${c}`); return ["false: no, the statement does not hold", "true: yes, the statement holds"]; }
  buildSequence(stateIds, q) {
    const { cls_id: CLS, sep_id: SEP, mask_id: MASK, max_len: ML = 1024, head_max_len: HL = 256 } = this.cfg; const mt = this.tok.mask_token || "[MASK]";
    const head = this.ids(`${q.type} question: ${String(q.instructions).replaceAll(mt, " ")}`); let opt = LayaJev.renderOptions(q).map(o => [MASK, ...this.ids(" " + o.replaceAll(mt, " "), 48)]);
    let budget = HL - opt.reduce((s, o) => s + o.length, 0); if (budget < 16) { const per = Math.max(4, Math.floor((HL - 16) / Math.max(1, opt.length))); opt = opt.map(o => o.slice(0, per)); budget = HL - opt.reduce((s, o) => s + o.length, 0); }
    const ids = [CLS, ...head.slice(0, Math.max(8, budget)), SEP]; const markers = []; for (const o of opt) { markers.push(ids.length); ids.push(...o); } ids.push(SEP);
    const room = Math.max(0, ML - ids.length - 1); ids.push(...stateIds.slice(0, room), SEP); return { ids: ids.slice(0, ML), markers: markers.filter(m => m < ML) };
  }
  async decide(state, questions, context, opts = {}) {
    const t0 = performance.now(); const full = (!context || !context.length) ? state : "[根拠]\n" + context.map(c => "- " + c).join("\n") + "\n[状況]\n" + state; const mt = this.tok.mask_token || "[MASK]";
    const stateIds = this.ids(full.replaceAll(mt, " ")); const items = questions.map(q => ({ ...this.buildSequence(stateIds, q), qtype: { choice: 0, score: 1, noul: 2 }[q.type] }));
    const n = items.length, L = Math.max(...items.map(i => i.ids.length)), K = Math.max(...items.map(i => i.markers.length)); const PAD = this.cfg.pad_id ?? 0;
    const ids = new BigInt64Array(n * L).fill(BigInt(PAD)), att = new BigInt64Array(n * L), mpos = new BigInt64Array(n * K), mmask = new Uint8Array(n * K), qt = new BigInt64Array(n);
    items.forEach((it, i) => { it.ids.forEach((x, j) => { ids[i * L + j] = BigInt(x); att[i * L + j] = 1n; }); it.markers.forEach((m, k) => { mpos[i * K + k] = BigInt(m); mmask[i * K + k] = 1; }); qt[i] = BigInt(it.qtype); });
    opts.onProgress?.(`${questions.length} 問を 1 回で判定中（${this.name}, ${L} tok）`);
    const out = await this.sess.run({ input_ids: new ort.Tensor("int64", ids, [n, L]), attention_mask: new ort.Tensor("int64", att, [n, L]), marker_pos: new ort.Tensor("int64", mpos, [n, K]), marker_mask: new ort.Tensor("bool", mmask, [n, K]), qtype: new ort.Tensor("int64", qt, [n]) });
    const Z = Array.from(out.logits.data), Kout = out.logits.dims[1]; const temp = this.cfg.temperature || [1, 1, 1], tbo = this.cfg.temperature_by_options || {};
    const answers = questions.map((q, i) => { const k = items[i].markers.length; let z = Z.slice(i * Kout, i * Kout + k); const size = k <= 2 ? "2" : k <= 5 ? "3-5" : k <= 10 ? "6-10" : "11+"; const tb = tbo[`${q.type}:${size}`] ?? temp[items[i].qtype] ?? 1; const T = Math.min(5, Math.max(0.5, +tb || 1)) * (opts.T || 1);
      const labels = JevJa.optionsOf(q); if (q.type === "noul") z = [z[1], z[0]]; /* [false,true] → [yes,no] */ const p = softmax(z, T); return mkAnswer(q, labels, p, z); });
    return { model: this.name, backend: this.backend, probability_kind: "native", latency_ms: performance.now() - t0, tokens: L, answers };
  }
}

/* "jevlet": Jevlet v6（bge-small 33.5M、英語、MIT）。docs/models/<name>/ に export_jevlet.py で置いた int8 ONNX（約 34 MB、スマホ可）。
   系列 [CLS][STATE] state | [QUESTION] q [OPTION] o [END_OPTION]… [DECIDE] | …（jevlet.pretrained.PretrainedCollator と同じ packing、block_bidir）
   分岐は state と自分だけを見る 2 次元マスク、位置は state の末尾から再開。logits = (K·mean(option)) · (Q·decide) / √w、種類別温度 */
export class JevletJev {
  constructor(entry) { this.name = entry.name; this.cfg = entry; this.kind = "jevlet"; this.probability_kind = "native"; this.base = new URL(`./models/${entry.name}/`, import.meta.url).href; }
  async load(onProgress = () => {}) {
    const getf = f => getModelFile(this.name, f, this.base, onProgress);
    const txt = b => new TextDecoder().decode(b);
    this.cfg = { ...this.cfg, ...JSON.parse(txt(await getf("config.json"))) };
    this.tok = new PreTrainedTokenizer(JSON.parse(txt(await getf("tokenizer.json"))), JSON.parse(txt(await getf("tokenizer_config.json"))));
    const hb = new Float32Array(await getf("head.bin")); const w = this.cfg.hidden; this.Wq = hb.slice(0, w * w); this.Wk = hb.slice(w * w, 2 * w * w);
    onProgress("ONNX セッションを作成中"); this.sess = await createSession(new Uint8Array(await getf("encoder.onnx")), { executionProviders: ["wasm"] }); this.backend = "wasm"; onProgress(""); return this;
  }
  ids(t) { return Array.from(this.tok(t, { add_special_tokens: false }).input_ids.data, Number); }
  pack(state, questions) {
    const S = this.cfg.special_ids, C = this.cfg; const enc = this.ids(state); const qs = questions.map(q => ({ q, kind: q.type, body: this.ids(q.instructions).slice(0, C.max_question_tokens), opts: (q.type === "noul" ? ["True", "False"] : JevJa.optionsOf(q)).map(o => { const b = this.ids(String(o)).slice(0, C.max_option_tokens); return b.length ? b : [S.unk]; }) }));
    const extents = qs.map(x => 1 + x.body.length + x.opts.reduce((s, o) => s + o.length + 2, 0) + 1); const room = C.max_position - 1 - 1 - Math.max(0, ...extents);
    const ids = [S.cls, S.state, ...enc.slice(0, Math.max(0, Math.min(C.max_state_tokens, room)))]; const stateLen = ids.length; const branch = new Array(stateLen).fill(-1), pos = ids.map((_, i) => i); const recs = [];
    qs.forEach((x, j) => { const offset = ids.length; const bt = [S.question, ...x.body]; const bp = bt.map((_, i) => stateLen + i); const spans = []; for (const o of x.opts) { const seg = [S.option, ...o, S.end_option]; const start = offset + bt.length + 1; spans.push([start, start + o.length]); const fp = stateLen + bt.length; for (let i = 0; i < seg.length; i++) bp.push(fp + i); bt.push(...seg); } bt.push(S.decide); bp.push(stateLen + bt.length - 1); if (Math.max(...bp) >= C.max_position) throw new Error("入力が長すぎます（Jevlet の位置上限）"); ids.push(...bt); pos.push(...bp); for (let i = 0; i < bt.length; i++) branch.push(j); recs.push({ kind: x.kind, spans, decide: ids.length - 1, labels: JevJa.optionsOf(x.q) }); });
    if (ids.length > C.max_packed_len) throw new Error("入力が長すぎます（max_packed_len）");
    return { ids, pos, branch, recs };
  }
  async decide(state, questions, context, opts = {}) {
    const t0 = performance.now(); const full = (!context || !context.length) ? state : "[Context]\n" + context.map(c => "- " + c).join("\n") + "\n[State]\n" + state; const { ids, pos, branch, recs } = this.pack(full, questions); const L = ids.length, w = this.cfg.hidden;
    const mask = new Uint8Array(L * L); for (let q = 0; q < L; q++) for (let k = 0; k < L; k++) { const qs = branch[q] === -1, ks = branch[k] === -1; mask[q * L + k] = (qs && ks) || (!qs && (ks || branch[k] === branch[q])) ? 1 : 0; }
    const i64 = a => BigInt64Array.from(a, x => BigInt(x)); opts.onProgress?.(`${questions.length} 問を 1 回で判定中（${this.name}, ${L} tok）`);
    const H = (await this.sess.run({ input_ids: new ort.Tensor("int64", i64(ids), [1, L]), attention_mask: new ort.Tensor("bool", mask, [1, 1, L, L]), position_ids: new ort.Tensor("int64", i64(pos.map(p => p + (this.cfg.position_offset || 0))), [1, L]), token_type_ids: new ort.Tensor("int64", new BigInt64Array(L), [1, L]) })).last_hidden_state.data;
    const row = t => H.subarray(t * w, (t + 1) * w); const mv = (W, v) => { const o = new Float32Array(w); for (let i = 0; i < w; i++) { let s = 0; const off = i * w; for (let j = 0; j < w; j++) s += W[off + j] * v[j]; o[i] = s; } return o; };
    const temps = this.cfg.temperatures || {}; const answers = recs.map((r, i) => { const qd = mv(this.Wq, row(r.decide)); const z = r.spans.map(([s0, s1]) => { const m = new Float32Array(w); for (let t = s0; t < s1; t++) { const h = row(t); for (let j = 0; j < w; j++) m[j] += h[j]; } for (let j = 0; j < w; j++) m[j] /= Math.max(1, s1 - s0); const k = mv(this.Wk, m); let d = 0; for (let j = 0; j < w; j++) d += k[j] * qd[j]; return d / Math.sqrt(w); }); const T = (temps[r.kind] ?? temps.default ?? 1) * (opts.T || 1); const p = softmax(z, T); return mkAnswer(questions[i], r.labels, p, z); });
    return { model: this.name, backend: this.backend, probability_kind: "native", latency_ms: performance.now() - t0, tokens: L, answers };
  }
}
function mkAnswer(q, labels, p, z) { const am = p.indexOf(Math.max(...p)); const a = { id: q.id, type: q.type, distribution: { labels, probabilities: p }, logits: z, confidence: p[am] }; if (q.type === "choice") a.choice = labels[am]; else if (q.type === "score") { const vals = q.values || labels.map((_, i) => i); a.score = p.reduce((s, pi, i) => s + pi * vals[i], 0); a.level = labels[am]; } else { a.p_yes = p[0]; a.noul = p[0] >= 0.5; } return a; }

/* "gliclass": OpenJev Verdict（heman10x/rlcd-modernbert-151m、GLiClass 形、英語、Apache-2.0）。
   系列 <<LABEL>>opt1<<LABEL>>opt2…<<LABEL>>insufficient evidence<<SEP>>Question: q\n\nContext:\nstate（core/formatting.py と同じ）。
   出力 logits[:, :K] を選択肢順に読み、K（棄権枠込み）ごとの温度で割って softmax。棄権枠は「判断できない」として answer.abstain に残す */
export class GliClassJev {
  constructor(entry) { this.name = entry.name; this.cfg = entry; this.kind = "gliclass"; this.probability_kind = "native"; this.base = new URL(`./models/${entry.name}/`, import.meta.url).href; }
  async load(onProgress = () => {}) {
    const getf = f => getModelFile(this.name, f, this.base, onProgress); const txt = b => new TextDecoder().decode(b);
    this.cfg = { ...this.cfg, ...JSON.parse(txt(await getf("config.json"))) };
    this.tok = new PreTrainedTokenizer(JSON.parse(txt(await getf("tokenizer.json"))), JSON.parse(txt(await getf("tokenizer_config.json"))));
    const ext = []; for (const f of (this.cfg.external_data || [])) ext.push({ path: f, data: new Uint8Array(await getf(f)) });
    onProgress("ONNX セッションを作成中"); this.sess = await createSession(new Uint8Array(await getf("model.onnx")), { executionProviders: ["wasm"], externalData: ext }); this.backend = "wasm"; onProgress(""); return this;
  }
  static labelsOf(q) { if (q.type === "choice") return q.options.map(o => `It is ${o}`); if (q.type === "score") return q.levels.map((l, i) => `${l} (Value: ${q.values?.[i] ?? i})`); return [`true: ${q.instructions}`, `false: not ${q.instructions}`]; }
  async decide(state, questions, context, opts = {}) {
    const t0 = performance.now(); const ctx = (!context || !context.length) ? state : `Evidence:\n${context.map(c => "- " + c).join("\n")}\n\n${state}`; const ABST = "insufficient evidence"; const MAXC = this.cfg.max_candidates || 25;
    const prompts = questions.map(q => { const labels = [...GliClassJev.labelsOf(q).slice(0, MAXC - 1), ABST]; const text = q.type === "noul" ? `Context:\n${ctx}\n\nEvaluate proposition: ${q.instructions}` : `Question: ${q.instructions}\n\nContext:\n${ctx}`; return { labels, s: labels.map(l => `<<LABEL>>${l}`).join("") + "<<SEP>>" + text }; });
    const encs = prompts.map(p => Array.from(this.tok(p.s, { truncation: true, max_length: this.cfg.max_len || 1024 }).input_ids.data, Number)); const n = encs.length, L = Math.max(...encs.map(e => e.length)); const PAD = 50283;
    const ids = new BigInt64Array(n * L).fill(BigInt(PAD)), att = new BigInt64Array(n * L); encs.forEach((e, i) => e.forEach((x, j) => { ids[i * L + j] = BigInt(x); att[i * L + j] = 1n; }));
    opts.onProgress?.(`${questions.length} 問を 1 回で判定中（${this.name}, ${L} tok）`);
    const out = await this.sess.run({ input_ids: new ort.Tensor("int64", ids, [n, L]), attention_mask: new ort.Tensor("int64", att, [n, L]) }); const Z = Array.from(out.logits.data), C = out.logits.dims[1];
    const answers = questions.map((q, i) => { const K = prompts[i].labels.length; const z = Z.slice(i * C, i * C + K); const T = (this.cfg.per_k?.[String(K)] ?? this.cfg.temperature ?? 1) * (opts.T || 1); const pAll = softmax(z, T); const labels = JevJa.optionsOf(q); let p = pAll.slice(0, K - 1); const abst = pAll[K - 1]; const s = p.reduce((a, b) => a + b, 0) || 1; p = p.map(x => x / s); const a = mkAnswer(q, labels, p, z.slice(0, K - 1)); a.abstain = abst; if (abst > Math.max(...p) * (1 - abst)) a.abstained = true; return a; });
    return { model: this.name, backend: this.backend, probability_kind: "native", latency_ms: performance.now() - t0, tokens: L, answers };
  }
}
/* "biencoder": verdict-small（Manav2op/verdict-small、multilingual-e5-small、多言語、Apache-2.0）。
   状況 "query: state" と各選択肢 "passage: 質問 選択肢" を別々に埋め込み、cos × 20 を logit として softmax（Verdict パッケージと同じ） */
export class BiEncJev {
  constructor(entry) { this.name = entry.name; this.cfg = entry; this.kind = "biencoder"; this.probability_kind = "native"; this.base = new URL(`./models/${entry.name}/`, import.meta.url).href; }
  async load(onProgress = () => {}) {
    const getf = f => getModelFile(this.name, f, this.base, onProgress); const txt = b => new TextDecoder().decode(b);
    this.cfg = { ...this.cfg, ...JSON.parse(txt(await getf("config.json"))) };
    const tj = JSON.parse(txt(await getf("tokenizer.json"))); const fixMeta = pt => { if (!pt) return; if (pt.type === "Metaspace" && pt.add_prefix_space == null) pt.add_prefix_space = (pt.prepend_scheme ?? "always") !== "never"; (pt.pretokenizers || []).forEach(fixMeta); }; fixMeta(tj.pre_tokenizer);
    this.tok = new PreTrainedTokenizer(tj, JSON.parse(txt(await getf("tokenizer_config.json"))));
    const ext = []; for (const f of (this.cfg.external_data || [])) ext.push({ path: f, data: new Uint8Array(await getf(f)) });
    onProgress("ONNX セッションを作成中"); this.sess = await createSession(new Uint8Array(await getf("model.onnx")), { executionProviders: ["wasm"], externalData: ext }); this.backend = "wasm"; onProgress(""); return this;
  }
  async embed(texts) { const encs = texts.map(t => Array.from(this.tok(t, { truncation: true, max_length: this.cfg.max_len || 512 }).input_ids.data, Number)); const n = encs.length, L = Math.max(...encs.map(e => e.length)); const ids = new BigInt64Array(n * L).fill(1n), att = new BigInt64Array(n * L); encs.forEach((e, i) => e.forEach((x, j) => { ids[i * L + j] = BigInt(x); att[i * L + j] = 1n; }));
    const out = await this.sess.run({ input_ids: new ort.Tensor("int64", ids, [n, L]), attention_mask: new ort.Tensor("int64", att, [n, L]) }); const E = out.sentence_embedding; const d = E.dims[1]; const V = []; for (let i = 0; i < n; i++) { const v = Array.from(E.data.slice(i * d, (i + 1) * d)); const nm = Math.sqrt(v.reduce((s, x) => s + x * x, 0)) || 1; V.push(v.map(x => x / nm)); } return V; }
  async decide(state, questions, context, opts = {}) {
    const t0 = performance.now(); const full = (!context || !context.length) ? state : "[根拠]\n" + context.map(c => "- " + c).join("\n") + "\n[状況]\n" + state; const qp = this.cfg.query_prefix || "query: ", pp = this.cfg.passage_prefix || "passage: "; const scale = this.cfg.scale || 20;
    opts.onProgress?.(`${questions.length} 問を判定中（${this.name}）`); const [qv] = await this.embed([qp + full]);
    const answers = []; for (const q of questions) { const labels = JevJa.optionsOf(q); const texts = q.type === "noul" ? [`${pp}${q.instructions} — yes, this holds`, `${pp}${q.instructions} — no, this does not hold`] : labels.map(l => `${pp}${q.instructions} ${l}`); const P = await this.embed(texts); const z = P.map(v => scale * v.reduce((s, x, k) => s + x * qv[k], 0)); const p = softmax(z, opts.T || 1); answers.push(mkAnswer(q, labels, p, z)); }
    return { model: this.name, backend: this.backend, probability_kind: "native", latency_ms: performance.now() - t0, answers };
  }
}

export async function isStoredAny(name, files) { const ks = new Set(await store.keys()); return files.every(f => ks.has(`${name}/${f}`)); }

/* registry + index からモデル一覧を作り、名前でロードする */
export async function listAllModels() {
  const out = [];
  try { const idx = await (await fetch(new URL("./models/index.json", import.meta.url))).json(); for (const m of idx.models) out.push({ ...m, kind: m.kind || "jev_ja", group: m.kind === "jev_ja" || !m.kind ? "自作 JEV-JA（日本語・学習可）" : (m.language === "ja" ? "既存 Jev（日本語）" : /multi/.test(m.language || "") ? "既存 Jev（多言語）" : "既存 Jev（英語）") }); } catch { }
  try { const reg = await (await fetch(new URL("./models/registry.json", import.meta.url))).json(); for (const m of reg.models) out.push({ ...m, group: m.group || "既存 Jev（HF から取得）" }); } catch { }
  try { const custom = await store.get("registry:custom"); if (custom) for (const m of custom) out.push({ ...m, group: "追加したモデル（この端末）" }); } catch { }
  return out;
}
export async function loadModelByName(name, onProgress) {
  const all = await listAllModels(); const e = all.find(m => m.name === name); if (!e) throw new Error("unknown model: " + name);
  if (e.kind === "jev_ja") return new JevJa(name).load(onProgress);
  if (e.kind === "crossenc") return new CrossEncJev(e).load(onProgress);
  if (e.kind === "open-jev-onnx") return new OpenJevOnnx(e).load(onProgress);
  if (e.kind === "laya") return new LayaJev(e).load(onProgress);
  if (e.kind === "jevlet") return new JevletJev(e).load(onProgress);
  if (e.kind === "gliclass") return new GliClassJev(e).load(onProgress);
  if (e.kind === "biencoder") return new BiEncJev(e).load(onProgress);
  throw new Error("unsupported kind: " + e.kind);
}
