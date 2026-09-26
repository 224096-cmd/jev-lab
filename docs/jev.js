/* JEV-JA 端末内推論（ONNX Runtime Web + Transformers.js tokenizer）
   Python 側 jev_lab/adapters/jev_ja.py の encode_example / forward_logits と同じ手順を JS で再現する。 */
import { AutoTokenizer, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.5.2/dist/transformers.min.js";
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.webgpu.min.mjs";

env.allowRemoteModels = false;
env.allowLocalModels = true;
env.localModelPath = new URL("./models/", import.meta.url).href;
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";

export class JevJa {
  constructor(name) { this.name = name; this.base = `${env.localModelPath}${name}/`; }

  async load(onProgress = () => {}) {
    onProgress("設定を読込中");
    this.cfg = await (await fetch(this.base + "config.json")).json();
    onProgress("トークナイザを読込中");
    this.tok = await AutoTokenizer.from_pretrained(this.name);
    const ep = (this.cfg.prefer_webgpu && navigator.gpu) ? ["webgpu", "wasm"] : ["wasm"];
    onProgress(`エンコーダ ONNX を読込中（${this.cfg.size_mb} MB、初回のみ）`);
    this.enc = await ort.InferenceSession.create(this.base + "encoder.onnx", { executionProviders: ep });
    this.head = await ort.InferenceSession.create(this.base + "head.onnx", { executionProviders: ["wasm"] });
    this.backend = ep[0];
    onProgress("");
    return this;
  }

  _raw(text) { return Array.from(this.tok(text, { add_special_tokens: false }).input_ids.data, Number); }

  /* Python(tokenizers) と同じ byte fallback を再現:
     語彙に無い文字は <unk> ではなく UTF-8 バイト列 <0xXX> にする（改行・稀な漢字など） */
  ids(text) {
    const unk = this.cfg.special_ids.unk, B = this.cfg.byte_ids;
    let out = this._raw(text);
    if (!out.includes(unk) || !B) return out;
    this._unkCache ??= new Map();
    const isUnk = ch => { if (!this._unkCache.has(ch)) this._unkCache.set(ch, this._raw(ch).includes(unk)); return this._unkCache.get(ch); };
    out = []; let seg = "";
    for (const ch of text) {
      if (isUnk(ch)) { if (seg) { out.push(...this._raw(seg)); seg = ""; } for (const b of new TextEncoder().encode(ch)) out.push(B[b]); }
      else seg += ch;
    }
    if (seg) out.push(...this._raw(seg));
    return out;
  }

  static optionsOf(q) { return q.type === "choice" ? q.options : q.type === "score" ? q.levels : ["yes", "no"]; }

  buildState(state, context) {
    if (!context || !context.length) return state;
    return "[根拠]\n" + context.map(c => "- " + c).join("\n") + "\n[状況]\n" + state;
  }

  /* 入力列と各スパン位置（Python と同一） */
  encode(state, questions) {
    const S = this.cfg.special_ids;
    const ids = [S.cls, S.state, ...this.ids(state)];
    const spans = {}, marker = {};
    questions.forEach((q, j) => {
      ids.push(S.q);
      let s = ids.length; ids.push(...this.ids(q.instructions)); spans[`q${j}`] = [s, ids.length];
      JevJa.optionsOf(q).forEach((o, k) => {
        marker[`o${j}_${k}`] = ids.length; ids.push(S.opt);
        s = ids.length; ids.push(...this.ids(o)); spans[`o${j}_${k}`] = [s, ids.length];
      });
    });
    ids.push(S.sep);
    if (ids.length > this.cfg.max_length) throw new Error(`入力長 ${ids.length} が max_length ${this.cfg.max_length} を超えました`);
    return { ids, spans, marker };
  }

  async decide(state, questions, context) {
    const t0 = performance.now();
    const full = this.buildState(state, context);
    const { ids, spans, marker } = this.encode(full, questions);
    const L = ids.length, d = this.cfg.hidden;
    const inp = new ort.Tensor("int64", BigInt64Array.from(ids, x => BigInt(x)), [1, L]);
    const H = (await this.enc.run({ input_ids: inp })).last_hidden_state.data; // (L*d)
    const mean = ([s, e]) => { const v = new Float32Array(d); for (let t = s; t < e; t++) for (let i = 0; i < d; i++) v[i] += H[t * d + i]; for (let i = 0; i < d; i++) v[i] /= (e - s); return v; };
    const row = t => H.slice(t * d, (t + 1) * d);
    const answers = [];
    for (let j = 0; j < questions.length; j++) {
      const q = questions[j], opts = JevJa.optionsOf(q), K = opts.length;
      const u = mean(spans[`q${j}`]);
      const V = new Float32Array(K * d);
      for (let k = 0; k < K; k++) V.set(this.cfg.pool === "span" ? mean(spans[`o${j}_${k}`]) : row(marker[`o${j}_${k}`]), k * d);
      const z = (await this.head.run({ u: new ort.Tensor("float32", u, [d]), v: new ort.Tensor("float32", V, [K, d]) })).logits.data;
      const T = this.cfg.temperature || 1, m = Math.max(...z);
      const e = Array.from(z, x => Math.exp((x - m) / T)), sum = e.reduce((a, b) => a + b, 0);
      const p = e.map(x => x / sum);
      const am = p.indexOf(Math.max(...p));
      const a = { id: q.id, type: q.type, distribution: { labels: opts, probabilities: p }, confidence: p[am] };
      if (q.type === "choice") a.choice = opts[am];
      else if (q.type === "score") { const vals = q.values || opts.map((_, i) => i); a.score = p.reduce((s, pi, i) => s + pi * vals[i], 0); a.level = opts[am]; }
      else { a.p_yes = p[0]; a.noul = p[0] >= 0.5; }
      answers.push(a);
    }
    return { model: this.name, backend: this.backend, probability_kind: "native", latency_ms: performance.now() - t0, tokens: L, answers };
  }
}
