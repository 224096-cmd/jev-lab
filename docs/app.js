import { JevJa, store, isStored, removeStored, storedBytes, softmax, pca2, metrics } from "./jev.js";
import { TEMPLATES } from "./templates.js";
import { SOURCES, OSINT_QUESTIONS, collect, evidencePack, geoContext, trustScore, TRUST_WEIGHTS_DEFAULT } from "./osint.js";
import { clusterByCos } from "./jev.js";

const $ = s => document.querySelector(s), $$ = s => [...document.querySelectorAll(s)];
let jev = null, last = null, models = [], batchItems = [], batchResult = null;
const status = t => $("#status").textContent = t;
const fmt = (x, d = 1) => (x == null || isNaN(x)) ? "–" : (+x).toFixed(d);

/* ---------------- tabs ---------------- */
$$(".tabs button").forEach(b => b.onclick = () => { try { localStorage.setItem("jev.tab", b.dataset.t); } catch { } $$(".tabs button").forEach(x => x.classList.remove("on")); b.classList.add("on"); ["play", "verify", "viz", "batch", "osint", "notes", "tmpl", "settings"].forEach(t => $("#t-" + t).classList.toggle("hidden", t !== b.dataset.t)); if (b.dataset.t === "settings") refreshSettings(); if (b.dataset.t === "notes") refreshNotes(); });

/* ---------------- KaTeX ---------------- */
const K = (el, tex) => window.katex && katex.render(tex, $(el), { displayMode: true, throwOnError: false });
K("#f0", String.raw`s \;+\; \{q_j=(\text{instr}_j,\,O_j)\}_{j=1}^{N} \;\xrightarrow{\;1\text{ forward}\;}\; \{p_j\in\Delta^{|O_j|-1}\}_{j=1}^{N}`);
K("#f1", String.raw`H=\mathrm{Enc}([\mathrm{CLS}][\mathrm{STATE}]\,s\,[\mathrm{Q}]\,\text{instr}_j\,[\mathrm{OPT}]\,o_{j1}\cdots[\mathrm{SEP}]),\qquad u_j=\overline{H_{\text{instr}_j}},\quad v_{jk}=\overline{H_{o_{jk}}}`);
K("#f2", String.raw`z_{jk}=w^{\top}\tanh\!\big(W\,[u_j;\,v_{jk};\,u_j\odot v_{jk}]+b\big),\qquad p_{jk}=\frac{e^{z_{jk}/T}}{\sum_{k'}e^{z_{jk'}/T}},\qquad \mathbb{E}[\text{score}]=\sum_k p_{jk}\,\ell_k`);
K("#f3", String.raw`\mathcal{L}=-\log p_{j,y}+\lambda\sum_k\big(p_{jk}-\mathbb{1}[k=y]\big)^2,\qquad \mathrm{ECE}=\sum_b\frac{|B_b|}{n}\big|\mathrm{acc}(B_b)-\mathrm{conf}(B_b)\big|`);

/* ---------------- question builder ---------------- */
const qb = $("#qb");
function qCard(q) {
  const div = document.createElement("div"); div.className = "qcard"; div.dataset.type = q.type;
  const listLabel = q.type === "choice" ? "選択肢（1 行 1 つ）" : q.type === "score" ? "段階（低 → 高、1 行 1 つ）" : null;
  const list = q.type === "choice" ? q.options : q.type === "score" ? q.levels : [];
  div.innerHTML = `<div class="hd"><select class="qt"><option value="choice">Choice</option><option value="score">Score</option><option value="noul">Noul</option></select><input class="qid" placeholder="id" value="${q.id || ""}"><button class="qdel" title="削除">✕</button></div>
   <label>質問文（instructions）</label><input class="qi" value="${(q.instructions || "").replace(/"/g, "&quot;")}">
   ${listLabel ? `<label>${listLabel}</label><textarea class="ql" style="min-height:70px">${list.join("\n")}</textarea>` : `<p class="muted small">yes / no で答える問い。p(yes) が返る。</p>`}`;
  div.querySelector(".qt").value = q.type;
  div.querySelector(".qt").onchange = e => { const nq = readCard(div); nq.type = e.target.value; if (nq.type === "choice" && !nq.options) nq.options = nq.levels || ["A", "B"]; if (nq.type === "score" && !nq.levels) nq.levels = nq.options || ["低い", "中程度", "高い"]; div.replaceWith(qCard(nq)); syncJson(); };
  div.querySelector(".qdel").onclick = () => { div.remove(); syncJson(); };
  div.querySelectorAll("input,textarea").forEach(el => el.oninput = syncJson);
  return div;
}
function readCard(div) {
  const type = div.querySelector(".qt").value, id = div.querySelector(".qid").value.trim() || "q", instructions = div.querySelector(".qi").value;
  const list = div.querySelector(".ql") ? div.querySelector(".ql").value.split("\n").map(s => s.trim()).filter(Boolean) : null;
  return type === "choice" ? { type, id, instructions, options: list } : type === "score" ? { type, id, instructions, levels: list } : { type, id, instructions };
}
const readQuestions = () => $$("#qb .qcard").map(readCard);
const syncJson = () => $("#p-q").value = JSON.stringify(readQuestions(), null, 1);
function setQuestions(qs) { qb.innerHTML = ""; qs.forEach(q => qb.appendChild(qCard(structuredClone(q)))); syncJson(); }
$("#q-add-choice").onclick = () => { qb.appendChild(qCard({ type: "choice", id: "q" + (qb.children.length + 1), instructions: "", options: ["A", "B", "C"] })); syncJson(); };
$("#q-add-score").onclick = () => { qb.appendChild(qCard({ type: "score", id: "q" + (qb.children.length + 1), instructions: "", levels: ["低い", "中程度", "高い"] })); syncJson(); };
$("#q-add-noul").onclick = () => { qb.appendChild(qCard({ type: "noul", id: "q" + (qb.children.length + 1), instructions: "" })); syncJson(); };
$("#q-apply").onclick = () => { try { setQuestions(JSON.parse($("#p-q").value)); } catch (e) { alert("JSON エラー: " + e.message); } };

/* ---------------- templates ---------------- */
function applyTemplate(t) { $("#p-state").value = t.state; $("#p-ctx").value = t.context.join("\n"); setQuestions(t.questions); }
TEMPLATES.forEach(t => { const o = document.createElement("option"); o.value = t.id; o.textContent = t.name; $("#p-tmpl").appendChild(o); });
$("#p-tmpl").onchange = () => applyTemplate(TEMPLATES.find(t => t.id === $("#p-tmpl").value));
$("#tmpl-list").innerHTML = TEMPLATES.map(t => `<div class="qcard"><h2>${t.name}</h2><p>${t.desc}</p><p class="small">質問: ${t.questions.map(q => `<span class="pill">${q.type}: ${q.id}</span>`).join(" ")}</p><button data-id="${t.id}" class="use">Playground に読み込む</button> <button data-id="${t.id}" class="tri">トリアージ（複数テキスト）を開く</button><div class="tri-box hidden"><label>テキスト（1 行 1 件）</label><textarea class="tri-texts">${[t.state, ...t.texts].join("\n")}</textarea><label>並び替えキー</label><input class="tri-sort" value="${(t.questions.find(q => q.type === "score") || t.questions[0]).id}"><p><button class="primary tri-run" data-id="${t.id}">一括判断</button> <button class="tri-csv">CSV 保存</button></p><div class="tri-out"></div></div></div>`).join("");
$$("#tmpl-list .use").forEach(b => b.onclick = () => { applyTemplate(TEMPLATES.find(t => t.id === b.dataset.id)); $("#p-tmpl").value = b.dataset.id; $$(".tabs button")[0].click(); });
$$("#tmpl-list .tri").forEach(b => b.onclick = () => b.parentElement.querySelector(".tri-box").classList.toggle("hidden"));
$$("#tmpl-list .tri-run").forEach(b => b.onclick = async () => {
  if (!jev) return alert("先にモデルを読み込んでください"); const t = TEMPLATES.find(x => x.id === b.dataset.id), box = b.closest(".tri-box");
  const texts = box.querySelector(".tri-texts").value.split("\n").filter(s => s.trim()), key = box.querySelector(".tri-sort").value; const rows = [];
  status("一括判断中…"); for (const text of texts) { const r = await jev.decide(text, t.questions, t.context.length ? t.context : null, curOpts()); rows.push({ text, r }); }
  const g = r => { const a = r.answers.find(a => a.id === key) || {}; return a.score ?? a.p_yes ?? a.confidence ?? 0; }; rows.sort((x, y) => g(y.r) - g(x.r));
  box.querySelector(".tri-out").innerHTML = triageTable(rows, t.questions); box.dataset.csv = triageCsv(rows, t.questions); status("");
});
$$("#tmpl-list .tri-csv").forEach(b => b.onclick = () => { const csv = b.closest(".tri-box").dataset.csv; if (csv) download("triage.csv", "﻿" + csv, "text/csv"); });
function triageTable(rows, qs) { let h = "<table><tr><th>テキスト</th>" + qs.map(q => `<th>${q.id}</th>`).join("") + "<th>ms</th></tr>"; for (const { text, r } of rows) { h += `<tr><td>${esc(text)}</td>`; for (const a of r.answers) h += "<td>" + (a.type === "choice" ? `${a.choice}<br><span class="muted small">${(a.confidence * 100).toFixed(0)}%</span>` : a.type === "score" ? a.score.toFixed(2) : `${a.noul ? "yes" : "no"} (${a.p_yes.toFixed(2)})`) + "</td>"; h += `<td>${r.latency_ms.toFixed(0)}</td></tr>`; } return h + "</table>"; }
function triageCsv(rows, qs) { const cell = v => `"${String(v).replace(/"/g, '""')}"`; return [["text", ...qs.map(q => q.id), ...qs.map(q => q.id + "_conf"), "ms"].join(",")].concat(rows.map(({ text, r }) => [cell(text), ...r.answers.map(a => cell(a.choice ?? (a.score != null ? a.score.toFixed(3) : (a.noul ? "yes" : "no")))), ...r.answers.map(a => (a.p_yes ?? a.confidence).toFixed(3)), r.latency_ms.toFixed(0)].join(","))).join("\n"); }
const esc = s => String(s).replace(/[&<>]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
function download(name, data, type = "application/json") { const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([data], { type })); a.download = name; a.click(); }

/* ---------------- model list / load ---------------- */
async function loadIndex() { models = (await (await fetch("models/index.json")).json()).models; for (const id of ["#model", "#v-models"]) { const s = $(id); s.innerHTML = ""; for (const m of models) { const o = document.createElement("option"); o.value = m.name; o.textContent = `${m.name} — ${m.size_mb} MB ${m.trained ? "(学習済)" : "(未学習)"}`; s.appendChild(o); } } await markStored(); }
async function markStored() { for (const o of $("#model").options) { const s = await isStored(o.value); o.textContent = o.textContent.replace(/ ・保存済$/, "") + (s ? " ・保存済" : ""); } }
async function storageInfo() { try { const e = await navigator.storage.estimate(); const p = await navigator.storage.persisted?.(); const mb = await storedBytes(); $("#storage").textContent = `端末内ストレージ: モデル ${(mb / 1e6).toFixed(0)} MB 保存済 ・ サイト全体 ${(e.usage / 1e6).toFixed(0)} / ${(e.quota / 1e6).toFixed(0)} MB ・ 永続化 ${p ? "済" : "未"}`; } catch { } }
async function offlineCheck() { const name = $("#model").value; const okModel = await isStored(name); const sw = navigator.serviceWorker?.controller; if (!sw) { $("#offline").textContent = okModel ? "モデルは端末内に保存済。ライブラリの準備状況はページを開き直すと表示されます" : "オフライン準備: 未（オンラインで一度「モデルを読み込む」）"; return; }
  navigator.serviceWorker.onmessage = ev => { if (ev.data?.type !== "offline-status") return; const ok = ev.data.ready && okModel; $("#offline").innerHTML = ok ? `✅ オフライン準備: 済（${name} は機内モードでも動きます）` : `⚠️ オフライン準備: 未 — ${okModel ? "" : "モデル未保存 "}${ev.data.missing.length ? "ライブラリ不足: " + ev.data.missing.join(", ") : ""}（オンラインで「モデルを読み込む」を実行）`; };
  sw.postMessage({ type: "offline-check", model: name }); }
$("#model").onchange = offlineCheck;
$("#load").onclick = async () => {
  $("#o-judge").disabled = true;
  try { await navigator.storage.persist?.(); } catch { }
  $("#p-run").disabled = $("#b-run").disabled = true; $("#load").disabled = true;
  try { jev = await new JevJa($("#model").value).load(status); $("#minfo").textContent = `${jev.name} / ${jev.backend} / pool=${jev.cfg.pool} / T=${jev.cfg.temperature} / ${jev.cfg.trained ? "学習済" : "未学習"}`; $("#p-run").disabled = false; $("#b-run").disabled = !batchItems.length; $("#o-judge").disabled = !oItems.length; $("#T").value = jev.cfg.temperature || 1; $("#Tv").textContent = (+$("#T").value).toFixed(2); try { localStorage.setItem("jev.lastModel", jev.name); } catch { } }
  catch (e) { $("#minfo").textContent = "ロード失敗: " + e.message; console.error(e); }
  $("#load").disabled = false; status(""); storageInfo(); markStored(); offlineCheck();
};

/* ---------------- playground run + render ---------------- */
const curOpts = () => ({ pool: $$("input[name=pool]").find(r => r.checked).value, T: +$("#T").value, method: $$("input[name=method]").find(r => r.checked).value, cosScale: +$("#cosScale").value || 20 });
const optsLabel = o => `${o.method}${o.method === "cos" ? "×" + o.cosScale : ""}/${o.pool}/T=${o.T.toFixed(2)}`;
function renderAnswers(res, answers) {
  let h = `<p><b>${res.model}</b> <span class="tag">${res.probability_kind}</span> <span class="tag">${res.backend}</span> <span class="tag">${curOpts().method}</span> <span class="tag">pool=${curOpts().pool}</span> <span class="tag">T=${curOpts().T.toFixed(2)}</span> <span class="muted">${fmt(res.latency_ms, 0)} ms / ${res.n_tokens ?? res.tokens} tok</span></p>`;
  for (const a of answers) { const d = a.distribution; h += `<p><b>${a.id}</b>: `; if (a.type === "choice") h += `<b>${a.choice}</b> (${(a.confidence * 100).toFixed(0)}%)`; else if (a.type === "score") h += `期待値 <b>${a.score.toFixed(2)}</b> / 最頻 ${a.level}`; else h += `<b>${a.noul ? "yes" : "no"}</b> p(yes)=${a.p_yes.toFixed(2)}`; h += "</p><table>";
    d.labels.forEach((l, i) => h += `<tr><td style="width:32%">${esc(l)}</td><td><div class="bar"><i style="width:${d.probabilities[i] * 100}%"></i></div></td><td style="width:58px">${(d.probabilities[i] * 100).toFixed(1)}%</td><td style="width:70px" class="muted small">z=${a.logits[i].toFixed(2)}</td></tr>`); h += "</table>"; }
  return h;
}
function rerender() { if (!last) return; last.answers = JevJa.answersFrom(last, curOpts()); $("#p-out").innerHTML = renderAnswers(last, last.answers); renderViz(last); }
$("#T").oninput = () => { $("#Tv").textContent = (+$("#T").value).toFixed(2); rerender(); };
$$("input[name=pool],input[name=method]").forEach(r => r.onchange = rerender); $("#cosScale").oninput = rerender;
const ctx = () => $("#p-ctx").value.split("\n").map(s => s.trim()).filter(Boolean);
$("#p-run").onclick = async () => { status("推論中…"); try { last = await jev.analyze($("#p-state").value, readQuestions(), ctx(), curOpts()); rerender(); } catch (e) { $("#p-out").textContent = e.message; } status(""); };
$("#p-save").onclick = () => { if (!last) return; const { u, Vs, Vm, ...rest } = {}; download(`decide-${Date.now()}.json`, JSON.stringify({ model: last.model, backend: last.backend, opts: curOpts(), timing: last.timing, n_tokens: last.n_tokens, input: last.input, answers: last.answers, tokens: last.tokens.map(t => t.text) }, null, 1)); };

/* ---------------- viz ---------------- */
function renderViz(res) {
  $("#viz-tokens").innerHTML = res.tokens.map(t => `<span class="tk ${t.role}" title="${t.role} id=${t.id}">${esc(t.text)}</span>`).join("");
  const { pool, T } = curOpts();
  $("#viz-uv").innerHTML = res.questions.map((qr, j) => `<p><b>${qr.q.id}</b> — |u|=${norm(qr.u).toFixed(2)}, 質問トークン ${res.spans["q" + j][1] - res.spans["q" + j][0]} 個</p><table><tr><th>選択肢</th><th>トークン数</th><th>cos(u,v) span</th><th>cos(u,v) marker</th></tr>${qr.labels.map((l, k) => `<tr><td>${esc(l)}</td><td>${res.spans[`o${j}_${k}`][1] - res.spans[`o${j}_${k}`][0]}</td><td>${qr.cos_span[k].toFixed(3)}</td><td>${qr.cos_marker[k].toFixed(3)}</td></tr>`).join("")}</table>`).join("");
  $("#viz-logits").innerHTML = res.questions.map(qr => { const o = curOpts(); const z = o.method === "cos" ? (pool === "marker" ? qr.cos_marker : qr.cos_span).map(c => c * o.cosScale) : (pool === "marker" ? qr.z_marker : qr.z_span), p = softmax(z, T), p1 = softmax(z, 1); return `<p><b>${qr.q.id}</b>（${o.method} / ${pool}）</p><table><tr><th>選択肢</th><th>logit z</th><th>softmax(T=1)</th><th>softmax(T=${T.toFixed(2)})</th></tr>${qr.labels.map((l, k) => `<tr><td>${esc(l)}</td><td>${z[k].toFixed(3)}</td><td>${(p1[k] * 100).toFixed(1)}%</td><td><b>${(p[k] * 100).toFixed(1)}%</b></td></tr>`).join("")}</table>`; }).join("");
  $("#viz-time").innerHTML = `<span>トークン数</span><span>${res.n_tokens}</span><span>エンコーダ</span><span>${fmt(res.timing.encoder_ms, 0)} ms</span><span>ヘッド（${res.questions.length} 問 × 2 読み出し）</span><span>${fmt(res.timing.head_ms, 0)} ms</span><span>合計</span><span>${fmt(res.timing.total_ms, 0)} ms</span><span>バックエンド</span><span>${res.backend}</span>`;
  drawPca(res, pool);
}
const norm = v => Math.sqrt(v.reduce((s, x) => s + x * x, 0));
function drawPca(res, pool) {
  const c = $("#viz-pca"), g = c.getContext("2d"); g.clearRect(0, 0, c.width, c.height);
  const pts = [], meta = []; res.questions.forEach((qr, j) => { pts.push(qr.u); meta.push({ j, k: -1, l: qr.q.id }); (pool === "marker" ? qr.Vm : qr.Vs).forEach((v, k) => { pts.push(v); meta.push({ j, k, l: qr.labels[k] }); }); });
  if (pts.length < 3) return; const xy = pca2(pts); const xs = xy.map(p => p[0]), ys = xy.map(p => p[1]); const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  const X = x => 30 + (x - x0) / ((x1 - x0) || 1) * (c.width - 60), Y = y => c.height - 30 - (y - y0) / ((y1 - y0) || 1) * (c.height - 60);
  const cols = ["#0b6e99", "#b45309", "#15803d", "#7c3aed", "#be123c", "#0f766e"]; const dark = matchMedia("(prefers-color-scheme:dark)").matches; g.font = "12px system-ui"; g.lineWidth = 1;
  meta.forEach((m, i) => { const col = cols[m.j % cols.length]; const x = X(xy[i][0]), y = Y(xy[i][1]); g.fillStyle = col; g.strokeStyle = col; g.beginPath(); g.arc(x, y, m.k < 0 ? 7 : 5, 0, 7); if (m.k < 0) g.fill(); else { g.fillStyle = dark ? "#171b21" : "#fff"; g.fill(); g.stroke(); } g.fillStyle = dark ? "#e6e8eb" : "#1a1d21"; g.fillText((m.k < 0 ? "● " : "") + m.l, x + 8, y + 4); });
  meta.forEach((m, i) => { if (m.k >= 0) { const ui = meta.findIndex(t => t.j === m.j && t.k < 0); g.strokeStyle = cols[m.j % cols.length] + "55"; g.beginPath(); g.moveTo(X(xy[ui][0]), Y(xy[ui][1])); g.lineTo(X(xy[i][0]), Y(xy[i][1])); g.stroke(); } });
}

/* ---------------- verify ---------------- */
const l1 = (a, b) => a.reduce((s, x, i) => s + Math.abs(x - b[i]), 0);
const ready = () => { if (!jev) { alert("先にモデルを読み込んでください"); return false; } return true; };
$("#v-ctx").onclick = async () => { if (!ready()) return; status("…"); const qs = readQuestions(), s = $("#p-state").value; const a = await jev.decide(s, qs, null, curOpts()), b = await jev.decide(s, qs, ctx(), curOpts());
  $("#v-ctx-out").innerHTML = "<table><tr><th>質問</th><th>context なし</th><th>context あり</th><th>L1 距離</th><th>argmax 変化</th></tr>" + qs.map((q, i) => { const pa = a.answers[i].distribution.probabilities, pb = b.answers[i].distribution.probabilities; const la = a.answers[i].choice ?? a.answers[i].level ?? (a.answers[i].noul ? "yes" : "no"), lb = b.answers[i].choice ?? b.answers[i].level ?? (b.answers[i].noul ? "yes" : "no"); return `<tr><td>${q.id}</td><td>${la} (${(Math.max(...pa) * 100).toFixed(0)}%)</td><td>${lb} (${(Math.max(...pb) * 100).toFixed(0)}%)</td><td>${l1(pa, pb).toFixed(3)}</td><td>${la === lb ? "なし" : "<b>あり</b>"}</td></tr>`; }).join("") + "</table>"; status(""); };
$("#v-shuf").onclick = async () => { if (!ready()) return; status("…"); const N = +$("#v-n").value || 8, qs = readQuestions(), s = $("#p-state").value, c = ctx(); const base = await jev.decide(s, qs, c, curOpts()); const agree = qs.map(() => 0), l1s = qs.map(() => 0);
  for (let n = 0; n < N; n++) { const qs2 = qs.map(q => { const q2 = structuredClone(q); const key = q.type === "choice" ? "options" : null; if (key) q2[key] = [...q[key]].sort(() => Math.random() - 0.5); return q2; }); const r = await jev.decide(s, qs2, c, curOpts());
    qs.forEach((q, i) => { if (q.type !== "choice") return; const d = r.answers[i].distribution; const p = q.options.map(o => d.probabilities[d.labels.indexOf(o)]); const pb = base.answers[i].distribution.probabilities; if (r.answers[i].choice === base.answers[i].choice) agree[i]++; l1s[i] += l1(p, pb); }); }
  $("#v-shuf-out").innerHTML = "<table><tr><th>質問</th><th>argmax 一致率</th><th>平均 L1</th></tr>" + qs.map((q, i) => q.type === "choice" ? `<tr><td>${q.id}</td><td>${(agree[i] / N * 100).toFixed(0)}%</td><td>${(l1s[i] / N).toFixed(3)}</td></tr>` : `<tr><td>${q.id}</td><td colspan=2 class="muted">Choice のみ</td></tr>`).join("") + "</table>"; status(""); };
$("#v-neg").onclick = async () => { if (!ready()) return; status("…"); const qs = readQuestions().filter(q => q.type === "noul"); if (!qs.length) { $("#v-neg-out").textContent = "Noul の質問がありません"; status(""); return; } const s = $("#p-state").value, c = ctx();
  const a = await jev.decide(s, qs, c, curOpts()); const neg = qs.map(q => ({ ...q, instructions: `次の命題は誤りか: ${q.instructions}` })); const b = await jev.decide(s, neg, c, curOpts());
  $("#v-neg-out").innerHTML = "<table><tr><th>質問</th><th>p(yes) 元</th><th>p(yes) 否定形</th><th>反転</th></tr>" + qs.map((q, i) => { const p = a.answers[i].p_yes, pn = b.answers[i].p_yes; return `<tr><td>${q.id}</td><td>${p.toFixed(2)}</td><td>${pn.toFixed(2)}</td><td>${(p >= 0.5) !== (pn >= 0.5) ? "✅ 反転" : "❌ 同じ（質問を読んでいない可能性）"}</td></tr>`; }).join("") + "</table>"; status(""); };
$("#v-drop").onclick = async () => { if (!ready()) return; status("…"); const qs = readQuestions().filter(q => q.type === "choice"); const s = $("#p-state").value, c = ctx(); let h = "";
  for (const q of qs) { const base = (await jev.decide(s, [q], c, curOpts())).answers[0]; h += `<p><b>${q.id}</b> 全選択肢: ${base.choice}</p><table><tr><th>抜いた選択肢</th><th>残りの argmax</th><th>確信度</th></tr>`; for (const o of q.options) { const q2 = { ...q, options: q.options.filter(x => x !== o) }; const r = (await jev.decide(s, [q2], c, curOpts())).answers[0]; h += `<tr><td>${esc(o)}</td><td>${esc(r.choice)}</td><td>${(r.confidence * 100).toFixed(0)}%</td></tr>`; } h += "</table>"; }
  $("#v-drop-out").innerHTML = h || "Choice の質問がありません"; status(""); };
const loaded = {};
async function getModel(name) { if (jev && jev.name === name) return jev; if (!loaded[name]) loaded[name] = await new JevJa(name).load(status); return loaded[name]; }
let lastMatrix = null;
$("#v-cmp").onclick = async () => { const names = [...$("#v-models").selectedOptions].map(o => o.value); if (!names.length) return alert("モデルを選んでください"); const qs = readQuestions(), s = $("#p-state").value, c = ctx(); const T = +$("#T").value, cosScale = +$("#cosScale").value || 20; const rows = [];
  try { for (const n of names) { const m = await getModel(n); const r = await m.analyze(s, qs, c); for (const method of ["head", "cos"]) for (const pool of ["span", "marker"]) rows.push({ model: n, method, pool, ms: r.timing.total_ms, answers: JevJa.answersFrom(r, { method, pool, T, cosScale }) }); } } catch (e) { $("#v-cmp-out").textContent = e.message; status(""); return; }
  lastMatrix = { kind: "matrix", state: s, context: c, questions: qs, T, cosScale, rows: rows.map(r => ({ model: r.model, method: r.method, pool: r.pool, ms: r.ms, answers: r.answers.map(a => ({ id: a.id, label: a.choice ?? a.level ?? (a.noul ? "yes" : "no"), score: a.score, p_yes: a.p_yes, confidence: a.confidence, probabilities: a.distribution.probabilities })) })) };
  $("#v-cmp-out").innerHTML = "<table><tr><th>モデル</th><th>手法</th><th>pool</th>" + qs.map(q => `<th>${q.id}</th>`).join("") + "<th>ms</th></tr>" + rows.map(r => `<tr><td>${r.model}</td><td>${r.method}</td><td>${r.pool}</td>` + r.answers.map(a => `<td>${esc(a.choice ?? (a.score != null ? a.score.toFixed(2) + " / " + a.level : (a.noul ? "yes" : "no")))}<br><span class="muted small">${(a.confidence * 100).toFixed(0)}%</span></td>`).join("") + `<td>${r.ms.toFixed(0)}</td></tr>`).join("") + `</table><p class="muted small">T=${T.toFixed(2)}, cos×${cosScale}。未学習モデルの head 行はほぼ一様になる。</p>`; status(""); markStored(); };
$("#v-cmp-note").onclick = async () => { if (!lastMatrix) return; await saveNote({ kind: "matrix", name: prompt("実験名", "matrix " + new Date().toLocaleString()) || "matrix", ...lastMatrix }); alert("保存しました"); };
$("#v-bench").onclick = async () => { if (!ready()) return; const N = +$("#v-bn").value || 10; status("計測中…"); const qs = readQuestions(), s = $("#p-state").value, c = ctx(); const enc = [], hd = [], tot = []; for (let i = 0; i < N; i++) { const r = await jev.analyze(s, qs, c); enc.push(r.timing.encoder_ms); hd.push(r.timing.head_ms); tot.push(r.timing.total_ms); }
  const pct = (a, p) => { const s2 = [...a].sort((x, y) => x - y); return s2[Math.min(s2.length - 1, Math.floor((s2.length - 1) * p))]; };
  $("#v-bench-out").innerHTML = `<table><tr><th></th><th>p50</th><th>p95</th><th>min</th></tr><tr><td>エンコーダ</td><td>${pct(enc, .5).toFixed(0)}</td><td>${pct(enc, .95).toFixed(0)}</td><td>${Math.min(...enc).toFixed(0)}</td></tr><tr><td>ヘッド</td><td>${pct(hd, .5).toFixed(0)}</td><td>${pct(hd, .95).toFixed(0)}</td><td>${Math.min(...hd).toFixed(0)}</td></tr><tr><td><b>合計 ms</b></td><td><b>${pct(tot, .5).toFixed(0)}</b></td><td>${pct(tot, .95).toFixed(0)}</td><td>${Math.min(...tot).toFixed(0)}</td></tr></table><p class="muted small">${jev.name} / ${jev.backend} / ${N} 回 / ${navigator.userAgent.slice(0, 80)}</p>`; status(""); };

/* ---------------- batch ---------------- */
let batchName = "";
function loadBatchText(txt, name) { batchItems = txt.split("\n").map(s => s.trim()).filter(Boolean).map(s => JSON.parse(s)); batchName = name; const fams = [...new Set(batchItems.map(i => i.family))]; $("#b-info").textContent = `${name}: ${batchItems.length} state / ${batchItems.reduce((n, it) => n + it.questions.length, 0)} question / family: ${fams.join(", ")}`; $("#b-run").disabled = !jev; }
$("#b-file").onchange = async e => { const f = e.target.files[0]; if (f) loadBatchText(await f.text(), f.name); };
$("#b-sample").onclick = async () => { const f = $("#b-ds").value; loadBatchText(await (await fetch("bench/" + f)).text(), f); };
(async () => { try { const idx = await (await fetch("bench/index.json")).json(); for (const f of idx.files) { const o = document.createElement("option"); o.value = f.file; o.textContent = `${f.file} — ${f.desc || ""} (${f.states} state)`; $("#b-ds").appendChild(o); } } catch { } })();
const selectedBatch = () => { const fam = $("#b-family").value.split(",").map(s => s.trim()).filter(Boolean); const lim = +$("#b-limit").value || 0; let it = fam.length ? batchItems.filter(i => fam.includes(i.family)) : batchItems; return lim ? it.slice(0, lim) : it; };
$("#b-run").onclick = async () => { if (!ready()) return; const rows = [], lat = [], perRow = []; const useCtx = $("#b-ctx").checked; let i = 0; const items = selectedBatch(); if (!items.length) return alert("該当データがありません");
  for (const it of items) { status(`評価中 ${++i}/${items.length}`); const r = await jev.decide(it.state, it.questions, useCtx ? it.context : null, curOpts()); lat.push(r.latency_ms);
    it.questions.forEach((q, k) => { const a = r.answers[k]; let gold = it.gold[q.id]; if (q.type === "noul") gold = ["yes", "true", "1"].includes(String(gold).toLowerCase()) ? "yes" : "no"; const labels = a.distribution.labels, p = a.distribution.probabilities, gi = labels.indexOf(gold), pi = p.indexOf(Math.max(...p)); const row = { family: it.family || "?", type: q.type, correct: pi === gi, conf: p[pi], brier: p.reduce((s, x, j) => s + (x - (j === gi ? 1 : 0)) ** 2, 0) }; if (q.type === "score") { const vals = q.values || labels.map((_, j) => j); row.mae = Math.abs(p.reduce((s, x, j) => s + x * vals[j], 0) - vals[gi]); } rows.push(row); perRow.push({ id: it.id, family: it.family, q: q.id, type: q.type, gold, pred: labels[pi], conf: p[pi], correct: row.correct, ms: r.latency_ms }); }); }
  const m = metrics(rows, lat); const o = curOpts(); batchResult = { data: batchName + ($("#b-family").value ? " [" + $("#b-family").value + "]" : ""), device: navigator.userAgent, results: [{ model: jev.name + " (" + optsLabel(o) + ", device)", probability_kind: "native", use_context: useCtx, backend: jev.backend, opts: o, ...m }], rows: perRow };
  await saveNote({ kind: "batch", name: $("#b-name").value || `${jev.name} ${optsLabel(o)} ${batchName}`, model: jev.name, method: o.method, pool: o.pool, T: o.T, cosScale: o.cosScale, dataset: batchResult.data, n_states: items.length, use_context: useCtx, backend: jev.backend, accuracy: m.accuracy, brier: m.brier, ece: m.ece, latency_p50_ms: m.latency_p50_ms, latency_p95_ms: m.latency_p95_ms, n_questions: m.n_questions, per_family: m.per_family, ece_table: m.ece_table, rows: perRow });
  $("#b-out").innerHTML = `<table><tr><th>モデル</th><th>精度</th><th>Brier</th><th>ECE</th><th>p50 ms</th><th>p95 ms</th><th>n</th></tr><tr><td>${jev.name}</td><td>${(m.accuracy * 100).toFixed(1)}%</td><td>${m.brier.toFixed(3)}</td><td>${m.ece.toFixed(3)}</td><td>${fmt(m.latency_p50_ms, 0)}</td><td>${fmt(m.latency_p95_ms, 0)}</td><td>${m.n_questions}</td></tr></table><h2>family 別</h2><table><tr><th>family</th><th>n</th><th>精度</th><th>MAE</th></tr>${Object.entries(m.per_family).map(([k, v]) => `<tr><td>${k}</td><td>${v.n}</td><td>${(v.accuracy * 100).toFixed(1)}%</td><td>${v.mae != null ? v.mae.toFixed(2) : ""}</td></tr>`).join("")}</table><h2>較正（信頼度ビン）</h2><table><tr><th>bin</th><th>n</th><th>conf</th><th>acc</th></tr>${m.ece_table.filter(b => b.n).map(b => `<tr><td>${b.bin}</td><td>${b.n}</td><td>${b.confidence.toFixed(2)}</td><td>${b.accuracy.toFixed(2)}</td></tr>`).join("")}</table><details><summary>行ごとの結果（${perRow.length}）</summary><table><tr><th>id</th><th>q</th><th>gold</th><th>pred</th><th>conf</th><th>✓</th></tr>${perRow.map(r => `<tr><td>${r.id}</td><td>${r.q}</td><td>${esc(r.gold)}</td><td>${esc(r.pred)}</td><td>${r.conf.toFixed(2)}</td><td>${r.correct ? "✓" : ""}</td></tr>`).join("")}</table></details>`;
  drawCal(m.ece_table); status(""); };
function drawCal(bins) { const c = $("#b-cal"); c.classList.remove("hidden"); const g = c.getContext("2d"); const W = c.width, H = c.height, P = 36; g.clearRect(0, 0, W, H); const dark = matchMedia("(prefers-color-scheme:dark)").matches; g.strokeStyle = dark ? "#9aa3ad" : "#6b7280"; g.fillStyle = dark ? "#e6e8eb" : "#1a1d21"; g.font = "12px system-ui";
  g.beginPath(); g.moveTo(P, H - P); g.lineTo(W - P, H - P); g.lineTo(W - P, P); g.stroke(); g.setLineDash([4, 4]); g.beginPath(); g.moveTo(P, H - P); g.lineTo(W - P, P); g.stroke(); g.setLineDash([]); g.fillText("stated confidence →", P, H - 10); g.save(); g.translate(12, H - P); g.rotate(-Math.PI / 2); g.fillText("accuracy →", 0, 0); g.restore();
  const tot = bins.reduce((s, b) => s + b.n, 0); for (const b of bins) if (b.n) { const x = P + b.confidence * (W - 2 * P), y = H - P - b.accuracy * (H - 2 * P), r = 4 + 14 * Math.sqrt(b.n / tot); g.fillStyle = "#0b6e99aa"; g.beginPath(); g.arc(x, y, r, 0, 7); g.fill(); } }
$("#b-dl").onclick = () => { if (batchResult) download(`bench-device-${Date.now()}.json`, JSON.stringify(batchResult, null, 1)); };
$("#b-csv").onclick = () => { if (!batchResult) return; const rows = batchResult.rows; download("bench-rows.csv", "﻿" + [Object.keys(rows[0]).join(",")].concat(rows.map(r => Object.values(r).map(v => `"${String(v).replace(/"/g, '""')}"`).join(","))).join("\n"), "text/csv"); };

/* ---------------- OSINT（メイン） ---------------- */
$("#o-src").innerHTML = Object.entries(SOURCES).map(([k, s]) => `<label style="display:block"><input type="checkbox" class="o-s" value="${k}" ${["jma", "wikipedia", "nominatim", "bluesky"].includes(k) ? "checked" : ""} style="width:auto"> <b>${s.name}</b> <span class="muted small">${s.desc}</span></label>`).join("");
let oItems = [], oJudged = null, oGeo = [], oClusters = null, oLabels = {};
const W = { ...TRUST_WEIGHTS_DEFAULT }; const WN = { official: "公的発表", supported: "根拠と整合", has_specifics: "具体性", kind_report: "種類", no_spread: "拡散依頼なし", relevant: "関連" };
$("#o-w").innerHTML = Object.keys(W).map(k => `<span>${WN[k]}</span><span><input type="range" class="o-wk" data-k="${k}" min="0" max="1" step="0.05" value="${W[k]}" style="width:60%"> <b class="o-wv" data-k="${k}">${W[k].toFixed(2)}</b></span>`).join("");
$$(".o-wk").forEach(r => r.oninput = () => { W[r.dataset.k] = +r.value; $(`.o-wv[data-k=${r.dataset.k}]`).textContent = (+r.value).toFixed(2); if (oJudged) renderOsint(); });
async function savedQueries() { try { const q = JSON.parse(localStorage.getItem("jev.queries") || "[]"); $("#o-saved").innerHTML = q.map(x => `<option value="${esc(x)}">`).join(""); return q; } catch { return []; } }
savedQueries();
$("#o-run").onclick = async () => { const q = $("#o-q").value.trim(); if (!q) return; const srcs = $$(".o-s:checked").map(c => c.value); $("#o-run").disabled = true; oJudged = null; oClusters = null; oLabels = {}; $("#o-diff").textContent = "";
  try { const qs = await savedQueries(); localStorage.setItem("jev.queries", JSON.stringify([q, ...qs.filter(x => x !== q)].slice(0, 20))); } catch { }
  oItems = await collect(q, srcs, { limit: +$("#o-limit").value || 5, prefCode: $("#o-pref").value, lang: $("#o-lang").value }, status);
  // 前回との差分（新着検出）
  try { const key = "osint:seen:" + q; const seen = new Set(await store.get(key) || []); const now = oItems.filter(i => !i.error).map(i => i.url); const fresh = now.filter(u => !seen.has(u)); oItems.forEach(i => { if (!i.error) i.is_new = !seen.has(i.url); }); await store.put(key, [...new Set([...seen, ...now])]); $("#o-diff").textContent = seen.size ? `前回の収集から新着 ${fresh.length} 件（🆕 印）` : "初回の収集（次回から新着を検出）"; } catch { }
  if ($("#o-geo").checked) { status("地理照合中…"); oGeo = await geoContext(oItems); }
  if ($("#o-cluster").checked && jev) { status("類似投稿をまとめ中…"); const idx = oItems.map((it, i) => i).filter(i => !oItems[i].error); const vecs = []; for (const i of idx) vecs.push(await jev.embed(`${oItems[i].title} ${oItems[i].text}`)); const cl = clusterByCos(vecs, +$("#o-th").value || 0.9); oClusters = {}; idx.forEach((i, n) => oClusters[i] = cl[n]); }
  $("#o-run").disabled = false; $("#o-judge").disabled = !jev; $("#o-pack").disabled = $("#o-csv").disabled = $("#o-label").disabled = false; status(""); renderOsint(); };
function renderOsint() {
  const nOk = oItems.filter(i => !i.error).length; const nCl = oClusters ? new Set(Object.values(oClusters)).size : null;
  let h = `<p>${nOk} 件取得（エラー ${oItems.filter(i => i.error).length}）${nCl != null ? ` ・ 類似クラスタ ${nCl}` : ""}${oGeo.length ? ` ・ 地理根拠 ${oGeo.length} 件` : ""}</p>`;
  if (oGeo.length) h += `<details><summary>地理照合の根拠（context として注入）</summary><ul>${oGeo.map(g => `<li class="small">${esc(g)}</li>`).join("")}</ul></details>`;
  h += `<table><tr><th>#</th><th>出典</th><th>内容</th><th>時刻</th>` + (oClusters ? "<th>群</th>" : "") + (oJudged ? "<th>信頼性</th><th>関連</th><th>種類</th><th>緊急</th><th>具体</th><th>拡散</th><th>整合</th>" : "") + "<th>人手ラベル</th></tr>";
  const ts = oJudged ? oItems.map((it, i) => trustScore(it, oJudged[i], W)) : null;
  const order = [...oItems.keys()].sort((x, y) => ts ? ((ts[y]?.score ?? -1) - (ts[x]?.score ?? -1)) : 0);
  for (const i of order) { const it = oItems[i]; if (it.error) { h += `<tr><td>${i}</td><td>${it.source}</td><td class="muted" colspan=12>取得失敗: ${esc(it.error).slice(0, 120)}</td></tr>`; continue; }
    h += `<tr><td>${i}</td><td>${it.official ? "🏛 " : ""}${it.source}${it.is_new ? " 🆕" : ""}<br><a href="${it.url}" target="_blank" rel="noopener" class="small">開く</a></td><td><b>${esc(it.title || "")}</b><br><span class="small">${esc((it.text || "").slice(0, 240))}</span>${it.elevation_m != null ? `<br><span class="pill">標高 ${it.elevation_m.toFixed(0)}m</span>` : ""}</td><td class="small">${(it.time || "").slice(0, 16)}</td>`;
    if (oClusters) h += `<td>${oClusters[i] ?? ""}</td>`;
    if (oJudged) { const j = oJudged[i], t = ts[i]; if (!j) h += "<td colspan=7></td>"; else { const ny = a => a ? (a.noul ? "✅" : "—") + ` <span class="muted small">${a.p_yes.toFixed(2)}</span>` : ""; h += `<td><b>${(t.score * 100).toFixed(0)}</b><div class="bar"><i style="width:${t.score * 100}%"></i></div><span class="muted small" title="${Object.entries(t.parts).map(([k, v]) => WN[k] + "=" + v.toFixed(2)).join(", ")}">内訳▸</span></td><td>${ny(j.relevant)}</td><td>${j.kind?.choice || ""}<br><span class="muted small">${j.kind ? (j.kind.confidence * 100).toFixed(0) + "%" : ""}</span></td><td>${j.urgency ? j.urgency.score.toFixed(1) : ""}</td><td>${ny(j.has_specifics)}</td><td>${ny(j.asks_spread)}</td><td>${ny(j.supported)}</td>`; } }
    h += `<td><select class="o-lab" data-i="${i}" style="width:auto"><option value="">—</option>${["公的機関の発表", "報道", "一般の投稿・目撃", "意見・感想", "宣伝・無関係"].map(k => `<option ${oLabels[i]?.kind === k ? "selected" : ""}>${k}</option>`).join("")}</select> <select class="o-lab-t" data-i="${i}" style="width:auto"><option value="">信頼?</option>${["高", "中", "低"].map(k => `<option ${oLabels[i]?.trust === k ? "selected" : ""}>${k}</option>`).join("")}</select></td></tr>`; }
  $("#o-out").innerHTML = h + "</table>";
  $$(".o-lab").forEach(sel => sel.onchange = () => { (oLabels[sel.dataset.i] ??= {}).kind = sel.value; }); $$(".o-lab-t").forEach(sel => sel.onchange = () => { (oLabels[sel.dataset.i] ??= {}).trust = sel.value; }); }
$("#o-judge").onclick = async () => { if (!ready()) return; const q = $("#o-q").value; const qs = OSINT_QUESTIONS(q); const official = [...oItems.filter(i => !i.error && (i.official || i.source === "wikipedia")).map(i => `${i.title}: ${(i.text || "").slice(0, 200)}`).slice(0, 4), ...oGeo.slice(0, 3)]; oJudged = [];
  for (let i = 0; i < oItems.length; i++) { const it = oItems[i]; if (it.error) { oJudged.push(null); continue; } status(`整理中 ${i + 1}/${oItems.length}`); const ctxs = official.filter(c => !c.startsWith(it.title + ":")); const r = await jev.decide(`${it.title}\n${it.text}`, qs, ctxs.length ? ctxs : null, curOpts()); const j = {}; for (const a of r.answers) j[a.id] = a; j._ms = r.latency_ms; oJudged.push(j); }
  status(""); $("#o-note").disabled = false; renderOsint(); };
const judgedPlain = () => oJudged?.map(j => j && Object.fromEntries(Object.entries(j).filter(([k]) => k !== "_ms").map(([k, a]) => [k, { label: a.choice ?? a.level ?? (a.noul ? "yes" : "no"), score: a.score, p_yes: a.p_yes, confidence: a.confidence }])));
$("#o-pack").onclick = () => download(`evidence-${Date.now()}.json`, JSON.stringify({ ...evidencePack($("#o-q").value, oItems, judgedPlain()), geo_context: oGeo, clusters: oClusters, trust_weights: W, human_labels: oLabels }, null, 1));
$("#o-csv").onclick = () => { const cell = v => `"${String(v ?? "").replace(/"/g, '""')}"`; const ts = oJudged ? oItems.map((it, i) => trustScore(it, oJudged[i], W)) : null; const rows = oItems.map((it, i) => { const j = oJudged?.[i]; return [i, it.source, it.title, it.text, it.time, it.url, it.fetched_at, it.is_new ? 1 : 0, oClusters?.[i], it.elevation_m, ts?.[i]?.score?.toFixed(3), j?.relevant?.p_yes?.toFixed(2), j?.kind?.choice, j?.urgency?.score?.toFixed(2), j?.has_specifics?.p_yes?.toFixed(2), j?.asks_spread?.p_yes?.toFixed(2), j?.supported?.p_yes?.toFixed(2), oLabels[i]?.kind, oLabels[i]?.trust].map(cell).join(","); }); download("osint.csv", "﻿" + ["i,source,title,text,time,url,fetched_at,is_new,cluster,elevation_m,trust,relevant,kind,urgency,has_specifics,asks_spread,supported,label_kind,label_trust"].concat(rows).join("\n"), "text/csv"); };
$("#o-note").onclick = async () => { const o = curOpts(); await saveNote({ kind: "osint", name: `OSINT: ${$("#o-q").value}`, model: jev.name, method: o.method, pool: o.pool, T: o.T, cosScale: o.cosScale, query: $("#o-q").value, n_items: oItems.length, trust_weights: { ...W }, items: evidencePack($("#o-q").value, oItems, null).items, judged: judgedPlain(), human_labels: oLabels }); alert("保存しました"); };
/* 人手ラベル → 学習用 JSONL（JevBench-JA 形式）。自分の OSINT データセットを育てる */
$("#o-label").onclick = () => { const q = $("#o-q").value; const lines = []; oItems.forEach((it, i) => { const L = oLabels[i]; if (it.error || !L || (!L.kind && !L.trust)) return; const qs = [], gold = {}; if (L.kind) { qs.push({ type: "choice", id: "kind", instructions: "この情報の種類", options: ["公的機関の発表", "報道", "一般の投稿・目撃", "意見・感想", "宣伝・無関係"] }); gold.kind = L.kind; } if (L.trust) { qs.push({ type: "score", id: "trust", instructions: "この情報の信頼性", levels: ["低", "中", "高"] }); gold.trust = L.trust; } lines.push(JSON.stringify({ id: `osint-${Date.now()}-${i}`, family: "osint_ja", split: "in_domain", state: `${it.title}\n${it.text}`, context: oGeo.slice(0, 3), meta: { query: q, source: it.source, url: it.url, time: it.time }, questions: qs, gold })); });
  if (!lines.length) return alert("ラベルを付けた行がありません（表の右端で選ぶ）"); download(`osint_ja-${Date.now()}.jsonl`, lines.join("\n") + "\n", "application/jsonl"); };

/* ---------------- 実験ノート ---------------- */
async function saveNote(e) { e.id = e.id || `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`; e.time = e.time || new Date().toISOString(); e.device = e.device || navigator.userAgent; e.webgpu = !!navigator.gpu; e.notes = e.notes || ""; await store.expPut(e); return e.id; }
$("#p-note").onclick = async () => { if (!last) return; const o = curOpts(); await saveNote({ kind: "decide", name: prompt("実験名", `${jev.name} ${optsLabel(o)}`) || "decide", model: jev.name, method: o.method, pool: o.pool, T: o.T, cosScale: o.cosScale, input: last.input, n_tokens: last.n_tokens, timing: last.timing, answers: last.answers.map(a => ({ id: a.id, type: a.type, label: a.choice ?? a.level ?? (a.noul ? "yes" : "no"), score: a.score, p_yes: a.p_yes, confidence: a.confidence, labels: a.distribution.labels, probabilities: a.distribution.probabilities, logits: a.logits })) }); alert("実験ノートに保存しました"); };
async function refreshNotes() { const es = (await store.expAll()).sort((x, y) => y.time.localeCompare(x.time)); if (!es.length) { $("#n-list").innerHTML = '<p class="muted">まだ記録がありません。バッチ評価を実行するか、Playground の「実験ノートに保存」を押す。</p>'; return; }
  $("#n-list").innerHTML = "<table><tr><th>時刻</th><th>種別</th><th>名前</th><th>モデル</th><th>設定</th><th>データ / 入力</th><th>精度</th><th>Brier</th><th>ECE</th><th>p50 ms</th><th>メモ</th><th></th></tr>" + es.map(e => `<tr><td class="small">${e.time.replace("T", " ").slice(0, 16)}</td><td>${e.kind}</td><td>${esc(e.name)}</td><td>${e.model || (e.rows ? [...new Set(e.rows.map(r => r.model))].join("/") : "")}</td><td class="small">${e.method ? optsLabel(e) : ""}</td><td class="small">${esc(e.dataset || (e.input?.state || e.state || "").slice(0, 40))}</td><td>${e.accuracy != null ? (e.accuracy * 100).toFixed(1) + "%" : ""}</td><td>${e.brier != null ? e.brier.toFixed(3) : ""}</td><td>${e.ece != null ? e.ece.toFixed(3) : ""}</td><td>${e.latency_p50_ms != null ? e.latency_p50_ms.toFixed(0) : (e.timing ? e.timing.total_ms.toFixed(0) : "")}</td><td><input data-id="${e.id}" class="n-memo" value="${esc(e.notes || "")}" placeholder="メモ"></td><td><button data-id="${e.id}" class="n-dl">JSON</button> <button data-id="${e.id}" class="n-del">✕</button></td></tr>`).join("") + "</table>";
  $$(".n-del").forEach(b => b.onclick = async () => { await store.expDel(b.dataset.id); refreshNotes(); }); $$(".n-dl").forEach(b => b.onclick = async () => { const e = (await store.expAll()).find(x => x.id === b.dataset.id); download(`exp-${e.id}.json`, JSON.stringify(e, null, 1)); });
  $$(".n-memo").forEach(i => i.onchange = async () => { const e = (await store.expAll()).find(x => x.id === i.dataset.id); e.notes = i.value; await store.expPut(e); }); }
$("#n-csv").onclick = async () => { const es = await store.expAll(); const cols = ["time", "kind", "name", "model", "method", "pool", "T", "cosScale", "dataset", "n_states", "n_questions", "use_context", "backend", "accuracy", "brier", "ece", "latency_p50_ms", "latency_p95_ms", "notes"]; download("experiments.csv", "\ufeff" + [cols.join(",")].concat(es.map(e => cols.map(c => `"${String(e[c] ?? "").replace(/"/g, '""')}"`).join(","))).join("\n"), "text/csv"); };
$("#n-json").onclick = async () => download(`experiments-${Date.now()}.json`, JSON.stringify(await store.expAll(), null, 1));
$("#n-import").onchange = async e => { const f = e.target.files[0]; if (!f) return; const es = JSON.parse(await f.text()); for (const x of (Array.isArray(es) ? es : [es])) await store.expPut(x); refreshNotes(); };
$("#n-clear").onclick = async () => { if (!confirm("実験ノートを全削除しますか？")) return; for (const e of await store.expAll()) await store.expDel(e.id); refreshNotes(); };

/* ---------------- settings ---------------- */
async function refreshSettings() { const ks = await store.keys(); const by = {}; for (const k of ks) { const [m] = k.split("/"); by[m] = (by[m] || 0) + ((await store.get(k))?.byteLength || 0); }
  $("#s-models").innerHTML = Object.keys(by).length ? "<table><tr><th>モデル</th><th>サイズ</th><th></th></tr>" + Object.entries(by).map(([m, b]) => `<tr><td>${m}</td><td>${(b / 1e6).toFixed(1)} MB</td><td><button data-m="${m}" class="s-del">削除</button></td></tr>`).join("") + "</table>" : "保存されたモデルはありません";
  $$(".s-del").forEach(b => b.onclick = async () => { await removeStored(b.dataset.m); refreshSettings(); markStored(); storageInfo(); });
  $("#s-env").innerHTML = `<span>UA</span><span class="small">${navigator.userAgent}</span><span>WebGPU</span><span>${navigator.gpu ? "あり" : "なし（WASM で実行）"}</span><span>CPU 論理コア</span><span>${navigator.hardwareConcurrency ?? "?"}</span><span>メモリ目安</span><span>${navigator.deviceMemory ? navigator.deviceMemory + " GB" : "?"}</span><span>Service Worker</span><span>${navigator.serviceWorker?.controller ? "制御中" : "未"}</span>`; }
$("#s-persist").onclick = async () => { const ok = await navigator.storage.persist?.(); alert(ok ? "永続化されました" : "許可されませんでした（ホーム画面に追加すると通ることがあります）"); storageInfo(); };
$("#s-clear").onclick = async () => { if (!confirm("保存したモデルをすべて削除しますか？")) return; for (const k of await store.keys()) await store.del(k); refreshSettings(); markStored(); storageInfo(); };

/* ---------------- boot ---------------- */
applyTemplate(TEMPLATES[0]); $("#p-tmpl").value = TEMPLATES[0].id;
try { const t = localStorage.getItem("jev.tab"); if (t) { const b = $(`.tabs button[data-t=${t}]`); if (b) b.click(); } } catch { }
await loadIndex(); storageInfo();
try { const lastM = localStorage.getItem("jev.lastModel"); if (lastM && models.some(m => m.name === lastM)) $("#model").value = lastM; } catch { }
if ("serviceWorker" in navigator) navigator.serviceWorker.register("sw.js").then(() => setTimeout(offlineCheck, 800));
try { if (localStorage.getItem("jev.lastModel") && await isStored($("#model").value)) $("#load").click(); } catch { }
