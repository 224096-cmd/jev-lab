/* 画面共通の小さな部品（v2.0）
   - $, $$, esc, pc, fmt, download
   - ring(score): 信頼性・確信度の円ゲージ（SVG）
   - confCls / confTxt: 確信度の色と文言
   - answerLine(a, q): 1 問の答えを 1 行で（ラベル・確信度・分布バー）
   - gate(conf, th): 確信度ゲート → "auto" | "check"
   - toast(msg): 画面下の短い通知 */
export const $ = s => document.querySelector(s), $$ = s => [...document.querySelectorAll(s)];
export const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
export const pc = x => (x * 100).toFixed(0) + "%";
export const fmt = (x, d = 1) => (x == null || isNaN(x)) ? "–" : (+x).toFixed(d);
export const download = (name, data, type = "application/json") => { const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([data], { type })); a.download = name; a.click(); };
export const lab = a => a.choice ?? (a.score != null ? a.score.toFixed(2) : (a.noul ? "yes" : "no"));
export const labJa = a => a.choice ?? (a.level != null ? a.level : (a.noul ? "はい" : "いいえ"));
export const confCls = c => c >= 0.7 ? "hi" : c >= 0.5 ? "mid" : "lo";
export const confTxt = c => (c >= 0.7 ? "確信 " : c >= 0.5 ? "やや " : "不確か ") + pc(c);
export const gate = (conf, th) => conf >= th ? "auto" : "check";
export const isMobile = /Android|iPhone|iPad/i.test(navigator.userAgent) || (navigator.deviceMemory && navigator.deviceMemory <= 4);
/* 円ゲージ：score 0..1、色は 70 以上=緑、40 未満=赤 */
export function ring(score, { size = 52, label, sub } = {}) { const r = (size - 8) / 2, c = 2 * Math.PI * r, s = Math.max(0, Math.min(1, score ?? 0)); const col = score == null ? "var(--line)" : s >= 0.7 ? "var(--ok)" : s < 0.4 ? "var(--bad)" : "var(--warn)"; return `<div class="ring" style="width:${size}px;height:${size}px"><svg viewBox="0 0 ${size} ${size}"><circle cx="${size / 2}" cy="${size / 2}" r="${r}" stroke="var(--line)" stroke-width="5" fill="none"/><circle cx="${size / 2}" cy="${size / 2}" r="${r}" stroke="${col}" stroke-width="5" fill="none" stroke-linecap="round" stroke-dasharray="${(c * s).toFixed(1)} ${c.toFixed(1)}" transform="rotate(-90 ${size / 2} ${size / 2})"/></svg><b>${label ?? (score == null ? "–" : Math.round(s * 100))}</b>${sub ? `<span>${esc(sub)}</span>` : ""}</div>`; }
/* 1 問の答え：質問 → 答え（確信度）→ 分布 */
export function answerLine(a, q, { open = false } = {}) { const d = a.distribution; const v = a.type === "choice" ? esc(a.choice) : a.type === "score" ? `${esc(a.level)} <span class="muted small">(${a.score.toFixed(1)})</span>` : (a.noul ? "はい" : "いいえ"); const abst = a.type === "choice" && /該当なし|判断できない|不明/.test(a.choice || "");
  return `<details class="ans ${abst ? "abst" : ""}" ${open || a.confidence < 0.5 ? "open" : ""}><summary><span class="qn">${esc(q?.instructions || a.id)}</span><b class="av">${v}</b><span class="conf ${confCls(a.confidence)}">${pc(a.confidence)}</span></summary><div class="dist">${d.labels.map((l, i) => `<span>${esc(l)}</span><div class="bar"><i style="width:${d.probabilities[i] * 100}%"></i></div><span class="muted">${(d.probabilities[i] * 100).toFixed(0)}%</span>`).join("")}</div>${a.type === "noul" ? `<div class="muted small">p(はい) = ${a.p_yes.toFixed(2)}</div>` : ""}</details>`; }
let toastT = null; export function toast(msg, ms = 2600) { let el = $("#toast"); if (!el) { el = document.createElement("div"); el.id = "toast"; document.body.appendChild(el); } el.textContent = msg; el.classList.add("show"); clearTimeout(toastT); toastT = setTimeout(() => el.classList.remove("show"), ms); }
export const status = t => { const el = $("#status"); if (el) el.textContent = t || ""; const m = $("#status2"); if (m) { m.textContent = t || ""; m.classList.toggle("hidden", !t); } };
export const progress = msg => { status(msg); const m = /([\d.]+) ?\/ ?([\d.]+) MB/.exec(msg || ""); const p = $("#progbar"); if (p) p.style.width = m ? (100 * +m[1] / +m[2]).toFixed(0) + "%" : (msg ? "30%" : "0%"); };
export const favicon = u => { try { const h = new URL(u).hostname; return `https://www.google.com/s2/favicons?domain=${h}&sz=32`; } catch { return ""; } };
export const hostOf = u => { try { return new URL(u).hostname.replace(/^www\./, ""); } catch { return ""; } };
export const ago = t => { if (!t) return ""; const d = (Date.now() - new Date(t).getTime()) / 6e4; if (isNaN(d)) return String(t).slice(0, 10); if (d < 60) return `${Math.max(1, Math.round(d))} 分前`; if (d < 1440) return `${Math.round(d / 60)} 時間前`; if (d < 43200) return `${Math.round(d / 1440)} 日前`; return String(t).slice(0, 10); };
export const copy = async (t, msg = "コピーしました") => { try { await navigator.clipboard.writeText(t); toast(msg); } catch { toast("コピーできませんでした"); } };
