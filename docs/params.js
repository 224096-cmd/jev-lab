/* 仕組みのパラメータ（v3.0）— 画面「研究 → 仕組み」で閲覧・変更できる。localStorage に保存。
   PARAMS: 既定値。get(key) は上書きがあればそれを返す。set(key, v) で保存、reset() で既定に戻す */
import { TRUST_WEIGHTS_DEFAULT } from "./osint.js";
export const DEFAULTS = {
  gate: 0.7,                 // 確信度ゲート：これ未満は必ず「人が確認」
  trust_hi: 0.70,            // 信頼性がこれ以上（かつ要確認フラグなし）→ 自動採用
  trust_lo: 0.40,            // 信頼性がこれ未満 → 除外
  contra_p: 0.35,            // 「根拠と整合」の p(はい) がこれ未満 → 矛盾の疑い
  urgent_post: 3.0,          // 一般投稿でこの緊急度以上 → 要確認
  read_bodies: true,         // 上位の件の本文を Reader で読んで判定し直す
  read_n: 6,                 // 本文を読む件数
  read_chars: 1500,          // 本文の先頭何文字を読むか
  dorks_auto: 2,             // 収集時に検索エンジンで自動実行する検索式の本数（メイン語 + プランの上位 n 本）
  per_source: 5,             // 情報源ごとの取得件数
  T: 1.0,                    // 温度（全モデル共通の倍率）
  key_points: 3,             // 要点として抜き出す文の数
  weights: { ...TRUST_WEIGHTS_DEFAULT },
};
const K = "jev.params";
let cur = load();
function load() { try { const o = JSON.parse(localStorage.getItem(K) || "{}"); return { ...DEFAULTS, ...o, weights: { ...DEFAULTS.weights, ...(o.weights || {}) } }; } catch { return structuredClone(DEFAULTS); } }
export const P = new Proxy({}, { get: (_, k) => cur[k] });
export function setParam(k, v) { cur[k] = v; try { const o = JSON.parse(localStorage.getItem(K) || "{}"); o[k] = v; localStorage.setItem(K, JSON.stringify(o)); } catch { } }
export function resetParams() { try { localStorage.removeItem(K); } catch { } cur = load(); }
export function allParams() { return structuredClone(cur); }
/* 質問セットの上書き（仕組みページで JSON 編集）。key: osint / verify / recon / site */
const QK = "jev.questions";
export function questionOverride(key) { try { return JSON.parse(localStorage.getItem(QK) || "{}")[key] || null; } catch { return null; } }
export function setQuestionOverride(key, qs) { try { const o = JSON.parse(localStorage.getItem(QK) || "{}"); if (qs) o[key] = qs; else delete o[key]; localStorage.setItem(QK, JSON.stringify(o)); } catch { } }
/* 説明（仕組みページに表示） */
export const PARAM_DOC = {
  gate: ["確信度ゲート", "答えの最大確率がこの値未満なら、その件は必ず「人が確認」へ落とす（TypeSafe の confidence-gated routing）。0.5〜0.9"],
  trust_hi: ["自動採用のしきい値", "信頼性スコアがこれ以上で要確認フラグが無ければ「自動採用」"],
  trust_lo: ["除外のしきい値", "信頼性スコアがこれ未満は「除外」（読まなくてよい）"],
  contra_p: ["矛盾の判定", "「根拠と整合しているか」の p(はい) がこれ未満で、話題に関係していれば「矛盾の疑い」"],
  urgent_post: ["目撃投稿の緊急度", "一般の投稿・目撃で緊急度がこれ以上なら「人が確認」"],
  read_bodies: ["本文を読む", "ON なら上位の件の本文を Reader（r.jina.ai）で取得し、本文で判定し直す（OFF は見出し＋抄録だけ）"],
  read_n: ["本文を読む件数", "信頼性上位から何件読むか（Reader は 1 分 20 回まで）"],
  read_chars: ["本文の文字数", "本文の先頭何文字をモデルに渡すか（長いほど遅い）"],
  dorks_auto: ["自動実行する検索式", "収集のとき、検索エンジンでメインの語に加えてプランの検索式を何本自動実行するか（0 で無効）"],
  per_source: ["情報源ごとの件数", "各 API から取る件数"],
  T: ["温度 T", "確率の鋭さ。1 より大きいと平ら（確信度が下がる）、小さいと尖る"],
  key_points: ["要点の文数", "原文から抜き出す文の数"],
};
