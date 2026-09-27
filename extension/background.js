// 設定（JEV Lab の URL）。自分で GitHub Pages に置いた URL に変える
const DEFAULT_APP = "https://224096-cmd.github.io/jev-lab/play.html";
const appUrl = async () => (await chrome.storage.sync.get({ app: DEFAULT_APP })).app;
chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({ id: "jev-judge", title: "JEV で判断（仕分け・信頼性）", contexts: ["selection"] });
  chrome.contextMenus.create({ id: "jev-recon", title: "JEV で調査（このリンク／ページのドメイン）", contexts: ["link", "page"] });
  chrome.contextMenus.create({ id: "jev-collect", title: "JEV で収集（選択語を話題として）", contexts: ["selection"] });
  chrome.contextMenus.create({ id: "jev-results", title: "この検索結果ページを JEV に取り込む（Google / Bing / DDG / Yahoo）", contexts: ["page"], documentUrlPatterns: ["*://www.google.com/search*", "*://www.google.co.jp/search*", "*://www.bing.com/search*", "*://duckduckgo.com/*", "*://html.duckduckgo.com/*", "*://search.yahoo.co.jp/*"] });
  chrome.sidePanel.setPanelBehavior({ openPanelOnActionClick: true });
});
async function openPanel(tab, params) {
  const base = await appUrl(); const url = base + "?" + new URLSearchParams(params).toString();
  await chrome.storage.session.set({ pending: url });
  try { await chrome.sidePanel.open({ tabId: tab.id }); } catch { chrome.tabs.create({ url }); }
}
/* 検索結果ページから結果（タイトル・URL・抜粋）を抜き出す。利用者が自分で開いたページを、同じブラウザで読むだけ（自動アクセスはしない） */
function scrapeResults() {
  const out = []; const seen = new Set(); const add = (a, snip) => { if (!a || !a.href || !/^https?:/.test(a.href) || seen.has(a.href)) return; if (/google\.|bing\.com|duckduckgo\.com|yahoo\.co\.jp|yahoo\.com/.test(new URL(a.href).hostname)) return; seen.add(a.href); out.push({ title: (a.textContent || "").trim().slice(0, 120), url: a.href, text: (snip || "").trim().slice(0, 300) }); };
  const h = location.hostname;
  if (/google\./.test(h)) document.querySelectorAll("div.g, div[data-hveid] > div").forEach(g => { const a = g.querySelector("a[href] h3")?.closest("a"); const s = g.querySelector('[data-sncf], [style*="-webkit-line-clamp"], .VwiC3b'); if (a) add(a, s?.textContent); });
  else if (/bing\.com/.test(h)) document.querySelectorAll("li.b_algo").forEach(li => { const a = li.querySelector("h2 a"); add(a, li.querySelector(".b_caption p, p")?.textContent); });
  else if (/duckduckgo/.test(h)) document.querySelectorAll("article, .result").forEach(r => { const a = r.querySelector('a[data-testid="result-title-a"], a.result__a, h2 a'); add(a, r.querySelector('[data-result="snippet"], .result__snippet')?.textContent); });
  else if (/yahoo\.co\.jp/.test(h)) document.querySelectorAll(".sw-Card, .Algo").forEach(c => { const a = c.querySelector("h3 a, .sw-Card__title a"); add(a, c.querySelector(".sw-Card__summary, .compText")?.textContent); });
  if (!out.length) document.querySelectorAll("h3 a[href], h2 a[href]").forEach(a => add(a, a.closest("div")?.parentElement?.textContent?.slice(0, 200)));
  const q = new URL(location.href).searchParams.get("q") || new URL(location.href).searchParams.get("p") || document.title;
  return { q, items: out.slice(0, 30) };
}
async function importResults(tab) { const [{ result } = {}] = await chrome.scripting.executeScript({ target: { tabId: tab.id }, func: scrapeResults }).catch(() => [{}]) || [{}]; if (!result || !result.items?.length) { chrome.notifications?.create?.({ type: "basic", iconUrl: "icon.png", title: "JEV Lab", message: "結果を読み取れませんでした" }); return; } const b64 = btoa(unescape(encodeURIComponent(JSON.stringify(result)))); openPanel(tab, { import: b64, run: "1" }); }
chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === "jev-results") return importResults(tab);
  if (info.menuItemId === "jev-judge") openPanel(tab, { text: (info.selectionText || "").slice(0, 2000), url: info.pageUrl || "", title: tab?.title || "", run: "1" });
  if (info.menuItemId === "jev-recon") { const u = info.linkUrl || info.pageUrl || ""; openPanel(tab, { target: u, run: "1" }); }
  if (info.menuItemId === "jev-collect") openPanel(tab, { q: (info.selectionText || "").slice(0, 100), run: "1" });
});
chrome.commands.onCommand.addListener(async (cmd, tab) => {
  if (cmd !== "judge-selection" || !tab?.id) return;
  const [{ result } = {}] = await chrome.scripting?.executeScript?.({ target: { tabId: tab.id }, func: () => String(getSelection()) }).catch(() => [{}]) || [{}];
  openPanel(tab, { text: (result || tab.title || "").slice(0, 2000), url: tab.url || "", run: "1" });
});
