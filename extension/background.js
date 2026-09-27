// 設定（JEV Lab の URL）。自分で GitHub Pages に置いた URL に変える
const DEFAULT_APP = "https://224096-cmd.github.io/jev-lab/play.html";
const appUrl = async () => (await chrome.storage.sync.get({ app: DEFAULT_APP })).app;
chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({ id: "jev-judge", title: "JEV で判断（仕分け・信頼性）", contexts: ["selection"] });
  chrome.contextMenus.create({ id: "jev-recon", title: "JEV で調査（このリンク／ページのドメイン）", contexts: ["link", "page"] });
  chrome.contextMenus.create({ id: "jev-collect", title: "JEV で収集（選択語を話題として）", contexts: ["selection"] });
  chrome.sidePanel.setPanelBehavior({ openPanelOnActionClick: true });
});
async function openPanel(tab, params) {
  const base = await appUrl(); const url = base + "?" + new URLSearchParams(params).toString();
  await chrome.storage.session.set({ pending: url });
  try { await chrome.sidePanel.open({ tabId: tab.id }); } catch { chrome.tabs.create({ url }); }
}
chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === "jev-judge") openPanel(tab, { text: (info.selectionText || "").slice(0, 2000), url: info.pageUrl || "", title: tab?.title || "", run: "1" });
  if (info.menuItemId === "jev-recon") { const u = info.linkUrl || info.pageUrl || ""; openPanel(tab, { target: u, run: "1" }); }
  if (info.menuItemId === "jev-collect") openPanel(tab, { q: (info.selectionText || "").slice(0, 100), run: "1" });
});
chrome.commands.onCommand.addListener(async (cmd, tab) => {
  if (cmd !== "judge-selection" || !tab?.id) return;
  const [{ result } = {}] = await chrome.scripting?.executeScript?.({ target: { tabId: tab.id }, func: () => String(getSelection()) }).catch(() => [{}]) || [{}];
  openPanel(tab, { text: (result || tab.title || "").slice(0, 2000), url: tab.url || "", run: "1" });
});
