# JEV Lab ブラウザ拡張（Chrome / Edge、Manifest V3）

どのサイトでも「選択 → 右クリック → JEV で判断」。サイドパネルに JEV Lab（GitHub Pages）を開き、選択した文章を仕分ける。
サイトの画面はそのまま、横に JEV が付く（Grammarly 型）。

## 入れ方（配布ストア不要・無料）
1. `chrome://extensions` を開く → 右上「デベロッパーモード」ON
2. 「パッケージ化されていない拡張機能を読み込む」→ この `extension` フォルダを選ぶ
3. ツールバーの JEV アイコンでサイドパネルが開く。文章を選択して右クリック →「JEV で判断」／リンクを右クリック →「JEV で調査」／`Alt+J`

自分の GitHub Pages に置いた場合は、サイドパネル上部の URL 欄を `https://<user>.github.io/jev-lab/play.html` に変えて保存。

## 検索結果ページを取り込む（v3.0）
Google / Bing / DuckDuckGo / Yahoo! JAPAN の検索結果ページで右クリック →「この検索結果ページを JEV に取り込む」。開いているページの結果（タイトル・URL・抜粋、最大 30 件）を JEV Lab に送り、1 件ずつ信頼性を判定する。自分で開いたページを同じブラウザで読むだけなので、検索エンジンへの自動アクセスではない。Google の検索式（dorks）はこの経路で JEV に入る。
