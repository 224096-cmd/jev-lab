# JEV Lab ブラウザ拡張（Chrome / Edge、Manifest V3）

どのサイトでも「選択 → 右クリック → JEV で判断」。サイドパネルに JEV Lab（GitHub Pages）を開き、選択した文章を仕分ける。
サイトの画面はそのまま、横に JEV が付く（Grammarly 型）。

## 入れ方（配布ストア不要・無料）
1. `chrome://extensions` を開く → 右上「デベロッパーモード」ON
2. 「パッケージ化されていない拡張機能を読み込む」→ この `extension` フォルダを選ぶ
3. ツールバーの JEV アイコンでサイドパネルが開く。文章を選択して右クリック →「JEV で判断」／リンクを右クリック →「JEV で調査」／`Alt+J`

自分の GitHub Pages に置いた場合は、サイドパネル上部の URL 欄を `https://<user>.github.io/jev-lab/play.html` に変えて保存。
