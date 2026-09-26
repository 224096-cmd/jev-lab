"""
PC 側の公開情報コレクタ（ブラウザ版 osint.js と同じ方針・同じ出力形式）。
RSS（自治体・報道・気象庁）や CORS 非対応の公開 API も扱える。結果は JSONL → bench/run.py や大型モデルで一括判断。

  python -m jev_lab.osint.collect --query "津市 大雨" --sources jma wikipedia rss --rss https://www.pref.mie.lg.jp/rss/index.rdf --out data/osint/tsu.jsonl
  python -m jev_lab.osint.judge   --in data/osint/tsu.jsonl --models jev_ja_30m gliner2_multi nli_mdeberta --out reports/osint-tsu.json

方針: 公開・ログイン不要・規約とレート制限を守る。対象は出来事・地域・話題。個人の特定には使わない。
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

UA = "jev-lab/0.2 (research; https://github.com/224096-cmd/jev-lab)"


def get(url: str, json_=True):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json, */*"})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = r.read()
    return json.loads(data) if json_ else data


def item(source, **kw):
    return {"source": source, "fetched_at": datetime.now(timezone.utc).isoformat(), **kw}


def src_jma(q, opt):
    out = []
    code = opt.get("pref", "240000")
    w = get(f"https://www.jma.go.jp/bosai/warning/data/warning/{code}.json")
    out.append(item("jma_warning", title=f"気象警報・注意報（{w.get('publishingOffice')}）", text=w.get("headlineText") or "（発表なし）", time=w.get("reportDatetime"), url="https://www.jma.go.jp/bosai/warning/", official=True))
    for e in get("https://www.jma.go.jp/bosai/quake/data/list.json")[: opt.get("limit", 5)]:
        out.append(item("jma_quake", title=f"地震情報 {e.get('anm', '')} M{e.get('mag', '?')} 最大震度{e.get('maxi', '?')}", text=f"{e.get('at', '')} {e.get('anm', '')} 深さ{e.get('dep', '?')}km M{e.get('mag', '?')} 最大震度{e.get('maxi', '?')}", time=e.get("at"), url="https://www.jma.go.jp/bosai/map.html#contents=earthquake_map", official=True))
    return out


def src_wikipedia(q, opt):
    r = get(f"https://ja.wikipedia.org/w/api.php?action=query&list=search&srsearch={urllib.parse.quote(q)}&srlimit={opt.get('limit', 5)}&format=json")
    out = []
    for s in r["query"]["search"]:
        e = get(f"https://ja.wikipedia.org/w/api.php?action=query&prop=extracts&exintro=1&explaintext=1&titles={urllib.parse.quote(s['title'])}&format=json")
        ext = list(e["query"]["pages"].values())[0].get("extract", "")
        out.append(item("wikipedia", title=s["title"], text=ext[:600], time=s.get("timestamp"), url=f"https://ja.wikipedia.org/wiki/{urllib.parse.quote(s['title'])}"))
        time.sleep(0.5)
    return out


def src_nominatim(q, opt):
    r = get(f"https://nominatim.openstreetmap.org/search?q={urllib.parse.quote(q)}&format=json&limit={opt.get('limit', 3)}&accept-language=ja")
    time.sleep(1)
    return [item("nominatim", title=p["display_name"], text=f"lat {p['lat']}, lon {p['lon']}, {p.get('type')}", lat=float(p["lat"]), lon=float(p["lon"]), url=f"https://www.openstreetmap.org/{p['osm_type']}/{p['osm_id']}") for p in r]


def src_gdelt(q, opt):
    r = get(f"https://api.gdeltproject.org/api/v2/doc/doc?query={urllib.parse.quote(q)}&mode=artlist&maxrecords={opt.get('limit', 10)}&format=json" + ("&sourcelang=japanese" if opt.get("lang") == "ja" else ""))
    time.sleep(5)
    return [item("gdelt", title=a["title"], text=a["title"], time=a.get("seendate"), url=a["url"], domain=a.get("domain")) for a in r.get("articles", [])]


def src_bluesky(q, opt):
    r = get(f"https://public.api.bsky.app/xrpc/app.bsky.feed.searchPosts?q={urllib.parse.quote(q)}&limit={opt.get('limit', 10)}" + (f"&lang={opt['lang']}" if opt.get("lang") else ""))
    return [item("bluesky", title="@" + p["author"]["handle"], text=p.get("record", {}).get("text", ""), time=p.get("record", {}).get("createdAt"), url=f"https://bsky.app/profile/{p['author']['handle']}/post/{p['uri'].split('/')[-1]}", likes=p.get("likeCount"), reposts=p.get("repostCount")) for p in r.get("posts", [])]


def src_rss(q, opt):
    out = []
    for url in opt.get("rss", []):
        try:
            root = ET.fromstring(get(url, json_=False))
        except Exception as e:
            out.append(item("rss", error=f"{url}: {e}")); continue
        ns = {"rss": "http://purl.org/rss/1.0/", "atom": "http://www.w3.org/2005/Atom", "dc": "http://purl.org/dc/elements/1.1/"}
        entries = root.findall(".//item") + root.findall(".//rss:item", ns) + root.findall(".//atom:entry", ns)
        for e in entries[: opt.get("limit", 10)]:
            t = lambda tag: (e.findtext(tag) or e.findtext(f"rss:{tag}", namespaces=ns) or e.findtext(f"atom:{tag}", namespaces=ns) or "").strip()
            link = t("link") or (e.find("atom:link", ns).get("href") if e.find("atom:link", ns) is not None else "")
            title, desc = t("title"), t("description") or t("summary") or t("content")
            if q and q.split()[0] not in title + desc:
                continue
            out.append(item("rss", title=title, text=desc[:600], time=t("pubDate") or t("updated") or e.findtext("dc:date", namespaces=ns), url=link, feed=url))
    return out


SOURCES = {"jma": src_jma, "wikipedia": src_wikipedia, "nominatim": src_nominatim, "gdelt": src_gdelt, "bluesky": src_bluesky, "rss": src_rss}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--sources", nargs="+", default=["jma", "wikipedia", "nominatim"])
    ap.add_argument("--rss", nargs="*", default=[])
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--pref", default="240000")
    ap.add_argument("--lang", default="ja")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    opt = {"limit": a.limit, "pref": a.pref, "lang": a.lang, "rss": a.rss}
    items = []
    for s in a.sources:
        try:
            got = SOURCES[s](a.query, opt); items += got; print(f"{s}: {len(got)} 件")
        except Exception as e:
            items.append(item(s, error=str(e))); print(f"{s}: エラー {e}")
        time.sleep(1)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps({"query": a.query, **it}, ensure_ascii=False) + "\n")
    print("saved", a.out, len(items))


if __name__ == "__main__":
    main()
