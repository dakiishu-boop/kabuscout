"""上場廃止のおそれがある銘柄を、その日の候補から外す（JPX の公表ページ）。

- 監理銘柄・整理銘柄（上場廃止の基準に当たるおそれ／上場廃止が決まった銘柄）
- 上場廃止が決まっている銘柄（上場廃止日が今日以降）

過去ペーパートレードの株価データには上場廃止した銘柄が無い（生存者バイアス）。倒産などで消えた銘柄を買ってしまう危険は
検証に出てこないので、本番では公表済みの危険な銘柄を最初から避ける。取得に失敗した日は前回の一覧を使う。
"""
from __future__ import annotations

import html
import json
import logging
import re

import pandas as pd
import requests

log = logging.getLogger(__name__)
PAGES = {"監理・整理銘柄": "https://www.jpx.co.jp/listing/market-alerts/supervision/index.html",
         "上場廃止予定": "https://www.jpx.co.jp/listing/stocks/delisted/index.html"}
UA = {"User-Agent": "Mozilla/5.0 (KabuScout; personal research)"}


def _rows(url: str) -> list[list[str]]:
    r = requests.get(url, headers=UA, timeout=30)
    r.raise_for_status()
    r.encoding = "utf-8"
    out = []
    for tr in re.findall(r"<tr.*?</tr>", r.text, re.S):
        cells = [html.unescape(re.sub(r"<.*?>", "", c)).strip() for c in re.findall(r"<t[dh].*?</t[dh]>", tr, re.S)]
        if len(cells) >= 3:
            out.append(cells)
    return out


def fetch(today: pd.Timestamp) -> dict[str, str]:
    """{コード: 理由}。"""
    out = {}
    for cells in _rows(PAGES["監理・整理銘柄"]):
        code = cells[2]
        if re.fullmatch(r"[0-9A-Z]{4}", code):
            out[code] = "監理・整理銘柄"
    for cells in _rows(PAGES["上場廃止予定"]):
        code, day = cells[2], pd.to_datetime(cells[0], errors="coerce")
        if re.fullmatch(r"[0-9A-Z]{4}", code) and pd.notna(day) and day >= today.normalize():
            out[code] = f"上場廃止予定（{day:%Y/%m/%d}・{cells[4] if len(cells) > 4 else ''}）"
    return out


def load(cfg: dict, today: pd.Timestamp) -> dict[str, str]:
    p = cfg["data"]["db_path"].parent / "alerts.json"
    try:
        codes = fetch(today)
        if codes:  # 0件はページの形が変わった可能性が高いので保存しない
            p.write_text(json.dumps({"date": today.strftime("%Y-%m-%d"), "codes": codes}, ensure_ascii=False))
        return codes
    except Exception as e:
        log.warning("JPX の監理・整理銘柄の取得に失敗（前回の一覧を使います）: %s", e)
    return json.loads(p.read_text())["codes"] if p.exists() else {}
