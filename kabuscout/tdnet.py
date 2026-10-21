"""適時開示（TDnet）: 全上場企業の開示のタイトルと時刻（2010年頃〜）。

TDnet 本体は直近1ヶ月分しか公開していないため、過去分は有志の無料API（やのしん TDnet WEB-API）から取得する。
非公式サービスなので、1日分ずつ間隔を空けて取得し、手元に保存する。

タイトルから次の出来事を判定して使う:
- 決算短信（全銘柄の決算発表日。Yahoo の決算情報が無い銘柄も補える）
- 自己株式の取得（自社株買い）の決定 … 需給の改善・株主還元
- 業績予想の上方/下方修正、増配
- 株式分割、株主優待の新設
- 公募増資・新株発行（株式の希薄化）
"""
from __future__ import annotations

import json
import logging
import re
import time

import pandas as pd
import requests

log = logging.getLogger(__name__)
API = "https://webapi.yanoshin.jp/webapi/tdnet/list/{d}.json?limit=5000"
UA = {"User-Agent": "KabuScout/1.0 (personal research; polite 1 req/s)"}

EVENT_RULES = [
    ("earnings", r"決算短信"),
    ("buyback", r"自己株式(の)?取得に係る事項の決定|自己株式の取得(及び|ならびに|並びに)?.*(決定|実施)|自己株式の公開買付け"),
    ("forecast_up", r"上方修正"),
    ("forecast_down", r"下方修正"),
    ("dividend_up", r"増配"),
    ("split", r"株式分割"),
    ("yutai_new", r"株主優待.*(新設|導入)"),
    ("offering", r"公募|新株式発行|第三者割当|新株予約権.*発行"),
]
EXCLUDE = re.compile(r"訂正|取得状況|取得終了|結果|中止|延期|お詫び")


def classify(title: str) -> list[str]:
    if EXCLUDE.search(title or ""):
        return []
    return [k for k, pat in EVENT_RULES if re.search(pat, title)]


def _path(cfg):
    return cfg["data"]["db_path"].parent / "tdnet.parquet"


def load(cfg: dict) -> pd.DataFrame | None:
    p = _path(cfg)
    return pd.read_parquet(p) if p.exists() else None


def update(cfg: dict, start: str = "2010-01-01", max_minutes: float | None = None) -> pd.DataFrame:
    """未取得の日を1日ずつ取得（初回は約1時間。時間切れなら次回の続きから）。"""
    p = _path(cfg)
    old = pd.read_parquet(p) if p.exists() else pd.DataFrame(columns=["pubdate", "code", "title", "events"])
    meta_p = p.with_suffix(".json")
    raw = json.loads(meta_p.read_text()) if meta_p.exists() else []
    done = set(raw.values() if isinstance(raw, dict) else raw)
    today = pd.Timestamp.today().normalize()
    days = [d for d in pd.date_range(start, today) if d.strftime("%Y%m%d") not in done or d >= today - pd.Timedelta(days=3)]
    t0, rows = time.time(), []
    for i, d in enumerate(days):
        if max_minutes and time.time() - t0 > max_minutes * 60:
            log.info("  適時開示: 時間切れ（%d/%d日、続きは次回）", i, len(days))
            break
        key = d.strftime("%Y%m%d")
        try:
            r = requests.get(API.format(d=key), headers=UA, timeout=30)
            r.raise_for_status()
            for it in r.json().get("items", []):
                t = it["Tdnet"]
                ev = classify(t["title"])
                if ev:
                    rows.append(dict(pubdate=pd.Timestamp(t["pubdate"]), code=str(t["company_code"])[:4],
                                     title=t["title"], events=",".join(ev)))
            if d < today - pd.Timedelta(days=3):
                done.add(key)
        except Exception as e:
            log.warning("  適時開示の取得失敗 %s: %s", key, e)
            time.sleep(10)
        if i % 250 == 0:
            log.info("  適時開示 %d / %d 日", i, len(days))
        time.sleep(1.0)
    if rows:
        new = pd.DataFrame(rows)
        old = pd.concat([old, new], ignore_index=True).drop_duplicates(["pubdate", "code", "title"])
        old.to_parquet(p)
    meta_p.write_text(json.dumps(sorted(done)))
    return old
