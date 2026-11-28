"""TDnet の決算短信・業績予想の修正の PDF を、読まずにそのまま手元に保存する（ジャーナル §101）。

TDnet は書類を約1ヶ月しか残さない。Claude API を止めている間（2026-11-08〜）、夜の読み取りのジョブも止まっていて、
その間の書類は二度と取れなくなっている。読む（分類する）のは後からでもできるので、PDF だけ先に貯めておく。
- 対象: tdnet_reasons（業績予想の修正・差異）と tdnet_quals（決算短信）が読む題名と同じ
- 保存先: data/tdnet_pdf/YYYY/MM/<id>.pdf（data/ は Git に入れない・公開しない）。すでにある書類は取りに行かない
- Claude API は使わない（費用 0）。TDnet への取得は1件ごとに間を空ける
- 保存のあと、まだ控えていない PDF を private のバックアップ用リポジトリのリリースに送る（pdf_backup.py、ジャーナル §103）
使い方: python -m kabuscout.tdnet_archive [直近の日数（既定 30）] [時間の上限（分、既定 90）]
"""
from __future__ import annotations

import logging
import sys
import time

import pandas as pd
import requests

from . import tdnet_quals, tdnet_reasons
from .tdnet import API, UA

log = logging.getLogger("kabuscout")
MAX_PDF = 20_000_000


def is_target(title: str) -> bool:
    return tdnet_reasons.is_target(title) or tdnet_quals.is_target(title)


def _dir(cfg: dict):
    d = cfg["data"]["db_path"].parent / "tdnet_pdf"
    d.mkdir(parents=True, exist_ok=True)
    return d


def path_of(cfg: dict, t: dict):
    pub = pd.Timestamp(t["pubdate"])
    return _dir(cfg) / f"{pub:%Y}" / f"{pub:%m}" / f"{t['id']}.pdf"


def list_targets(days: int, get=requests.get, sleep=time.sleep) -> list[dict]:
    today = pd.Timestamp.today().normalize()
    out = []
    for k in range(days):
        day = today - pd.Timedelta(days=k)
        try:
            r = get(API.format(d=day.strftime("%Y%m%d")), headers=UA, timeout=60)
            items = [it.get("Tdnet", it) for it in r.json().get("items", [])]
        except Exception as e:
            log.warning("適時開示の一覧 %s: %s", day.date(), e)
            items = []
        sleep(1.0)
        out += [t for t in items if t.get("document_url") and t.get("id") and is_target(t.get("title"))]
    return out


def update(cfg: dict, days: int = 30, max_minutes: float = 90, get=requests.get, sleep=time.sleep) -> dict:
    t0 = time.time()
    targets = [t for t in list_targets(days, get, sleep) if not path_of(cfg, t).exists()]
    targets.sort(key=lambda t: pd.Timestamp(t["pubdate"]))  # TDnet から先に消える古い日の分から保存する
    log.info("PDF の保存: 未保存 %d件（直近 %d日）", len(targets), days)
    saved = failed = 0
    size = 0
    for t in targets:
        if time.time() - t0 > max_minutes * 60:
            log.info("PDF の保存: 時間切れ（続きは次回）")
            break
        url = t["document_url"].split("rd.php?")[-1]
        try:
            resp = get(url, headers=UA, timeout=60)
            ok = resp.status_code == 200 and resp.content[:4] == b"%PDF" and len(resp.content) <= MAX_PDF
        except Exception as e:
            log.warning("PDF %s: %s", t.get("id"), e)
            ok = False
        sleep(1.0)
        if not ok:
            failed += 1
            continue
        p = path_of(cfg, t)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".part")
        tmp.write_bytes(resp.content)
        tmp.rename(p)  # 途中で止まっても壊れた PDF を「保存済み」と数えない
        saved += 1
        size += len(resp.content)
    total = sum(1 for _ in _dir(cfg).rglob("*.pdf"))
    log.info("PDF の保存: 新しく %d件（%.0f MB）・失敗 %d件・手元の合計 %d件", saved, size / 1e6, failed, total)
    return dict(targets=len(targets), saved=saved, failed=failed, mb=round(size / 1e6, 1), total=total)


if __name__ == "__main__":
    from .config import load_config
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    a = sys.argv[1:]
    c = load_config()
    print(update(c, days=int(a[0]) if a else 30, max_minutes=float(a[1]) if len(a) > 1 else 90))
    from . import pdf_backup
    try:  # 控えに失敗しても保存した PDF はそのまま。次の晩に、まだ送っていない分をまとめて送る
        print(pdf_backup.backup(c))
    except Exception as e:
        log.error("PDF の控えに失敗: %s", e)
        sys.exit(1)
