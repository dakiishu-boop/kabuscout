"""決算短信・業績予想の修正の数字（TDnet の XBRL の「サマリー」）を毎日集める。

会社予想（売上・営業利益・経常利益・純利益・1株利益・配当）と実績が、発表と同時に機械で読める形で手に入る。
J-Quants の無料版は12週遅れなので、本番で「直近の会社予想・修正」を使うにはこちらが要る。
TDnet は書類を1ヶ月ほどで消すので過去の分は取れない（ジャーナル 51）→ 毎日、直近30日分の未取得を集めて貯めていく。
元の書類は保存せず、サマリーの数値だけを data/tdnet_xbrl/ に置く（公開しない）。
"""
from __future__ import annotations

import io
import logging
import re
import time
import zipfile

import pandas as pd
import requests

from .tdnet import API, UA

log = logging.getLogger(__name__)
TITLE = re.compile(r"決算短信|業績予想|業績修正")
FACT = re.compile(r"<ix:nonFraction\b([^>]*?)(?:/>|>(.*?)</ix:nonFraction>)", re.S)
ATTR = lambda a, k: (re.search(rf'{k}="([^"]*)"', a) or [None, None])[1]


def _dir(cfg):
    d = cfg["data"]["db_path"].parent / "tdnet_xbrl"
    d.mkdir(parents=True, exist_ok=True)
    return d


def parse(content: bytes) -> pd.DataFrame:
    """XBRL の zip から、サマリーの数値（要素名・コンテキスト・値）を取り出す。"""
    z = zipfile.ZipFile(io.BytesIO(content))
    rows = []
    for n in z.namelist():
        if "/Summary/" not in n or not n.endswith("ixbrl.htm"):
            continue
        s = z.read(n).decode("utf-8", errors="ignore")
        for m in FACT.finditer(s):
            a, v = m.group(1), re.sub(r"<.*?>", "", m.group(2) or "").strip().replace(",", "")
            if 'xsi:nil="true"' in a or not v:
                continue
            try:
                x = float(v) * 10 ** int(ATTR(a, "scale") or 0)
            except ValueError:
                continue
            rows.append(dict(element=(ATTR(a, "name") or "").split(":")[-1], context=ATTR(a, "contextRef"),
                             value=-x if ATTR(a, "sign") == "-" else x))
    return pd.DataFrame(rows)


def load(cfg: dict) -> pd.DataFrame | None:
    parts = sorted(_dir(cfg).glob("facts_*.parquet"))
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True) if parts else None


def update(cfg: dict, days: int = 30, max_minutes: float = 20) -> dict:
    """直近 days 日の一覧から、まだ取っていない決算短信・業績予想の修正の XBRL を集める。"""
    d = _dir(cfg)
    done_p = d / "done.parquet"
    done = set(pd.read_parquet(done_p)["id"]) if done_p.exists() else set()
    t0, rows, new_ids, fails = time.time(), [], [], 0
    today = pd.Timestamp.today().normalize()
    for k in range(days):
        if time.time() - t0 > max_minutes * 60:
            break
        day = today - pd.Timedelta(days=k)
        try:
            r = requests.get(API.format(d=day.strftime("%Y%m%d")), headers=UA, timeout=60)
            items = [it.get("Tdnet", it) for it in r.json().get("items", [])]
        except Exception as e:
            log.warning("適時開示の一覧 %s: %s", day.date(), e)
            continue
        time.sleep(1.0)
        for t in items:
            if time.time() - t0 > max_minutes * 60:
                break
            if not t.get("url_xbrl") or not TITLE.search(t.get("title") or "") or str(t.get("id")) in done:
                continue
            url = t["url_xbrl"].split("rd.php?")[-1]
            try:
                resp = requests.get(url, headers=UA, timeout=60)
                if resp.status_code != 200:
                    fails += 1
                    continue
                f = parse(resp.content)
            except Exception as e:
                log.warning("決算短信の XBRL %s: %s", t.get("id"), e)
                fails += 1
                continue
            finally:
                time.sleep(0.7)
            code = str(t.get("company_code") or "")[:4]
            if len(f):
                rows.append(f.assign(id=str(t["id"]), code=code, pubdate=pd.Timestamp(t["pubdate"]), title=t["title"]))
            new_ids.append(str(t["id"]))
    if rows:
        pd.concat(rows, ignore_index=True).to_parquet(d / f"facts_{pd.Timestamp.now():%Y%m%d_%H%M%S}.parquet")
    if new_ids:
        pd.DataFrame({"id": sorted(done | set(new_ids))}).to_parquet(done_p)
    info = dict(new_docs=len(new_ids), facts=int(sum(len(x) for x in rows)), failed=fails, total_docs=len(done) + len(new_ids))
    log.info("決算短信の数値: 新しく %d件（失敗 %d件）／累計 %d件", info["new_docs"], fails, info["total_docs"])
    return info
