"""需給データ: JPX の銘柄別信用取引残高（日々・PDF）と空売り残高報告（日々・Excel）。

信用残は JPX のサイトに直近の数日分しか無い（2026-09-28 から全銘柄を毎営業日公表）。空売り残高報告は約1年分のバックナンバーがあるので、それも取り込む。
毎日集めて手元に貯めていき、
- 今すぐ: 候補銘柄の参考情報として表示（信用倍率・買残の上場比・大口の空売り残高）
- 約1年分たまったら: 研究ループ・AIの特徴量として効果を検証
という二段構えで使う。
"""
from __future__ import annotations

import io
import logging
import re

import numpy as np
import pandas as pd
import requests

log = logging.getLogger(__name__)
BASE = "https://www.jpx.co.jp"
MARGIN_PAGE = BASE + "/markets/statistics-equities/margin/01.html"
SHORT_PAGE = BASE + "/markets/public/short-selling/index.html"
NUM = r"([\d,]+)"
ROW = re.compile(r"(\d{4}[0-9A-Z])\s+JP[0-9A-Z]{10}\s+株数\s*Shs\.\s*" + NUM + r"\s+(▲\s*)?" + NUM + r"\s+([\d.]+)%\s+"
                 + NUM + r"\s+(▲\s*)?" + NUM + r"\s+([\d.]+)%")


def _dir(cfg):
    d = cfg["data"]["db_path"].parent / "flows"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _links(page: str, pattern: str) -> list[str]:
    html = requests.get(page, timeout=30).text
    return sorted(set(re.findall(r'href="([^"]+' + pattern + r')"', html)))


def _num(s: str) -> float:
    return float(s.replace(",", ""))


def parse_margin_pdf(content: bytes) -> pd.DataFrame:
    from pypdf import PdfReader
    text = "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(content)).pages)
    text = re.sub(r"(?<=\d) (?=\d{1,2}[,\s%])", "", text)  # PDF の改行位置で数字が割れるのを補修（例: "38,3 00"）
    rows = []
    for m in ROW.finditer(text):
        code5, sell, _, _sd, sell_pct, buy, _, _bd, buy_pct = m.groups()
        rows.append(dict(code=code5[:4], sell=_num(sell), buy=_num(buy),
                         sell_pct=float(sell_pct) / 100, buy_pct=float(buy_pct) / 100))
    return pd.DataFrame(rows).drop_duplicates("code")


def collect_margin(cfg: dict) -> int:
    path = _dir(cfg) / "margin.parquet"
    old = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    have = set(old["date"].dt.strftime("%Y%m%d")) if len(old) else set()
    added = []
    for href in _links(MARGIN_PAGE, r"_mtall\.pdf"):
        d = re.search(r"(\d{8})_mtall", href).group(1)
        if d in have:
            continue
        try:
            df = parse_margin_pdf(requests.get(BASE + href, timeout=60).content)
        except Exception as e:
            log.warning("信用残PDFの読み取り失敗 %s: %s", d, e)
            continue
        if len(df) < 500:  # 読み取りがおかしい日は捨てる
            log.warning("信用残PDFの読み取り件数が少なすぎます %s: %d", d, len(df))
            continue
        df["date"] = pd.Timestamp(d)
        added.append(df)
    if added:
        old = pd.concat([old] + added, ignore_index=True)
        old.to_parquet(path)
    return len(added)


def collect_shorts(cfg: dict) -> int:
    path = _dir(cfg) / "short_reports.parquet"
    old = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    have = set(old["file"]) if len(old) else set()
    added = []
    # 今月のページに加えて、バックナンバー（約1年分の月別ページ）も見る。初回に過去1年分が揃い、以後は新しい日だけ取る
    try:
        months = sorted(set(re.findall(r'value="(/markets/public/short-selling/00-archives-\d+\.html)"', requests.get(SHORT_PAGE, timeout=30).text)))
    except Exception:
        months = []
    hrefs = sorted({h for page in [SHORT_PAGE] + [BASE + m for m in months] for h in _links(page, r"_Short_Positions\.xlsx?")})
    for href in hrefs:
        f = href.rsplit("/", 1)[-1]
        if f in have:
            continue
        try:
            x = pd.read_excel(io.BytesIO(requests.get(BASE + href, timeout=60).content), header=None)
            x = x.iloc[8:, [1, 2, 5, 9, 10]]
            x.columns = ["calc_date", "code", "filer", "fund", "ratio"]
            x = x.dropna(subset=["code", "ratio"])
            x["code"] = x["code"].astype(str).str.strip().str[:4]
            x["calc_date"] = pd.to_datetime(x["calc_date"], errors="coerce")
            x["ratio"] = pd.to_numeric(x["ratio"], errors="coerce")
            x["filer"] = x["filer"].astype(str) + "|" + x["fund"].astype(str)
            x["file"] = f
            added.append(x.drop(columns=["fund"]).dropna(subset=["calc_date", "ratio"]))
        except Exception as e:
            log.warning("空売り残高ファイルの読み取り失敗 %s: %s", f, e)
    if added:
        old = pd.concat([old] + added, ignore_index=True)
        old.to_parquet(path)
    return len(added)


LENDING_PAGE = BASE + "/markets/statistics-equities/margin/02.html"


def collect_lending(cfg: dict) -> int:
    """品貸料（逆日歩）: 貸し出す株が足りないときに売り方が払う費用。空売りが溜まり、踏み上げが起きやすい合図。
    JPX には最新の1日分しか無いので、毎日貯める。"""
    path = _dir(cfg) / "lending.parquet"
    old = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    href = next(iter(_links(LENDING_PAGE, r"Premium_Charges\.xlsx")), None)
    if not href:
        return 0
    x = pd.read_excel(io.BytesIO(requests.get(BASE + href, timeout=60).content), header=None).iloc[3:, [0, 1, 4, 5, 6]]
    x.columns = ["date", "code", "excess", "max_rate", "rate"]
    x = x.dropna(subset=["code"])
    x["date"] = pd.to_datetime(x["date"].astype(str), format="%Y%m%d", errors="coerce")
    x["code"] = x["code"].astype(str).str.strip().str[:4]
    for c in ("excess", "max_rate", "rate"):
        x[c] = pd.to_numeric(x[c], errors="coerce")  # "*****" = 品貸料なし → 空欄
    x = x.dropna(subset=["date"])
    if len(old) and x["date"].iloc[0] in set(old["date"]):
        return 0
    pd.concat([old, x], ignore_index=True).to_parquet(path)
    return 1


def collect(cfg: dict) -> dict:
    out = {}
    for name, fn in (("margin", collect_margin), ("shorts", collect_shorts), ("lending", collect_lending)):
        try:
            out[name] = fn(cfg)
        except Exception as e:
            log.warning("需給データの収集に失敗（%s）: %s", name, e)
            out[name] = 0
    log.info("需給データを収集: 信用残 %d日分 / 空売り残高 %dファイル / 品貸料 %d日分 追加", out["margin"], out["shorts"], out["lending"])
    return out


def snapshot(cfg: dict) -> pd.DataFrame:
    """銘柄ごとの最新の需給: 信用倍率・買残/売残の上場比・その1週間前からの変化・大口空売りの合計。"""
    d = _dir(cfg)
    cols = ["buy_pct", "sell_pct", "margin_ratio", "buy_pct_chg5", "short_total", "short_filers"]
    out = pd.DataFrame(columns=cols)
    if (d / "margin.parquet").exists():
        m = pd.read_parquet(d / "margin.parquet").sort_values("date")
        last = m[m["date"] == m["date"].max()].set_index("code")
        dates = sorted(m["date"].unique())
        prev = m[m["date"] == dates[max(0, len(dates) - 6)]].set_index("code")
        out = pd.DataFrame(index=last.index)
        out["buy_pct"], out["sell_pct"] = last["buy_pct"], last["sell_pct"]
        out["margin_ratio"] = (last["buy"] / last["sell"].replace(0, np.nan)).round(2)
        out["buy_pct_chg5"] = last["buy_pct"] - prev["buy_pct"].reindex(last.index)
    if (d / "short_reports.parquet").exists():
        s = pd.read_parquet(d / "short_reports.parquet").sort_values("calc_date")
        cur = s.groupby(["code", "filer"]).tail(1)
        cur = cur[cur["ratio"] > 0]
        agg = cur.groupby("code").agg(short_total=("ratio", "sum"), short_filers=("filer", "nunique"))
        out = out.join(agg, how="outer")
    if (d / "lending.parquet").exists():
        lp = pd.read_parquet(d / "lending.parquet")
        last = lp[lp["date"] == lp["date"].max()].set_index("code")
        out = out.join(last[["rate"]].rename(columns={"rate": "lending_fee"}), how="outer")
    return out


def history_days(cfg: dict) -> dict:
    d = _dir(cfg)
    m = pd.read_parquet(d / "margin.parquet")["date"].nunique() if (d / "margin.parquet").exists() else 0
    s = pd.read_parquet(d / "short_reports.parquet")["file"].nunique() if (d / "short_reports.parquet").exists() else 0
    lend = pd.read_parquet(d / "lending.parquet")["date"].nunique() if (d / "lending.parquet").exists() else 0
    return dict(margin_days=int(m), short_files=int(s), lending_days=int(lend))
