"""データ取得層。

- 銘柄一覧: JPX「東証上場銘柄一覧」(無料・毎月更新)
- 株価: Yahoo Finance (yfinance, 無料・日足)。分割調整済みの OHLCV を SQLite にキャッシュし、
  毎日は直近分だけを差分取得する。株式分割を検知したらその銘柄だけ全期間を取り直す。
"""
from __future__ import annotations

import io
import logging
import re
import time
from datetime import datetime, timedelta

import pandas as pd
import requests
import yfinance as yf

log = logging.getLogger(__name__)

JPX_PAGE = "https://www.jpx.co.jp/markets/statistics-equities/misc/01.html"
FIELDS = ["open", "high", "low", "close", "volume"]


# ---------------------------------------------------------------- universe
def fetch_jpx_universe(markets: list[str]) -> pd.DataFrame:
    page = requests.get(JPX_PAGE, timeout=30)
    page.raise_for_status()
    m = re.search(r'href="([^"]+data_j\.xlsx?)"', page.text)
    if not m:
        raise RuntimeError("JPXの銘柄一覧ファイルが見つかりません")
    r = requests.get("https://www.jpx.co.jp" + m.group(1), timeout=60)
    r.raise_for_status()
    df = pd.read_excel(io.BytesIO(r.content), dtype=str)
    df = df.rename(columns={
        "コード": "code", "銘柄名": "name", "市場・商品区分": "market",
        "33業種区分": "sector33", "17業種区分": "sector17", "規模区分": "scale",
    })
    df = df[df["market"].str.contains("内国株式", na=False)]
    df = df[df["market"].apply(lambda s: any(k in s for k in markets))]
    df["code"] = df["code"].str.strip()
    return df[["code", "name", "market", "sector33", "sector17", "scale"]].reset_index(drop=True)


def refresh_universe(con, cfg: dict, force: bool = False) -> pd.DataFrame:
    cur = pd.read_sql("SELECT * FROM universe", con)
    if not cur.empty and not force:
        age = (datetime.now() - pd.to_datetime(cur["updated"]).max()).days
        if age < cfg["universe"]["refresh_days"]:
            return cur
    try:
        df = fetch_jpx_universe(cfg["universe"]["markets"])
    except Exception as e:  # ネットワーク障害時は既存の一覧で続行
        if cur.empty:
            raise
        log.warning("銘柄一覧の更新に失敗（既存を使用）: %s", e)
        return cur
    df["updated"] = datetime.now().isoformat(timespec="seconds")
    con.execute("DELETE FROM universe")
    df.to_sql("universe", con, if_exists="append", index=False)
    con.commit()
    log.info("銘柄一覧を更新: %d 銘柄", len(df))
    return df


# ---------------------------------------------------------------- prices
def _to_long(raw: pd.DataFrame, codes: list[str]) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["code", "date"] + FIELDS)
    if not isinstance(raw.columns, pd.MultiIndex):
        raw.columns = pd.MultiIndex.from_product([[f"{codes[0]}.T"], raw.columns])
    frames = []
    for tk in raw.columns.get_level_values(0).unique():
        sub = raw[tk].rename(columns=str.lower)
        if "close" not in sub:
            continue
        sub = sub[FIELDS].dropna(subset=["close"])
        if sub.empty:
            continue
        sub = sub.reset_index().rename(columns={"Date": "date", "index": "date"})
        sub["code"] = tk.replace(".T", "")
        frames.append(sub)
    if not frames:
        return pd.DataFrame(columns=["code", "date"] + FIELDS)
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    return out[["code", "date"] + FIELDS]


def _download(codes: list[str], batch: int, **kw) -> pd.DataFrame:
    frames = []
    for i in range(0, len(codes), batch):
        chunk = codes[i:i + batch]
        for attempt in range(3):
            try:
                raw = yf.download([f"{c}.T" for c in chunk], group_by="ticker",
                                  auto_adjust=False, progress=False, threads=True, **kw)
                frames.append(_to_long(raw, chunk))
                break
            except Exception as e:
                log.warning("取得失敗 (%d/%d) 再試行 %d: %s", i, len(codes), attempt + 1, e)
                time.sleep(5 * (attempt + 1))
        if (i // batch) % 5 == 0:
            log.info("  株価取得 %d / %d", min(i + batch, len(codes)), len(codes))
        time.sleep(0.5)
    return pd.concat(frames, ignore_index=True) if frames else _to_long(None, [])


def _upsert(con, df: pd.DataFrame) -> None:
    if df.empty:
        return
    con.executemany(
        "INSERT OR REPLACE INTO prices(code,date,open,high,low,close,volume) VALUES (?,?,?,?,?,?,?)",
        df[["code", "date"] + FIELDS].itertuples(index=False, name=None),
    )
    con.commit()


def update_prices(con, codes: list[str], cfg: dict) -> None:
    """差分更新。新規銘柄・分割検知銘柄は全期間取得。"""
    batch = cfg["data"]["batch_size"]
    years = cfg["data"]["history_years"]
    last = pd.read_sql("SELECT code, MAX(date) AS last FROM prices GROUP BY code", con)
    last = dict(zip(last["code"], last["last"]))

    cutoff = (datetime.now() - timedelta(days=45)).strftime("%Y-%m-%d")
    fresh = [c for c in codes if c not in last or last[c] < cutoff]
    incremental = [c for c in codes if c in last and last[c] >= cutoff]

    if incremental:
        start = (pd.to_datetime(min(last[c] for c in incremental)) - timedelta(days=10)).strftime("%Y-%m-%d")
        log.info("差分取得: %d 銘柄 (%s〜)", len(incremental), start)
        recent = _download(incremental, batch, start=start)
        # 分割・併合の検知: 既存データと重なる期間の終値が2%以上ずれたら取り直し
        old = pd.read_sql("SELECT code, date, close FROM prices WHERE date >= ?", con, params=(start,))
        merged = recent.merge(old, on=["code", "date"], suffixes=("", "_old"))
        merged["diff"] = (merged["close"] / merged["close_old"] - 1).abs()
        split_codes = sorted(set(merged.loc[merged["diff"] > 0.02, "code"]))
        if split_codes:
            log.info("分割等を検知、全期間再取得: %s", split_codes[:20])
            con.executemany("DELETE FROM prices WHERE code=?", [(c,) for c in split_codes])
            fresh += split_codes
            recent = recent[~recent["code"].isin(split_codes)]
        _upsert(con, recent)

    if fresh:
        log.info("全期間取得: %d 銘柄 (%d年)", len(fresh), years)
        _upsert(con, _download(fresh, batch, period=f"{years}y"))


def load_panel(con, start: str | None = None, codes: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """日付×銘柄のワイド形式 DataFrame の辞書 {open, high, low, close, volume} を返す。"""
    q = "SELECT * FROM prices"
    params: tuple = ()
    if start:
        q += " WHERE date >= ?"
        params = (start,)
    df = pd.read_sql(q, con, params=params)
    if codes is not None:
        df = df[df["code"].isin(set(codes))]
    df["date"] = pd.to_datetime(df["date"])
    panel = {}
    for f in FIELDS:
        panel[f] = df.pivot(index="date", columns="code", values=f).sort_index()
    return clean_panel(panel)


def clean_panel(panel: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Yahoo の記録ミス（1日だけ価格が1/10になる等）を除去する。
    東証には値幅制限があり、数日で2.5倍/0.4倍以上動くことはまずないので、
    前後1週間の中央値から2.5倍以上離れた足を欠損扱いにする。"""
    c = panel["close"]
    med = c.rolling(7, center=True, min_periods=3).median()
    ratio = (c / med).abs()
    bad = (ratio > 2.5) | (ratio < 0.4) | (c <= 0)
    n = int(bad.to_numpy().sum())
    if n:
        log.info("異常値を除去: %d 本", n)
        panel = {k: v.mask(bad) for k, v in panel.items()}
    return panel
