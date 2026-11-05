"""5分足の蓄積（研究用）。場中の売買を将来検証するためのデータを、無料の Yahoo Finance から貯めていく。

Yahoo は5分足を直近約60日分しか返さない（ジャーナル §69）ので、毎日（Mac が数日止まっても60日以内なら取りこぼさない）
全銘柄の5分足を取り、月ごとの Parquet（data/intraday/5m/YYYY-MM.parquet）に足していく。
- 時刻は日本時間（タイムゾーン無し）。各足の時刻はその5分の始まり。前場の寄付直後の足が欠ける日や、
  15:25〜15:30 のクロージング・オークションの足が無いのは Yahoo 側の仕様（終値は日足のほうを使う）
- 株価は分割調整していない（その日の実際の株価）
- 同じ足をもう一度取ったら新しいほうで上書き。その日の取引が終わる前（15:40 より前）に動かしたら、その日の足は保存しない
- Yahoo の利用条件により、取ったデータは手元だけで使い、公開リポジトリには入れない（data/ は git の対象外）
使い方: python -m kabuscout.intraday（初回は60日分、以後は直近8日分） / --period 60d で取り直し
"""
from __future__ import annotations

import argparse
import logging
import sqlite3
import time
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)
FIELDS = ["open", "high", "low", "close", "volume"]
_YF = {"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}


def out_dir(cfg: dict) -> Path:
    return Path(cfg["data"]["db_path"]).parent / "intraday" / "5m"


def to_long(df: pd.DataFrame, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """yf.download(group_by="ticker") の横長の表 → 縦長（datetime, code, open, high, low, close, volume）。
    値の無い足は捨てる。now（日本時間）が 15:40 より前なら、その日の足は捨てる（取引の途中の足を残さない）。"""
    if df is None or df.empty:
        return pd.DataFrame(columns=["datetime", "code"] + FIELDS)
    if not isinstance(df.columns, pd.MultiIndex):
        raise ValueError("group_by='ticker' の表を渡してください")
    try:
        long = df.stack(level=0, future_stack=True)
    except TypeError:  # 古い pandas
        long = df.stack(level=0, dropna=False)
    long.index.names = ["datetime", "ticker"]
    long = long.rename(columns=_YF).reset_index()
    long = long.dropna(subset=["open", "high", "low", "close"])
    t = pd.to_datetime(long["datetime"])
    if t.dt.tz is not None:
        t = t.dt.tz_convert("Asia/Tokyo").dt.tz_localize(None)
    long["datetime"] = t
    long["code"] = long["ticker"].astype(str).str.replace(".T", "", regex=False)
    if now is not None and now.hour * 60 + now.minute < 15 * 60 + 40:
        long = long[long["datetime"].dt.normalize() < now.normalize()]
    out = long[["datetime", "code"] + FIELDS].copy()
    out[["open", "high", "low", "close"]] = out[["open", "high", "low", "close"]].astype("float32")
    out["volume"] = out["volume"].fillna(0).astype("int64")
    return out.reset_index(drop=True)


def merge_save(new: pd.DataFrame, folder: Path) -> dict[str, int]:
    """月ごとのファイルに足す（同じ時刻・銘柄は新しいほうで上書き）。月ごとの行数を返す。"""
    folder.mkdir(parents=True, exist_ok=True)
    counts = {}
    if new.empty:
        return counts
    for ym, g in new.groupby(new["datetime"].dt.strftime("%Y-%m")):
        p = folder / f"{ym}.parquet"
        old = pd.read_parquet(p) if p.exists() else None
        allr = g if old is None else pd.concat([old, g], ignore_index=True)
        allr = allr.drop_duplicates(["datetime", "code"], keep="last").sort_values(["datetime", "code"])
        tmp = p.with_suffix(".tmp")
        allr.to_parquet(tmp, index=False)
        tmp.replace(p)
        counts[ym] = len(allr)
    return counts


def universe_codes(cfg: dict) -> list[str]:
    with sqlite3.connect(str(cfg["data"]["db_path"])) as con:
        return [r[0] for r in con.execute("SELECT code FROM universe ORDER BY code")]


def collect(cfg: dict, period: str | None = None, chunk: int = 100, pause: float = 2.0) -> dict:
    import yfinance as yf
    folder = out_dir(cfg)
    if period is None:
        period = "8d" if any(folder.glob("*.parquet")) else "60d"
    proxy = cfg["universe"]["market_proxy"]
    tickers = [f"{c}.T" for c in universe_codes(cfg)] + [proxy, "1321.T"]
    now = pd.Timestamp.now(tz="Asia/Tokyo").tz_localize(None)
    t0, rows, failed, buf = time.time(), 0, 0, []

    def flush():
        if buf:
            merge_save(pd.concat(buf, ignore_index=True), folder)
            buf.clear()

    for n, i in enumerate(range(0, len(tickers), chunk)):
        part = tickers[i:i + chunk]
        try:
            df = yf.download(part, period=period, interval="5m", group_by="ticker", progress=False,
                             auto_adjust=False, threads=True)
        except Exception as e:  # 取得制限などは無理に回避せず、次回にまわす（60日以内なら取りこぼさない）
            log.warning("5分足の取得に失敗（%d〜）: %s", i, e)
            failed += len(part)
            if "Rate" in type(e).__name__ or "rate" in str(e).lower():
                log.warning("取得制限に当たったので今回はここまで")
                break
            continue
        lg = to_long(df, now)
        failed += len(set(part) - set(lg["code"].map(lambda c: c if c.startswith("^") else f"{c}.T")))
        buf.append(lg)
        rows += len(lg)
        if n % 10 == 9:  # 月ごとのファイルの読み書きは10回分ずつまとめて（途中で止まっても取った分は残す）
            flush()
        time.sleep(pause)
    flush()
    res = dict(period=period, tickers=len(tickers), rows=rows, missing=failed, seconds=round(time.time() - t0))
    log.info("5分足: %s", res)
    return res


def main(argv=None):
    from .config import load_config
    ap = argparse.ArgumentParser(description="全銘柄の5分足を Yahoo Finance から取って貯める（研究用）")
    ap.add_argument("--period", default=None, help="取る期間（既定: 初回 60d、以後 8d）")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    print(collect(load_config(), period=a.period))


if __name__ == "__main__":
    main()
