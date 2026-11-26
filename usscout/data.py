"""株価（Yahoo Finance・無料・非公式）。日足を「日付×銘柄」の表にして data/us_scout/prices/ に保存する。

- open / close: 配当・分割を調整した値（配当を再投資した場合の値動き＝トータルリターン）。成績の計算に使う
- raw: 実際にその日に付いた終値（分割前の値段に戻したもの）。1株単位で何株買えるかの判定に使う
- volume: 出来高（分割調整済み）
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
FIELDS = ("open", "close", "raw", "volume")


@dataclass
class Panel:
    open: pd.DataFrame
    close: pd.DataFrame
    raw: pd.DataFrame
    volume: pd.DataFrame

    def upto(self, date) -> "Panel":
        """その日の引けまでに分かっている値だけ（先読みを防ぐ）。"""
        return Panel(*(getattr(self, f).loc[:date] for f in FIELDS))

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.close.index


def to_yahoo(sym: str) -> str:
    """Wikipedia などの表記（BRK.B）を Yahoo の表記（BRK-B）に。"""
    return str(sym).strip().upper().replace(".", "-")


def split_factor(splits: pd.DataFrame) -> pd.DataFrame:
    """その日より後に起きた分割の比率の積。Yahoo の終値（分割調整済み）にかけると、その日の実際の値段に戻る。"""
    r = splits.replace(0, np.nan).fillna(1.0)
    return r.iloc[::-1].cumprod().iloc[::-1].shift(-1).fillna(1.0)


def _from_yahoo(raw: pd.DataFrame, tickers: list[str]) -> dict[str, pd.DataFrame]:
    if raw is None or raw.empty:
        return {f: pd.DataFrame() for f in FIELDS}
    if not isinstance(raw.columns, pd.MultiIndex):
        raw.columns = pd.MultiIndex.from_product([[tickers[0]], raw.columns])
    raw.index = pd.to_datetime(raw.index).tz_localize(None).normalize()

    def f(name):
        try:
            return raw.xs(name, axis=1, level=1)
        except KeyError:
            return pd.DataFrame(index=raw.index)

    close, adj, op, vol = f("Close"), f("Adj Close"), f("Open"), f("Volume")
    spl = f("Stock Splits").reindex(columns=close.columns).fillna(0.0)
    factor = (adj / close).where(close > 0)
    return {"open": op * factor, "close": adj, "raw": close * split_factor(spl), "volume": vol}


def download(tickers: list[str], start: str, batch: int = 100, end: str | None = None) -> Panel:
    import yfinance as yf
    parts = {f: [] for f in FIELDS}
    tickers = list(dict.fromkeys(tickers))
    for i in range(0, len(tickers), batch):
        chunk = tickers[i:i + batch]
        for attempt in range(3):
            try:
                raw = yf.download(chunk, start=start, end=end, group_by="ticker", auto_adjust=False, actions=True,
                                  progress=False, threads=True)
                for k, v in _from_yahoo(raw, chunk).items():
                    parts[k].append(v)
                break
            except Exception as e:  # Yahoo の一時的な失敗は少し待って再試行
                log.warning("株価の取得に失敗 (%d/%d) 再試行 %d: %s", i, len(tickers), attempt + 1, e)
                time.sleep(5 * (attempt + 1))
        log.info("株価 %d / %d", min(i + batch, len(tickers)), len(tickers))
    out = {}
    for k in FIELDS:
        df = pd.concat(parts[k], axis=1) if parts[k] else pd.DataFrame()
        out[k] = df.loc[:, ~df.columns.duplicated()].sort_index()
    keep = out["close"].dropna(how="all", axis=1).columns
    idx = out["close"].dropna(how="all").index
    return align(Panel(*(out[k].reindex(index=idx, columns=keep).astype("float64") for k in FIELDS)))


MACRO = ("^IRX", "JPY=X")  # 株ではない値（金利・為替）。取引所の休みが違うので、米国株の営業日に合わせて前の値で埋める


def align(p: Panel, market: str = "SPY") -> Panel:
    """行を米国株の営業日（SPY に値段がある日）だけにする。為替・金利だけが動いた日の行があると「21営業日」の数え方がずれるため。"""
    if market not in p.close:
        return p
    full = p.close.index
    idx = p.close[market].dropna().index
    out = []
    for k in FIELDS:
        df = getattr(p, k)
        m = [c for c in MACRO if c in df.columns]
        if m:
            df = df.copy()
            df[m] = df[m].reindex(full).ffill()
        out.append(df.reindex(idx))
    return Panel(*out)


def save(p: Panel, root: Path) -> None:
    d = Path(root) / "prices"
    d.mkdir(parents=True, exist_ok=True)
    for k in FIELDS:
        getattr(p, k).to_parquet(d / f"{k}.parquet")


def load(root: Path) -> Panel:
    d = Path(root) / "prices"
    if not (d / "close.parquet").exists():
        raise FileNotFoundError(f"{d} に株価がありません。先に `python -m usscout fetch` を実行してください")
    return align(Panel(*(pd.read_parquet(d / f"{k}.parquet") for k in FIELDS)))


CASH = "CASH"  # 米ドルMMF（売買の手数料なし）。値段 = 13週国債の利回り（^IRX）− 信託報酬などで毎日増える作り物の値


def with_cash(p: Panel, fee: float = 0.005, irx: str = "^IRX") -> Panel:
    """米ドルMMF を「CASH」という作り物の銘柄として足す（ETF の BIL と違い売買の手数料がかからない）。

    楽天証券・松井証券・三菱UFJ eスマート証券は米ドルMMFの買付・換金の手数料を無料と公表（docs/JOURNAL.md §90 の出典）。
    利回りは ^IRX から fee（年率、既定 0.5% と厳しめ）を引いた値。マイナスにはしない。
    """
    cal = p.close.index
    y = (p.close[irx].reindex(cal).ffill().fillna(0) / 100 - fee).clip(lower=0) if irx in p.close else pd.Series(0.0, index=cal)
    px = (1 + y / 252).cumprod()
    px.iloc[0] = 1.0
    add = lambda df, v: df.assign(**{CASH: v})  # noqa: E731
    return Panel(add(p.open, px.shift(1).fillna(1.0)), add(p.close, px), add(p.raw, px), add(p.volume, np.nan))


def calendar(p: Panel, market: str = "SPY") -> pd.DatetimeIndex:
    """営業日 = SPY に値段がある日。"""
    return p.close[market].dropna().index


def month_ends(dates: pd.DatetimeIndex, offset: int = 0) -> pd.DatetimeIndex:
    """各月の最終営業日（offset>0 なら、その offset 営業日前）。判定日の候補。"""
    s = pd.Series(dates, index=dates)
    pos = np.flatnonzero(s.groupby(dates.to_period("M")).transform("max").values == dates.values)
    pos = pos - offset
    return dates[pos[pos >= 0]]
