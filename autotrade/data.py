"""kabuscout のデータ（株価25年分・決算・適時開示）から、学習用の表とバックテスト用の値段の表を作る。

kabuscout のファイルは読むだけで、書き込まない。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import features as F
from .sim import Market

log = logging.getLogger(__name__)


@dataclass
class Universe:
    min_price: float = 100.0          # 実際の株価の下限（kabuscout の universe.min_price と同じ）
    min_turnover: float = 1e8         # 20日平均売買代金の下限（kabuscout と同じ1億円）
    score_turnover: float = 3e7       # 点数を付ける（保有を続けるか判断する）下限
    min_history: int = 250            # 上場からの営業日数の下限


def load_panel(cfg: dict, start: str = "2002-01-01"):
    from kabuscout.data import PriceStore
    store = PriceStore(cfg["data"]["price_dir"])
    panel = store.load(start=start)
    proxy = cfg["universe"]["market_proxy"]
    mkt = panel["close"][proxy].ffill()
    cal = mkt.dropna().index
    codes = [c for c in panel["close"].columns if not c.startswith("^")]
    px = {k: v.reindex(index=cal, columns=codes).astype("float32") for k, v in panel.items()}
    fac = store.split_factor(cal, codes).astype("float32")
    splits = store._raw("splits").reindex(index=cal, columns=codes) if store.has_splits() else None
    div = store.load_dividends(start=start)
    div = div.reindex(index=cal, columns=codes) if div is not None and not div.empty else None
    return px, mkt.reindex(cal), fac, splits, div


def sectors(cfg: dict) -> pd.Series:
    import sqlite3
    try:
        with sqlite3.connect(str(cfg["data"]["db_path"])) as con:
            return pd.Series(dict(con.execute("SELECT code, sector33 FROM universe").fetchall()))
    except Exception:
        return pd.Series(dtype=object)


def build(cfg: dict, horizons=(5, 10, 20), uni: Universe = Universe(), start: str = "2002-01-01",
          with_events: bool = True):
    """返す: (ds 縦長の学習表, market バックテスト用, masks)。"""
    px, mkt, fac, splits, div = load_panel(cfg, start)
    o, h, l, c, v = (px[k] for k in ("open", "high", "low", "close", "volume"))
    log.info("株価: %d日 × %d銘柄（%s〜%s）", *c.shape, c.index[0].date(), c.index[-1].date())
    feats = F.price_features(o, h, l, c, v, mkt, sectors(cfg), div)
    if with_events:
        feats.update(F.event_features(cfg, c, v, mkt))
    else:
        feats.update({k: c * np.nan for k in F.EVENT_FEATURES})
    raw_c = c * fac
    hist = c.notna().rolling(uni.min_history, min_periods=1).sum()
    base = c.notna() & (v > 0) & (hist >= uni.min_history * 0.9)
    score_mask = base & (feats["turn20"] >= uni.score_turnover)
    elig = base & (feats["turn20"] >= uni.min_turnover) & (raw_c >= uni.min_price)
    ds = F.stack(feats, score_mask, F.FEATURES)
    m = score_mask.to_numpy()
    ii, jj = np.nonzero(m)
    ds["elig"] = elig.to_numpy()[ii, jj]
    ds["raw_close"] = raw_c.to_numpy("float32")[ii, jj]
    for H in horizons:
        fwd = o.shift(-1 - H) / o.shift(-1) - 1
        fwd = fwd.where(elig)
        ds[f"fwd{H}"] = fwd.to_numpy("float32")[ii, jj]
        ds[f"y{H}"] = fwd.rank(axis=1, pct=True).to_numpy("float32")[ii, jj]
    log.info("学習表: %d行（新規に買える行 %d）", len(ds), int(ds["elig"].sum()))
    ma = {}
    for n in (75, 200):
        ma[n] = (mkt < mkt.rolling(n).mean()).to_numpy()
    market = Market(cal=c.index, codes=list(c.columns), open=(o * fac).to_numpy("float64"),
                    close=raw_c.ffill(limit=5).to_numpy("float64"),
                    splits=(splits.to_numpy("float64") if splits is not None else np.ones(c.shape)),
                    div=(div * fac).to_numpy("float64") if div is not None else None, weak=ma[200], weak_ma=ma)
    return ds, market, dict(ma=ma, ii=ii, jj=jj, shape=c.shape, mkt=mkt)


def to_wide(ds: pd.DataFrame, col: str, masks: dict) -> np.ndarray:
    T, N = masks["shape"]
    out = np.full((T, N), np.nan)
    out[masks["ii"], masks["jj"]] = ds[col].to_numpy()
    return out
