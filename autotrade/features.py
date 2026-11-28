"""全銘柄の特徴量（日付×銘柄の行列から、毎日の横並びの順位を作る）。

kabuscout は「シグナルが出た銘柄だけ」を見るが、こちらは流動性の条件を満たす全銘柄を毎日並べて
「他の銘柄より上がりそうな順」を学習する。値はすべてその日の引けまでに分かる情報だけを使う（先読みなし）。

- 株価・出来高から: 騰落率（1〜250日）、変動率、移動平均からの乖離、52週高値・安値からの距離、出来高の変化、
  売買代金（規模）、流動性の低さ（Amihud）、寄付の窓と場中の値動き、歪度、市場に対するベータと固有の変動
- kabuscout のデータから: 決算発表への反応（earnings.wide_features）、適時開示（events.wide_features）、配当利回り
- 業種（東証33業種）の中での相対的な騰落

各特徴量はその日の対象銘柄の中での順位（0〜1）に直してから学習に使う（相場全体の水準の丸暗記を避ける）。
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# 学習に使う特徴量（順位に直すもの）
PRICE_FEATURES = [
    "ret1", "ret5", "ret20", "ret60", "ret120", "ret250", "mom12_1",
    "vol20", "vol60", "atr_pct", "dev5", "dev25", "dev75", "dev200",
    "hi252", "lo252", "vr5_60", "vr1_20", "turn_log", "amihud20",
    "max_ret20", "min_ret20", "gap20", "intra20", "skew60", "beta120", "ivol120",
    "rs_sector20", "sector_ret60", "div_yield",
]
EVENT_FEATURES = ["earn_age", "earn_react", "earn_surprise", "earn_vol",
                  "ev_buyback_age", "ev_forecast_up_age", "ev_forecast_down_age", "ev_dividend_up_age", "ev_offering_age"]
FEATURES = PRICE_FEATURES + EVENT_FEATURES


def _roll(df: pd.DataFrame, n: int, fn: str, minp: int | None = None) -> pd.DataFrame:
    r = df.rolling(n, min_periods=minp or max(2, int(n * 0.8)))
    return getattr(r, fn)()


def price_features(o: pd.DataFrame, h: pd.DataFrame, l: pd.DataFrame, c: pd.DataFrame, v: pd.DataFrame,
                   mkt: pd.Series, sector: pd.Series | None = None, div: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """株価（分割調整済み）から特徴量を作る。どれも日付×銘柄の表（float32）。"""
    f: dict[str, pd.DataFrame] = {}
    lc = np.log(c)
    r = lc.diff()
    for n in (1, 5, 20, 60, 120, 250):
        f[f"ret{n}"] = lc - lc.shift(n)
    f["mom12_1"] = lc.shift(20) - lc.shift(250)
    f["vol20"] = _roll(r, 20, "std")
    f["vol60"] = _roll(r, 60, "std")
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()]).groupby(level=0).max().reindex(c.index)
    f["atr_pct"] = _roll(tr, 14, "mean") / c
    for n in (5, 25, 75, 200):
        f[f"dev{n}"] = c / _roll(c, n, "mean") - 1
    f["hi252"] = c / _roll(h, 252, "max", 120) - 1
    f["lo252"] = c / _roll(l, 252, "min", 120) - 1
    v60 = _roll(v, 60, "mean")
    f["vr5_60"] = _roll(v, 5, "mean") / v60
    f["vr1_20"] = v / _roll(v, 20, "mean").shift(1)
    turn = c * v  # 調整後の株価 × 調整後の出来高 = 実際の売買代金
    t20 = _roll(turn, 20, "mean")
    f["turn20"] = t20  # 対象銘柄の判定にも使う（特徴量としては対数の順位）
    f["turn_log"] = np.log(t20.where(t20 > 0))
    f["amihud20"] = _roll(r.abs() / turn.where(turn > 0), 20, "mean")
    f["max_ret20"] = _roll(r, 20, "max")
    f["min_ret20"] = _roll(r, 20, "min")
    f["gap20"] = _roll(np.log(o / pc), 20, "mean")
    f["intra20"] = _roll(np.log(c / o), 20, "mean")
    f["skew60"] = _roll(r, 60, "skew")
    m = np.log(mkt.reindex(c.index).ffill()).diff()
    mm = m.rolling(120, min_periods=100).mean()
    mv = m.rolling(120, min_periods=100).var()
    rm = r.mul(m, axis=0)
    cov = _roll(rm, 120, "mean", 100) - _roll(r, 120, "mean", 100).mul(mm, axis=0)
    beta = cov.div(mv, axis=0)
    f["beta120"] = beta
    resid = r - beta.shift(1).mul(m, axis=0)
    f["ivol120"] = _roll(resid, 120, "std", 100)
    if sector is not None:
        sec = sector.reindex(c.columns).fillna("?")
        r20 = f["ret20"]
        med20 = r20.T.groupby(sec).transform("median").T
        f["rs_sector20"] = r20 - med20
        f["sector_ret60"] = f["ret60"].T.groupby(sec).transform("mean").T
    else:
        f["rs_sector20"] = f["ret20"] * np.nan
        f["sector_ret60"] = f["ret60"] * np.nan
    if div is not None and not div.empty:
        d = div.reindex(index=c.index, columns=c.columns).fillna(0.0)
        f["div_yield"] = d.rolling(250, min_periods=1).sum() / c
    else:
        f["div_yield"] = c * np.nan
    return {k: x.replace([np.inf, -np.inf], np.nan).astype("float32") for k, x in f.items()}


def event_features(cfg: dict, c: pd.DataFrame, v: pd.DataFrame, mkt: pd.Series) -> dict[str, pd.DataFrame]:
    """決算発表への反応と適時開示（kabuscout の計算をそのまま使う。先読みになる earn_to_next は使わない）。"""
    from kabuscout import earnings, events, tdnet
    try:
        td = tdnet.load(cfg)
        ev = earnings.combine(earnings.load(cfg), events.earnings_dates(td))
        out = earnings.wide_features(ev, c, v, mkt.reindex(c.index).ffill())
        out.update(events.wide_features(td, c))
    except Exception as e:  # データが無い環境（テスト）では空にする
        log.warning("決算・適時開示の特徴量を作れませんでした: %s", e)
        out = {}
    return {k: out.get(k, c * np.nan).astype("float32") for k in EVENT_FEATURES}


def stack(feats: dict[str, pd.DataFrame], mask: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """対象（mask が True）の (日付, 銘柄) だけを縦長にし、各特徴量をその日の順位（0〜1）に直す。"""
    m = mask.to_numpy()
    ii, jj = np.nonzero(m)
    out = {"date": mask.index.to_numpy()[ii], "code": mask.columns.to_numpy()[jj]}
    for k in names:
        a = feats[k].reindex(index=mask.index, columns=mask.columns).where(mask)
        rk = a.rank(axis=1, pct=True)
        out[k] = rk.to_numpy("float32")[ii, jj]
    return pd.DataFrame(out)
