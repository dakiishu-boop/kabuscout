"""テクニカル指標。全銘柄を「日付×銘柄」のワイド行列で一括計算する（数千銘柄でも数秒）。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .regimes import market_stress

# 機械学習モデルに渡す特徴量（戦略フラグ s_* は strategies.py で追加）
FEATURES = [
    "dev5", "dev25", "dev75", "dev200", "slope25", "slope75", "slope200",
    "ret1", "ret5", "ret20", "ret60", "ret120",
    "rsi14", "rsi3", "stoch_k", "macd_hist", "atr_pct",
    "bb_pos", "bbw", "bbw_rank", "vol_ratio", "vol5_20", "turnover_log", "price_log",
    "dist_high52", "dist_low52", "pullback20", "rebound5", "down_streak",
    "body", "lower_shadow", "upper_shadow", "close_pos", "gap",
    "rs20", "rs60", "sector_mom20", "rs_sector20",
    "mkt_dev75", "mkt_ret5", "mkt_ret20", "mkt_vol20", "breadth25",
]

# 候補の特徴量（シグナル表には保存するが、検証で効くと分かるまでAIには渡さない）。
# ラウンド9の調べで「値動きの大きさ・出来高」が一番効いていたので、その周辺を増やす
CANDIDATE_FEATURES = [
    "updown_vol20",  # 直近20日の出来高のうち、上がった日の出来高の割合（買いの勢い）
    "amihud20",      # 売買代金あたりの値動きの大きさ（流動性の低さ。log10）
    "vol_trend",     # 出来高の20日平均 ÷ 120日平均
    "rvol_ratio",    # 日々の値動きのばらつき 20日 ÷ 120日（荒れ始め/落ち着き）
    "max_ret20",     # 直近20日で最大の1日の上昇率（宝くじ的な銘柄は後で負けやすいという知見）
    "skew60",        # 日々の騰落率の歪み（60日）
    "beta60",        # 日経平均に対する感応度（60日。低ベータ効果）
    "rvol60",        # 日々の騰落率のばらつき（60日・年率）
]


# 戦略の判定と売買シミュレーションに必要な生の値（シグナル表に一緒に保存する）
RAW_FIELDS = [
    "open", "high", "low", "close", "sma5", "sma25", "sma75", "sma200", "high20", "high60",
    "prior_high52", "bb_lower", "atr", "turnover", "box_hi60", "box_lo60", "rsi14_min3",
    "prev_high", "sma5_prev", "sma25_prev", "stress",
]


def _rsi(close: pd.DataFrame, n: int) -> pd.DataFrame:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def _streak(cond: pd.DataFrame) -> pd.DataFrame:
    a = cond.fillna(False).to_numpy()
    out = np.zeros(a.shape, dtype=float)
    for i in range(1, len(a)):
        out[i] = np.where(a[i], out[i - 1] + 1, 0)
    return pd.DataFrame(out, index=cond.index, columns=cond.columns)


def _bcast(s: pd.Series, like: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(np.repeat(s.to_numpy()[:, None], like.shape[1], axis=1),
                        index=like.index, columns=like.columns)


def compute(panel: dict[str, pd.DataFrame], market_code: str,
            sectors: pd.Series | None = None) -> dict[str, pd.DataFrame]:
    c = panel["close"].ffill(limit=5)
    o = panel["open"].fillna(c)
    h = panel["high"].fillna(c)
    l = panel["low"].fillna(c)
    v = panel["volume"].fillna(0)
    prev = c.shift(1)

    ind: dict[str, pd.DataFrame] = {"open": o, "high": h, "low": l, "close": c, "volume": v}
    sma = {n: c.rolling(n, min_periods=n).mean() for n in (5, 25, 75, 200)}
    for n, s in sma.items():
        ind[f"sma{n}"] = s
        ind[f"dev{n}"] = c / s - 1
    ind["slope25"] = sma[25] / sma[25].shift(5) - 1
    ind["slope75"] = sma[75] / sma[75].shift(10) - 1
    ind["slope200"] = sma[200] / sma[200].shift(20) - 1
    for n in (1, 5, 20, 60, 120):
        ind[f"ret{n}"] = c / c.shift(n) - 1

    ind["rsi14"] = _rsi(c, 14)
    ind["rsi3"] = _rsi(c, 3)
    ll14, hh14 = l.rolling(14).min(), h.rolling(14).max()
    ind["stoch_k"] = (c - ll14) / (hh14 - ll14).replace(0, np.nan)
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    ind["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / c

    tr = np.maximum(h - l, np.maximum((h - prev).abs(), (l - prev).abs()))
    atr = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    ind["atr"] = atr
    ind["atr_pct"] = atr / c

    mid, sd = c.rolling(20).mean(), c.rolling(20).std()
    ind["bb_lower"] = mid - 2 * sd
    ind["bb_pos"] = (c - (mid - 2 * sd)) / (4 * sd).replace(0, np.nan)
    bbw = 4 * sd / mid
    ind["bbw"] = bbw
    bmin, bmax = bbw.rolling(120, min_periods=60).min(), bbw.rolling(120, min_periods=60).max()
    ind["bbw_rank"] = (bbw - bmin) / (bmax - bmin).replace(0, np.nan)

    vavg20 = v.rolling(20, min_periods=10).mean()
    ind["vol_ratio"] = v / vavg20.shift(1).replace(0, np.nan)
    ind["vol5_20"] = v.rolling(5).mean() / vavg20.replace(0, np.nan)
    turnover = (c * v).rolling(20, min_periods=10).mean()
    ind["turnover"] = turnover
    ind["turnover_log"] = np.log10(turnover.clip(lower=1))
    ind["price_log"] = np.log10(c.clip(lower=1))

    high52 = h.rolling(250, min_periods=120).max()
    ind["prior_high52"] = h.shift(1).rolling(250, min_periods=120).max()
    ind["dist_high52"] = c / high52 - 1
    ind["dist_low52"] = c / l.rolling(250, min_periods=120).min() - 1
    ind["high20"] = h.rolling(20).max()
    ind["high60"] = h.rolling(60).max()
    ind["pullback20"] = c / ind["high20"] - 1
    ind["rebound5"] = c / l.rolling(5).min() - 1
    ind["down_streak"] = _streak(c < prev)
    # 戦略判定用（シフト・ローリングを事前計算しておくと、行単位の表でも同じ判定式が使える）
    ind["box_hi60"] = c.shift(1).rolling(60).max()
    ind["box_lo60"] = c.shift(1).rolling(60).min()
    ind["rsi14_min3"] = ind["rsi14"].rolling(3).min()
    ind["prev_high"] = h.shift(1)
    ind["sma5_prev"] = sma[5].shift(1)
    ind["sma25_prev"] = sma[25].shift(1)

    rng = (h - l).replace(0, np.nan)
    ind["body"] = (c - o) / atr
    ind["lower_shadow"] = (np.minimum(o, c) - l) / rng
    ind["upper_shadow"] = (h - np.maximum(o, c)) / rng
    ind["close_pos"] = (c - l) / rng
    ind["gap"] = o / prev - 1

    # ---- 地合い（市場全体）
    if market_code in c.columns:
        m = c[market_code]
    else:  # 指数ETFが無ければ等加重指数で代用
        m = (1 + c.pct_change().median(axis=1).fillna(0)).cumprod()
    m_ret = m.pct_change()
    ind["rs20"] = ind["ret20"].sub(m / m.shift(20) - 1, axis=0)
    ind["rs60"] = ind["ret60"].sub(m / m.shift(60) - 1, axis=0)
    ind["mkt_dev75"] = _bcast(m / m.rolling(75).mean() - 1, c)
    ind["mkt_ret5"] = _bcast(m / m.shift(5) - 1, c)
    ind["mkt_ret20"] = _bcast(m / m.shift(20) - 1, c)
    ind["mkt_vol20"] = _bcast(m_ret.rolling(20).std() * np.sqrt(250), c)
    ind["stress"] = _bcast(market_stress(m).astype(float), c)

    # ---- 候補の特徴量（CANDIDATE_FEATURES）
    r1 = c.pct_change(fill_method=None)
    v20 = v.rolling(20, min_periods=10).sum().replace(0, np.nan)
    ind["updown_vol20"] = v.where(r1 > 0, 0.0).rolling(20, min_periods=10).sum() / v20
    ind["amihud20"] = np.log10((r1.abs() / (c * v).replace(0, np.nan)).rolling(20, min_periods=10).mean().clip(lower=1e-15) * 1e9)
    ind["vol_trend"] = vavg20 / v.rolling(120, min_periods=60).mean().replace(0, np.nan)
    sd20, sd120 = r1.rolling(20, min_periods=15).std(), r1.rolling(120, min_periods=60).std()
    ind["rvol_ratio"] = sd20 / sd120.replace(0, np.nan)
    ind["max_ret20"] = r1.rolling(20, min_periods=15).max()
    ind["skew60"] = r1.rolling(60, min_periods=40).skew()
    mr = m_ret.reindex(c.index)
    cov = r1.mul(mr, axis=0).rolling(60, min_periods=40).mean() - r1.rolling(60, min_periods=40).mean().mul(mr.rolling(60, min_periods=40).mean(), axis=0)
    ind["beta60"] = cov.div(mr.rolling(60, min_periods=40).var().replace(0, np.nan), axis=0).clip(-3, 5)
    ind["rvol60"] = r1.rolling(60, min_periods=40).std() * np.sqrt(250)
    above = (c > sma[25]).where(sma[25].notna())
    ind["breadth25"] = _bcast(above.mean(axis=1), c)

    # ---- セクター（業種の勢いと、業種内での相対強度）
    if sectors is not None:
        sec = sectors.reindex(c.columns)
        r20 = ind["ret20"]
        sec_med = r20.T.groupby(sec).median().T  # 日付×業種
        mapped = sec_med.reindex(columns=sec.values)
        mapped.columns = c.columns
        ind["sector_mom20"] = mapped
        ind["rs_sector20"] = r20 - mapped
    else:
        ind["sector_mom20"] = ind["rs_sector20"] = c * np.nan
    return ind
