"""成績の指標（年率・最大下落・年ごと・税引き後・円建て）。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def years(s: pd.Series) -> float:
    return max((s.index[-1] - s.index[0]).days / 365.25, 1e-9)


def cagr(s: pd.Series) -> float:
    s = s.dropna()
    return float((s.iloc[-1] / s.iloc[0]) ** (1 / years(s)) - 1) if len(s) > 1 else np.nan


def max_dd(s: pd.Series) -> float:
    return float((s / s.cummax() - 1).min())


def yearly(s: pd.Series) -> pd.Series:
    """暦年ごとの騰落（最初の年は開始日から）。"""
    y = s.resample("YE").last()
    base = pd.concat([pd.Series([s.iloc[0]], index=[s.index[0] - pd.Timedelta(days=1)]), y])
    r = base.pct_change().dropna()
    r.index = r.index.year
    return r


def sharpe(s: pd.Series, rf: float = 0.0) -> float:
    r = s.pct_change().dropna()
    if r.std() == 0:
        return np.nan
    return float((r.mean() * 252 - rf) / (r.std() * np.sqrt(252)))


def after_tax(s: pd.Series, rate: float = 0.20315, carry: int = 3) -> pd.Series:
    """毎年末にその年の利益の rate を差し引く（損失は carry 年繰り越す）。ジャーナル §61 と同じ簡略計算（含み益にも課税＝厳しめ）。"""
    r = s.pct_change().fillna(0.0)
    out, losses, base = [], [], float(s.iloc[0])
    for yr, g in r.groupby(r.index.year):
        v = base * (1 + g).cumprod()
        gain = float(v.iloc[-1]) - base
        losses = [[y, l] for y, l in losses if yr - y <= carry and l > 0]
        tax = 0.0
        if gain > 0:
            left = gain
            for e in losses:  # 古い損失から使う
                take = min(e[1], left)
                e[1] -= take
                left -= take
            tax = left * rate
        else:
            losses.append([yr, -gain])
        v.iloc[-1] -= tax
        out.append(v)
        base = float(v.iloc[-1])
    return pd.concat(out)


def to_jpy(s: pd.Series, fx: pd.Series, spread: float = 0.0) -> pd.Series:
    """ドル建ての口座を円建てに（最初に円→ドル、評価のたびにドル→円。為替の手数料は1ドルあたり spread 円を片道ずつ）。"""
    f = fx.reindex(s.index).ffill().bfill()
    return s / s.iloc[0] * (f - spread) / (f.iloc[0] + spread)


def summary(s: pd.Series, bench: pd.Series | None = None, trades: pd.DataFrame | None = None) -> dict:
    y = yearly(s)
    out = {"cagr": cagr(s), "max_dd": max_dd(s), "worst_year": float(y.min()), "best_year": float(y.max()),
           "pos_years": float((y > 0).mean()), "sharpe": sharpe(s), "years": round(years(s), 1),
           "start": str(s.index[0].date()), "end": str(s.index[-1].date())}
    if bench is not None:
        b = yearly(bench.reindex(s.index).ffill())
        ex = (y - b.reindex(y.index)).dropna()
        out.update(excess_median=float(ex.median()), beat_years=float((ex > 0).mean()))
    if trades is not None and len(trades):
        # 売買の合計・コストの合計を「その年の平均の口座の額」で割り、全期間の年数で平均（売買の無い年も0として数える）
        eq = s.groupby(s.index.year).mean()
        yr = pd.to_datetime(trades["date"]).dt.year
        turn = (trades["value"].abs().groupby(yr).sum() / eq).reindex(eq.index).fillna(0).sum()
        cost = (trades["cost"].groupby(yr).sum() / eq).reindex(eq.index).fillna(0).sum()
        n = years(s)
        out.update(turnover_per_year=float(turn / n), cost_per_year=float(cost / n), n_trades=int(len(trades)))
    return out


def fmt(d: dict) -> str:
    def p(k):
        v = d.get(k)
        return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:+.1%}"
    s = f"年率 {p('cagr')}  最大下落 {p('max_dd')}  最悪の年 {p('worst_year')}  プラスの年 {d['pos_years']:.0%}"
    if "excess_median" in d:
        s += f"  SPYとの差（年ごとの中央値） {p('excess_median')}  SPYに勝った年 {d['beat_years']:.0%}"
    if "cost_per_year" in d:
        s += f"  コスト/年 {d['cost_per_year']:.2%}  売買/年 {d['turnover_per_year']:.1f}回転"
    return s
