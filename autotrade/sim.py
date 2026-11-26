"""過去での自動売買（バックテスト）。毎日の運用と同じ PaperBroker と strategy.decide を使う。

1日の流れ（kabuscout の backtest.simulate と同じ順番）:
  1) 分割・併合を保有に反映 → 2) 前日の注文を今日の寄付で約定 → 3) 終値で時価評価
  → 4) 引け後に今日の点数で判断し、翌営業日の寄付の注文を出す
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .broker import PaperBroker
from .strategy import Params, decide


class _Row:
    """その日の値段を {銘柄: 値} のように読む（全銘柄の辞書を毎日作らないため）。"""

    def __init__(self, arr: np.ndarray, cix: dict[str, int]):
        self.arr, self.cix = arr, cix

    def get(self, code, default=None):
        j = self.cix.get(code)
        if j is None:
            return default
        v = float(self.arr[j])
        return v if math.isfinite(v) else default


@dataclass
class Market:
    """バックテストに使う値段の表（実際の株価 = 分割調整後 × 倍率）。"""
    cal: pd.DatetimeIndex
    codes: list[str]
    open: np.ndarray        # T×N 実際の始値
    close: np.ndarray       # T×N 実際の終値（数日の欠けは前の値で埋めたもの）
    splits: np.ndarray      # T×N 分割の比率（その日に 1→r）。無ければ 1
    div: np.ndarray | None  # T×N 実際の1株配当（権利落ち日）
    weak: np.ndarray        # T 地合いが弱い日


def run(market: Market, scores: np.ndarray, eligible: np.ndarray, p: Params, capital: float = 1_000_000,
        start: str | None = None, end: str | None = None, dividends: bool = False) -> dict:
    """scores: T×N の点数（NaN = 点数なし）。eligible: T×N の新規に買ってよいか。"""
    cal = market.cal
    cix = {c: j for j, c in enumerate(market.codes)}
    i0 = 0 if start is None else int(cal.searchsorted(pd.Timestamp(start)))
    i1 = len(cal) if end is None else int(cal.searchsorted(pd.Timestamp(end), side="right"))
    b = PaperBroker(capital, p.cost, p.lot)
    curve, expo = [], []
    since_reb = 10 ** 9
    for i in range(i0, i1):
        d = cal[i].strftime("%Y-%m-%d")
        sp = market.splits[i]
        if b.positions() or b.open_orders():
            ratios = {c: float(sp[cix[c]]) for c in {*b.positions(), *(o.code for o in b.open_orders())}
                      if sp[cix[c]] == sp[cix[c]] and sp[cix[c]] not in (0, 1)}
            if ratios:
                b.apply_splits(ratios)
        b.execute_open(d, _Row(market.open[i], cix))
        if dividends and market.div is not None and b.positions():
            b.add_dividends({c: float(market.div[i, cix[c]]) for c in b.positions()})
        crow = _Row(market.close[i], cix)
        eq = b.mark(crow)
        curve.append((cal[i], eq))
        expo.append(1 - b.cash() / eq if eq > 0 else 0.0)
        if i + 1 >= i1:
            break
        since_reb += 1
        reb = since_reb >= p.rebalance_every
        s = scores[i]
        ok = np.isfinite(s)
        if reb and not ok.any():
            reb = False  # 点数の無い日（休場明けのデータ欠けなど）は入れ替えない
        if reb:
            since_reb = 0
            jj = np.nonzero(ok)[0]
            sc = {market.codes[j]: float(s[j]) for j in jj}
            el = {market.codes[j] for j in jj if eligible[i, j]}
        else:
            sc = {c: 0.0 for c in b.positions()}
            el = set()
        orders = decide(b, sc, el, crow, reb, bool(market.weak[i]), p)
        b.submit(orders)
    eqs = pd.Series(dict(curve))
    sells = [f for f in b.fills if f.side == "sell"]
    tr = pd.DataFrame([dict(date=pd.Timestamp(f.date), net=f.net, pnl=f.pnl, code=f.code, reason=f.reason) for f in sells],
                      columns=["date", "net", "pnl", "code", "reason"])
    return dict(equity=eqs, trades=tr, exposure=pd.Series(expo, index=eqs.index), broker=b)


def stats(res: dict, start: str | None = None) -> dict:
    """kabuscout の backtest._stats と同じ定義の成績（年率・最大下落・シャープなど）。"""
    from kabuscout.backtest import _stats, period_stats
    if start:
        return period_stats(res, start)
    return _stats(res["equity"], res["trades"], res["exposure"])


def yearly(eq: pd.Series) -> pd.Series:
    ends = eq.groupby(eq.index.year).last()
    return ends / np.r_[eq.iloc[0], ends.to_numpy()[:-1]] - 1
