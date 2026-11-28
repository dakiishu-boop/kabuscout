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
    weak: np.ndarray        # T 地合いが弱い日（日経平均が200日線の下）
    weak_ma: dict | None = None  # {移動平均の日数: T の弱い日}（Params.regime_ma で選ぶ）
    high: np.ndarray | None = None    # T×N（約定できたかの判定・取引の点検に使う）
    low: np.ndarray | None = None
    volume: np.ndarray | None = None
    turn20: np.ndarray | None = None  # T×N 20日平均売買代金（円）


# 上場企業の倒産件数（年）。生存者バイアスの検証で、保有中の銘柄が倒産する確率の目安に使う。
# 2008〜2010年は kabuscout のジャーナル §49 の値。それ以外の年は資料を確かめていないので、多めに見て一律5社とした（仮定）
BANKRUPTCIES = {2008: 33, 2009: 20, 2010: 10}
LISTED = 3600


def run(market: Market, scores: np.ndarray, eligible: np.ndarray, p: Params, capital: float = 1_000_000,
        start: str | None = None, end: str | None = None, dividends: bool = False,
        crash_mult: float = 0.0, seed: int = 0, realistic_fills: bool = False,
        lot_schedule: list | None = None, withdraw_above: float | None = None,
        index_level: np.ndarray | None = None) -> dict:
    """scores: T×N の点数（NaN = 点数なし）。eligible: T×N の新規に買ってよいか。
    crash_mult > 0: 生存者バイアスの検証。保有中の銘柄が1日あたり「その年の倒産率 × crash_mult ÷ 250」の確率で
    倒産したとみなし、その日の終値の5%で翌朝売る（実際のデータには倒産した銘柄が無いため）。
    realistic_fills: 出来高0の日・高値と安値が同じ日（ストップ高・安の張り付きなど）は寄付で売買できないとみなす
    （買いは取り消し、売りは翌日に回す）。
    lot_schedule: [(この日より前, 単元株数), ...]。例 [("2018-10-01", 1000)] = 2018年10月より前は全銘柄1,000株単位。
    値段が6営業日以上途切れた保有（上場廃止など）は、最後の終値で売ったものとして現金にする。
    withdraw_above: 毎年の最終営業日に、資産がこの額を超えていれば超えた分の現金を引き出す（利息0で別に持つ）。
    1注文の大きさを抑えた（複利で口座が巨大にならない）場合の成績を見るため。資産の推移は口座そのものの損益率をつないだもの
    （引き出した現金は含めない。含めると利息0の現金で成績が薄まる）。
    index_level: T の指数の値段（円。blend.index_levels）。p.index_share > 0 のとき、資金のその割合を指数で持つ（blend.Sleeve）。
    p.tax_ai > 0 のとき、AI の部分の利益に毎年税をかける（最後の年の分は最後の日の資産から差し引く）。"""
    from .blend import Sleeve
    from .broker import Order
    from dataclasses import replace as _replace
    rng = np.random.default_rng(seed)
    crashed: dict[str, float] = {}
    cal = market.cal
    cix = {c: j for j, c in enumerate(market.codes)}
    i0 = 0 if start is None else int(cal.searchsorted(pd.Timestamp(start)))
    i1 = len(cal) if end is None else int(cal.searchsorted(pd.Timestamp(end), side="right"))
    b = PaperBroker(capital, p.cost, p.lot)
    curve, expo = [], []
    since_reb = 10 ** 9
    weak = (market.weak_ma or {}).get(p.regime_ma, market.weak) if p.regime_ma else np.zeros(len(cal), dtype=bool)
    blocked = None
    if realistic_fills and market.volume is not None:
        blocked = (market.volume <= 0) | (market.high <= market.low)
    last_close: dict[str, float] = {}
    idx, prev_eq = 1.0, float(capital)
    sched = sorted((pd.Timestamp(x), int(n)) for x, n in (lot_schedule or []))
    sleeve = Sleeve(p.index_share, p.tax_ai) if (p.index_share > 0 or p.tax_ai > 0) else None
    if sleeve is not None and p.index_share > 0 and index_level is None:
        raise ValueError("index_share > 0 には index_level が要る")
    lvl = index_level if index_level is not None else np.ones(len(cal))
    for i in range(i0, i1):
        d = cal[i].strftime("%Y-%m-%d")
        if sleeve is not None:  # 年の最初の営業日の寄付の前: 前年の税 → 指数の比率を戻す（毎日の運用と同じ関数）
            prev = _Row(market.close[i - 1], cix) if i > i0 else None
            sleeve.new_year(cal[i].year, b, b.equity(prev) if prev is not None else b.cash(), float(lvl[i]), d)
        lot = next((n for x, n in sched if cal[i] < x), p.lot)
        pi = p if lot == p.lot else _replace(p, lot=lot)
        b.lot = lot
        sp = market.splits[i]
        if b.positions() or b.open_orders():
            ratios = {c: float(sp[cix[c]]) for c in {*b.positions(), *(o.code for o in b.open_orders())}
                      if sp[cix[c]] == sp[cix[c]] and sp[cix[c]] not in (0, 1)}
            if ratios:
                b.apply_splits(ratios)
        orow = _Row(market.open[i], cix)
        if blocked is not None and b.open_orders():
            arr = market.open[i].copy()
            arr[blocked[i]] = np.nan
            orow = _Row(arr, cix)
        if crashed:  # 倒産した銘柄は、倒産後の値段（最後の終値の5%）でしか売れない
            arr = orow.arr.copy()
            for c, v in crashed.items():
                arr[cix[c]] = v
            orow = _Row(arr, cix)
        b.execute_open(d, orow)
        crashed = {c: v for c, v in crashed.items() if c in b.positions()}
        if dividends and market.div is not None and b.positions():
            b.add_dividends({c: float(market.div[i, cix[c]]) for c in b.positions()})
        carr = market.close[i]
        if crash_mult > 0 and b.positions():
            h = BANKRUPTCIES.get(cal[i].year, 5) / LISTED * crash_mult / 250
            for c, ps in b.positions().items():
                if c not in crashed and rng.random() < h:
                    last = carr[cix[c]]
                    crashed[c] = 0.05 * (float(last) if last == last else ps.entry_px)
        if crashed:
            carr = carr.copy()
            for c, v in crashed.items():
                carr[cix[c]] = v
        crow = _Row(carr, cix)
        for c in list(b.positions()):
            v = carr[cix[c]]
            if v == v:
                last_close[c] = float(v)
            elif c in last_close:  # 6営業日以上値段が無い = 上場廃止など。最後の終値で現金にする
                b.force_sell(d, c, last_close.pop(c), "値段の途切れ（上場廃止など）")
        eq = b.mark(crow)
        ai_eq = eq
        if sleeve is not None:
            eq += sleeve.value(float(lvl[i]))
        rec = eq
        if sleeve is not None and i + 1 >= i1:  # 最後の日: まだ払っていない今年の税を差し引いて評価する
            rec = eq - sleeve.pending_tax(ai_eq, cal[i].year)
        if withdraw_above:  # 口座そのものの損益率をつなぐ（引き出した現金は成績に入れない＝時間加重の収益率）
            idx *= eq / prev_eq if prev_eq > 0 else 1.0
            rec = idx * capital
            if (i + 1 >= len(cal) or cal[i + 1].year != cal[i].year) and eq > withdraw_above:
                take = min(b.cash(), eq - withdraw_above)
                b._cash -= take
                eq -= take
            prev_eq = eq
        curve.append((cal[i], rec))
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
        orders = decide(b, sc, el, crow, reb, bool(weak[i]), pi,
                        _Row(market.turn20[i], cix) if market.turn20 is not None else None)
        if crashed:  # 倒産した銘柄は、入れ替え日でなくても翌朝売る（倒産後の値段で）
            have = {o.code for o in orders}
            orders += [Order(c, "sell", b.positions()[c].shares, v, "倒産（検証）") for c, v in crashed.items()
                       if c not in have]
        b.submit(orders)
    eqs = pd.Series(dict(curve))
    sells = [f for f in b.fills if f.side == "sell"]
    tr = pd.DataFrame([dict(date=pd.Timestamp(f.date), net=f.net, pnl=f.pnl, code=f.code, reason=f.reason,
                            entry_date=pd.Timestamp(f.entry_date) if f.entry_date else pd.NaT, shares=f.shares, px=f.raw_px)
                       for f in sells], columns=["date", "net", "pnl", "code", "reason", "entry_date", "shares", "px"])
    return dict(equity=eqs, trades=tr, exposure=pd.Series(expo, index=eqs.index), broker=b, sleeve=sleeve)


def stats(res: dict, start: str | None = None) -> dict:
    """kabuscout の backtest._stats と同じ定義の成績（年率・最大下落・シャープなど）。"""
    from kabuscout.backtest import _stats, period_stats
    if start:
        return period_stats(res, start)
    return _stats(res["equity"], res["trades"], res["exposure"])


def yearly(eq: pd.Series) -> pd.Series:
    ends = eq.groupby(eq.index.year).last()
    return ends / np.r_[eq.iloc[0], ends.to_numpy()[:-1]] - 1
