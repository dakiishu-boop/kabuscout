"""過去の口座のシミュレーション（月1回の組み直し）。

流れ（本番の自動売買と同じ）:
  1. 月末（または offset 営業日前）の引けまでのデータで、戦略が目標の比率を決める
  2. 次の営業日の寄付（日本時間 23:30）に成行で売買する。売りを先に、足りなければ買いを減らす
  3. 毎日の引けで口座を評価する（配当込みの調整後の値段＝配当は同じ銘柄に再投資したとみなす）
- コスト: 約定代金 × (手数料 + 気配差)。手数料の上限は設定で（既定は上限なし）
- 1株単位: 実際の値段（分割前に戻した値段）で何株買えるかを決める
- 持ち続ける銘柄は、比率のずれが rebalance_band 未満なら売買しない
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import data as D
from .strategies import NEEDS_MEMBERS, Strategy

CASH_LIKE = {"BIL", D.CASH}  # 短期国債の ETF・米ドルMMF（投資比率に数えない）


@dataclass
class Result:
    equity: pd.Series                      # 毎日の口座の評価額（ドル）
    trades: pd.DataFrame                   # 売買の記録
    targets: dict = field(default_factory=dict)   # 判定日 → 目標の比率
    exposure: pd.Series | None = None      # 判定日ごとの株・ETF（短期国債以外）の比率
    coverage: pd.Series | None = None      # 判定日ごとの「構成銘柄のうち株価がある割合」


def order_cost(value: float, costs: dict, symbol: str | None = None) -> float:
    """1注文のコスト。costs["free"] にある銘柄（米ドルMMF の CASH）は0。"""
    if symbol is not None and symbol in costs.get("free", (D.CASH,)):
        return 0.0
    v = abs(value)
    com = v * costs.get("commission", 0.0)
    cap = costs.get("commission_cap_usd")
    if cap:
        com = min(com, float(cap))
    return com + v * costs.get("spread", 0.0)


def _members_at(members: pd.DataFrame | None, d) -> set[str] | None:
    if members is None:
        return None
    m = members.loc[:d]
    if m.empty:
        return set()
    row = m.iloc[-1]
    return set(row.index[row.values])


def run(p: D.Panel, strat: Strategy, cfg: dict, start: str | None = None, end: str | None = None,
        members: pd.DataFrame | None = None, offset: int = 0, costs: dict | None = None,
        integer: bool | None = None, capital: float | None = None) -> Result:
    bt = cfg["backtest"]
    costs = cfg["costs"] if costs is None else costs
    integer = bt.get("integer_shares", True) if integer is None else integer
    capital = float(capital or bt["capital_usd"])
    band = float(bt.get("rebalance_band", 0.0))
    start = pd.Timestamp(start or bt["start"])
    cal = D.calendar(p)
    end = pd.Timestamp(end) if end else cal[-1]
    cal = cal[cal <= end]
    sig = D.month_ends(cal, offset)
    # 最初の判定日 = 開始日より前の最後の判定日（開始日の最初の営業日に約定する）
    first = sig[sig < start]
    sig = sig[sig >= (first[-1] if len(first) else start)]
    nxt = {s: cal[cal.get_loc(s) + 1] for s in sig if cal.get_loc(s) + 1 < len(cal)}
    exec_days = {v: k for k, v in nxt.items()}

    close = p.close.reindex(cal).ffill()
    opn = p.open.reindex(cal)
    raw = p.raw.reindex(cal).ffill()
    run_days = cal[cal >= (min(exec_days) if exec_days else start)]

    sh: dict[str, float] = {}
    cash = capital
    eq, logs, targets, expo, cov = [], [], {}, {}, {}
    for d in run_days:
        if d in exec_days:
            s = exec_days[d]
            mem = _members_at(members, s) if strat.name in NEEDS_MEMBERS else None
            if mem is not None and len(mem):
                have = {m for m in mem if m in p.close.columns and pd.notna(p.close.at[s, m])}
                cov[s] = len(have) / len(mem)
                mem = have
            w = strat(p.upto(s), mem)
            targets[s] = w
            expo[s] = sum(v for k, v in w.items() if k not in CASH_LIKE)
            cash = _rebalance(d, w, sh, cash, close, opn, raw, costs, integer, band, logs)
        eq.append(cash + sum(n * close.at[d, k] for k, n in sh.items()))
    trades = pd.DataFrame(logs, columns=["date", "symbol", "shares", "price", "value", "cost"])
    return Result(pd.Series(eq, index=run_days, name=strat.name), trades, targets,
                  pd.Series(expo).sort_index(), pd.Series(cov, dtype=float).sort_index())


def _px(d, k, opn, close) -> float:
    v = opn.at[d, k] if k in opn.columns else np.nan
    return float(v) if pd.notna(v) and v > 0 else float(close.at[d, k]) if k in close.columns else np.nan


def _real_px(d, k, px, close, raw) -> float:
    """実際の値段（分割前に戻した値段）。調整後の寄付 × (実際の終値 ÷ 調整後の終値)。お金の額は調整後の株数 × 調整後の値段と同じ。"""
    r, c = raw.at[d, k] if k in raw.columns else np.nan, close.at[d, k]
    return px * float(r / c) if pd.notna(r) and pd.notna(c) and c > 0 and r > 0 else px


def _rebalance(d, w, sh, cash, close, opn, raw, costs, integer, band, logs) -> float:
    names = set(sh) | set(w)
    px = {k: _px(d, k, opn, close) for k in names}
    names = {k for k in names if pd.notna(px[k]) and px[k] > 0}
    equity = cash + sum(n * px[k] for k, n in sh.items() if k in names)
    buffer = 1 - (costs.get("commission", 0) + costs.get("spread", 0))  # 買いのコストの分だけ現金を残す
    plan = {}
    for k in names:
        cur = sh.get(k, 0.0) * px[k]
        tgt = w.get(k, 0.0) * equity * buffer
        if k in sh and k in w and abs(tgt - cur) < band * equity:
            continue
        if integer and tgt > 0:  # 1株単位: 実際の値段で何株買えるか
            rpx = _real_px(d, k, px[k], close, raw)
            tgt = math.floor(tgt / rpx) * rpx
        if abs(tgt - cur) > 1e-6:
            plan[k] = tgt - cur
    for k, dv in sorted(plan.items(), key=lambda x: x[1]):  # 売り（マイナス）を先に
        if dv < 0:
            n = dv / px[k]
            sh[k] = sh.get(k, 0.0) + n
            c = order_cost(dv, costs, k)
            cash += -dv - c
            logs.append((d, k, n, px[k], dv, c))
            if abs(sh[k] * px[k]) < 1e-6:
                sh.pop(k)
    buys = {k: dv for k, dv in plan.items() if dv > 0}
    need = sum(v + order_cost(v, costs, k) for k, v in buys.items())
    scale = min(1.0, cash / need) if need > 0 else 1.0
    for k, dv in buys.items():
        v = dv * scale
        if integer and scale < 1:
            rpx = _real_px(d, k, px[k], close, raw)
            v = math.floor(v / rpx) * rpx
        if v <= 0:
            continue
        c = order_cost(v, costs, k)
        if v + c > cash + 1e-9:
            continue
        sh[k] = sh.get(k, 0.0) + v / px[k]
        cash -= v + c
        logs.append((d, k, v / px[k], px[k], v, c))
    return cash
