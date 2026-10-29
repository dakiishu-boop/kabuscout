"""口座の計算（backtest.simulate と risk.order_shares）。100株単位・損失2%の上限・同時保有数・1日の上限を、
小さな作り物のデータで手計算と比べる。"""
import numpy as np
import pandas as pd
import pytest

from kabuscout import backtest
from kabuscout.risk import order_shares
from tests.conftest import FakeStore

ACC = dict(lot_size=100, risk_per_trade=0.02)


def test_order_shares_lot_cash_and_risk():
    # 1銘柄の上限 20万円・1,001円 → 199.8株 → 100株単位で100株
    assert order_shares(ACC, 1_000_000, 1_000_000, 1001.0, 50.0, 200_000) == 100
    # 損失の上限: 100万円の2% = 2万円、損切まで100円 → 200株まで（上限額なら399株分）
    assert order_shares(ACC, 1_000_000, 1_000_000, 500.5, 100.0, 200_000) == 200
    # 現金が上限額より少なければ現金まで
    assert order_shares(ACC, 1_000_000, 60_000, 500.0, 10.0, 200_000) == 100
    # 1単元も買えない値段・値段が0
    assert order_shares(ACC, 1_000_000, 1_000_000, 3003.0, 30.0, 200_000) == 0
    assert order_shares(ACC, 1_000_000, 1_000_000, 0.0, 30.0, 200_000) == 0
    # 損失の上限を使わない設定なら上限額だけで決まる
    assert order_shares(dict(lot_size=100, risk_per_trade=0), 1_000_000, 1_000_000, 100.0, 99.0, 200_000) == 2000


def _cfg(cfg, **pt):
    cfg["paper_trading"].update({**dict(capital_jpy=1_000_000, max_positions=5, picks_per_day=2, risk_per_trade=0.02, lot_size=100,
                                        consensus=0, weak_consensus=0, core={"share": 0.0}), **pt})
    cfg["paper_trading"].pop("gate", None)
    cfg["research"]["cost"] = 0.002  # 片道 0.1%
    return cfg


def _preds(rows, cal):
    df = pd.DataFrame(rows)
    df["date"] = [cal[i] for i in df.pop("day")]
    for k, v in dict(crisis=0, mkt_dev75=0.01, mkt_ret20=0.01, strategies="squeeze", sector="?", agreement=1.0).items():
        if k not in df:
            df[k] = v
    df["entry"] = df.get("entry", df["close"])
    df["atr"] = df["_stop"]
    return df


def test_simulate_matches_hand_calculation(cfg):
    cfg = _cfg(cfg)
    cal = pd.bdate_range("2025-01-06", periods=12)
    preds = _preds([
        dict(day=0, code="A", prob=0.9, close=1000.0, _stop=50.0, ret=0.10, exit_days=5),   # 100株（上限額で決まる）
        dict(day=0, code="B", prob=0.8, close=3000.0, _stop=30.0, ret=0.00, exit_days=5),   # 1単元も買えず見送り
        dict(day=0, code="C", prob=0.7, close=500.0, _stop=100.0, ret=-0.05, exit_days=3),  # 200株（損失2%の上限で決まる）
    ], cal)
    close = pd.DataFrame({"A": 1000.0, "B": 3000.0, "C": 500.0}, index=cal)
    r = backtest.simulate(preds, cal, cfg, "ai", store=FakeStore(close))
    assert r["skipped"] == 1
    eq = r["equity"]
    assert eq.iloc[0] == 1_000_000
    # 翌日の寄付で約定: 現金 100万 − 100株×1,000×1.001 − 200株×500×1.001、株は時価（終値）で評価
    cash1 = 1_000_000 - 100 * 1000 * 1.001 - 200 * 500 * 1.001
    assert eq.iloc[1] == pytest.approx(cash1 + 100 * 1000 + 200 * 500)
    # 3日目に C を −5%で、5日目に A を +10%で決済（売りも片道 0.1%）
    pc = 200 * 500 * 0.95 * 0.999
    pa = 100 * 1100 * 0.999
    assert eq.iloc[-1] == pytest.approx(cash1 + pc + pa)
    t = r["trades"].sort_values("date").reset_index(drop=True)
    assert list(t["date"]) == [cal[3], cal[5]]
    assert t["net"].tolist() == pytest.approx([pc / (200 * 500 * 1.001) - 1, pa / (100 * 1000 * 1.001) - 1])


def _many(days, cal, n=8, crisis=0):
    """毎日 n 件の候補（同じ銘柄）。安い株で、保有期間は期間より長い（決済しない）。"""
    return _preds([dict(day=d, code=f"S{k}", prob=1 - k / 100, close=100.0, _stop=5.0, ret=0.0, exit_days=50, crisis=crisis)
                   for d in days for k in range(n)], cal)


def _held_after_each_day(r):
    """毎日の投資比率から保有銘柄数を戻す（1銘柄 = 上限額20万円 → 1,900株×100円 = 19万円）。"""
    return np.round(r["exposure"].to_numpy() * r["equity"].to_numpy() / 190_000).astype(int)


def test_simulate_respects_slots_and_daily_limit(cfg):
    cfg = _cfg(cfg)
    cal = pd.bdate_range("2025-01-06", periods=8)
    close = pd.DataFrame({f"S{k}": 100.0 for k in range(8)}, index=cal)
    r = backtest.simulate(_many(range(6), cal), cal, cfg, "ai", store=FakeStore(close))
    # 1日2件まで・同時5件まで（注文は翌日に約定）。同じ銘柄は二重に買わない
    assert _held_after_each_day(r).tolist() == [0, 2, 4, 5, 5, 5, 5, 5]
    assert r["skipped"] == 0


def test_simulate_crisis_allows_one_new_buy_per_day(cfg):
    cfg = _cfg(cfg)
    cal = pd.bdate_range("2025-01-06", periods=6)
    close = pd.DataFrame({f"S{k}": 100.0 for k in range(8)}, index=cal)
    r = backtest.simulate(_many(range(5), cal, crisis=1), cal, cfg, "ai", store=FakeStore(close))
    assert _held_after_each_day(r).tolist() == [0, 1, 2, 3, 4, 5]


def test_simulate_weak_market_consensus_filter(cfg):
    cfg = _cfg(cfg, weak_consensus=0.9)
    cal = pd.bdate_range("2025-01-06", periods=4)
    preds = _preds([dict(day=0, code="A", prob=0.9, close=100.0, _stop=5.0, ret=0.0, exit_days=50, agreement=0.5, mkt_dev75=-0.02),
                    dict(day=0, code="B", prob=0.8, close=100.0, _stop=5.0, ret=0.0, exit_days=50, agreement=0.95, mkt_dev75=-0.02)], cal)
    close = pd.DataFrame({"A": 100.0, "B": 100.0}, index=cal)
    r = backtest.simulate(preds, cal, cfg, "ai", store=FakeStore(close))
    # 日経平均が75日線割れの日は、3つのモデルがそろって上位（agreement ≥ 0.9）の B だけ買う
    assert _held_after_each_day(r).tolist() == [0, 1, 1, 1]
