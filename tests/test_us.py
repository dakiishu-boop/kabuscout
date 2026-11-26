"""米国株版（usscout）のテスト。ネットにつながらず、作り物の株価だけを使う。"""
from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from usscout import backtest, metrics, strategies, trader, universe
from usscout import broker as B
from usscout import data as D
from usscout.config import load as load_cfg

SYMS = ["SPY", "QQQ", "BIL", "AGG", "EFA", "IWM", "EEM", "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB",
        "TLT", "IEF", "GLD", "VNQ", "AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]


def make_panel(seed=0, start="2005-01-03", end="2010-12-31") -> D.Panel:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, end)
    drift = rng.normal(0.0003, 0.0004, len(SYMS))
    r = rng.normal(drift, 0.012, (len(idx), len(SYMS)))
    close = pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)), index=idx, columns=SYMS)
    opn = close.shift(1).fillna(close.iloc[0]) * np.exp(rng.normal(0, 0.003, close.shape))
    close["^IRX"] = 2.0
    opn["^IRX"] = 2.0
    return D.Panel(opn, close, close.copy(), pd.DataFrame(1e6, index=idx, columns=close.columns))


@pytest.fixture
def cfg(tmp_path):
    c = load_cfg()
    c["data"]["dir"] = tmp_path
    c["backtest"]["start"] = "2007-01-01"
    return c


def members_all(p, syms=("AAA", "BBB", "CCC", "DDD", "EEE", "FFF")):
    me = D.month_ends(p.dates)
    return pd.DataFrame(True, index=me, columns=list(syms))


@pytest.mark.parametrize("name", ["gem", "etf_rotation", "trend_spy", "stock_mom", "blend"])
def test_target_ignores_future(cfg, name):
    """判定日より後の値段を書き換えても、判定日の目標は変わらない（先読みしていない）。"""
    p = D.with_cash(make_panel())
    s = D.month_ends(p.dates)[30]
    mem = {"AAA", "BBB", "CCC", "DDD", "EEE", "FFF"}
    w1 = strategies.build(name, cfg)(p.upto(s), mem)
    q = D.Panel(*(getattr(p, f).copy() for f in D.FIELDS))
    for f in ("open", "close", "raw"):
        df = getattr(q, f)
        df.loc[df.index > s] *= 3.0
    w2 = strategies.build(name, cfg)(q.upto(s), mem)
    assert w1 == w2


def test_trades_only_day_after_signal(cfg):
    p = D.with_cash(make_panel())
    r = backtest.run(p, strategies.build("etf_rotation", cfg), cfg, members=members_all(p))
    cal = D.calendar(p)
    sig = set(D.month_ends(cal))
    for d in pd.to_datetime(r.trades["date"]).unique():
        prev = cal[cal.get_loc(d) - 1]
        assert prev in sig, f"{d} は判定日の次の営業日ではない"


def test_future_change_does_not_alter_past_equity(cfg):
    p = D.with_cash(make_panel())
    cut = pd.Timestamp("2008-06-30")
    r1 = backtest.run(p, strategies.build("stock_mom", cfg), cfg, members=members_all(p))
    q = D.Panel(*(getattr(p, f).copy() for f in D.FIELDS))
    for f in ("open", "close", "raw"):
        df = getattr(q, f)
        df.loc[df.index > cut + pd.Timedelta(days=5)] *= 0.5
    r2 = backtest.run(q, strategies.build("stock_mom", cfg), cfg, members=members_all(q))
    a, b = r1.equity.loc[:cut], r2.equity.loc[:cut]
    pd.testing.assert_series_equal(a, b)


def test_buy_and_hold_cost_once(cfg):
    p = make_panel()
    cfg["backtest"]["integer_shares"] = False
    costs = {"commission": 0.00495, "spread": 0.0005}
    r = backtest.run(p, strategies.build("hold_spy", cfg), cfg, costs=costs)
    assert len(r.trades) == 1
    t = r.trades.iloc[0]
    assert t["cost"] == pytest.approx(t["value"] * 0.00545)
    first = r.equity.index[0]
    # 寄付で買って、その日の終値で評価
    expect = (cfg["backtest"]["capital_usd"] - t["value"] - t["cost"]) + t["shares"] * p.close.at[first, "SPY"]
    assert r.equity.iloc[0] == pytest.approx(expect)


def test_commission_cap_and_free_cash():
    c = {"commission": 0.00495, "spread": 0.0, "commission_cap_usd": 22}
    assert backtest.order_cost(10000, c) == pytest.approx(22)
    assert backtest.order_cost(1000, c) == pytest.approx(4.95)
    assert backtest.order_cost(1000, c, D.CASH) == 0.0


def test_integer_shares_use_real_price(cfg):
    """1株単位は実際の値段（分割前）で判定する。実際の値段が資金より高ければ買えない。"""
    p = make_panel()
    p.raw.loc[:, "SPY"] = p.close["SPY"] * 1000  # 実際の値段 ≈ 10万ドル
    r = backtest.run(p, strategies.build("hold_spy", cfg), cfg, costs={}, integer=True)
    assert r.trades.empty or r.trades["symbol"].ne("SPY").all()


def test_split_factor():
    idx = pd.bdate_range("2020-01-01", periods=5)
    spl = pd.DataFrame({"X": [0, 0, 4.0, 0, 0]}, index=idx)
    f = D.split_factor(spl)
    assert f["X"].tolist() == [4.0, 4.0, 1.0, 1.0, 1.0]


def test_membership_rollback():
    cur = pd.DataFrame({"symbol": ["A", "B", "NEW"], "added": pd.to_datetime(["1990-01-01", "2015-06-01", "2020-03-02"])})
    ch = pd.DataFrame({"date": pd.to_datetime(["2020-03-02"]), "add": ["NEW"], "rem": ["OLD"]})
    dates = pd.to_datetime(["2010-01-29", "2016-01-29", "2020-02-28", "2020-03-31"])
    m = universe.membership(cur, ch, dates)
    assert m.loc["2020-03-31", ["A", "B", "NEW"]].all() and not m.at[pd.Timestamp("2020-03-31"), "OLD"]
    assert m.at[pd.Timestamp("2020-02-28"), "OLD"] and not m.at[pd.Timestamp("2020-02-28"), "NEW"]
    assert not m.at[pd.Timestamp("2010-01-29"), "B"]  # 採用日（2015-06）より前
    assert m.at[pd.Timestamp("2016-01-29"), "B"]


def test_random_baseline_reproducible(cfg):
    p = D.with_cash(make_panel())
    mem = members_all(p)
    a = backtest.run(p, strategies.build("stock_mom", cfg, {"score": "random"}, seed=3), cfg, members=mem).equity
    b = backtest.run(p, strategies.build("stock_mom", cfg, {"score": "random"}, seed=3), cfg, members=mem).equity
    pd.testing.assert_series_equal(a, b)


def test_after_tax_with_loss_carry():
    idx = pd.to_datetime(["2019-12-31", "2020-12-31", "2021-12-31", "2022-12-30"])
    s = pd.Series([100.0, 80.0, 110.0, 121.0], index=idx)
    t = metrics.after_tax(s, rate=0.2, carry=3)
    # 2020: −20（繰越）、2021: +30 − 20 = 10 に課税 2 → 108、2022: 108 × 1.1 = 118.8、+10.8 に課税 2.16
    assert t.iloc[-2] == pytest.approx(108.0)
    assert t.iloc[-1] == pytest.approx(118.8 - 2.16)


def test_last_trading_day():
    assert trader.last_trading_day(2024, 3) == date(2024, 3, 28)   # 3/29 は聖金曜日
    assert trader.last_trading_day(2021, 5) == date(2021, 5, 28)   # 5/31 はメモリアルデー
    assert trader.last_trading_day(2026, 11) == date(2026, 11, 30)
    assert trader.last_trading_day(2026, 10) == date(2026, 10, 30)  # 10/31 は土曜


def test_signal_date():
    idx = pd.bdate_range("2026-10-01", "2026-11-25")
    assert trader.signal_date(idx) == pd.Timestamp("2026-10-30")
    idx2 = pd.bdate_range("2026-10-01", "2026-10-30")
    assert trader.signal_date(idx2) == pd.Timestamp("2026-10-30")


def test_plan_orders_sells_first_and_skips_cash(cfg):
    w = {"AAA": 0.5, D.CASH: 0.5}
    pos = {"BBB": 10.0, "AAA": 1.0}
    px = {"AAA": 100.0, "BBB": 50.0}
    orders = trader.plan_orders(w, pos, px, 2000.0, cfg)
    assert [o.side for o in orders] == ["sell", "buy"]
    assert orders[0].symbol == "BBB" and orders[0].qty == 10
    assert orders[1].symbol == "AAA" and orders[1].qty == math.floor(0.5 * 2000 * (1 - 0.00545) / 100) - 1
    assert all(o.symbol != D.CASH for o in orders)


def test_plan_orders_band(cfg):
    orders = trader.plan_orders({"AAA": 0.5}, {"AAA": 9.0}, {"AAA": 100.0}, 2000.0, cfg)
    assert orders == []  # 目標 9株 との差が資金の5%未満


def test_checks_block(cfg):
    p = make_panel(end=str(pd.Timestamp.now().date()))
    s = p.dates[-1]
    o = [trader.Order("AAA", "buy", 100, 100.0)]
    blocked, _, _ = trader.checks(cfg, o, {"AAA": 1.0}, 5000.0, 5000.0, p, s)
    assert any("売買の合計" in b for b in blocked)
    (cfg["data"]["dir"] / "STOP").write_text("")
    blocked, _, _ = trader.checks(cfg, [], {}, 5000.0, 5000.0, p, s)
    assert any("緊急停止" in b for b in blocked)


def test_drawdown_stop_keeps_only_sells(cfg):
    p = make_panel(end=str(pd.Timestamp.now().date()))
    o = [trader.Order("AAA", "sell", 1, 100.0), trader.Order("BBB", "buy", 1, 100.0)]
    _, warn, ok = trader.checks(cfg, o, {"BBB": 0.1}, 600.0, 1000.0, p, p.dates[-1])
    assert warn and [x.side for x in ok] == ["sell"]


def test_live_requires_flag_and_env(cfg, monkeypatch):
    monkeypatch.delenv("USSCOUT_LIVE", raising=False)
    cfg["trading"]["live"] = True
    with pytest.raises(B.LiveTradingRefused):
        B.make(cfg, "paper")
    monkeypatch.setenv("USSCOUT_LIVE", "1")
    assert B.live_allowed(cfg)
    cfg["trading"]["live"] = False
    assert not B.live_allowed(cfg)


def test_alpaca_defaults_to_paper_url(monkeypatch):
    from usscout.broker.alpaca import PAPER_URL, AlpacaBroker
    a = AlpacaBroker(key="k", secret="s")
    assert a.base == PAPER_URL and not a.live


def test_paper_broker_fills_next_open(cfg):
    p = make_panel()
    br = B.make(cfg, "paper")
    s = pd.Timestamp("2008-01-31")
    br.submit([trader.Order("AAA", "buy", 5, 100.0)], str(s.date()))
    f = br.settle(p)
    nxt = p.dates[p.dates > s][0]
    assert f[0]["fill_date"] == str(nxt.date()) and f[0]["price"] == pytest.approx(p.open.at[nxt, "AAA"], rel=1e-6)
    assert br.positions() == {"AAA": 5}
    a = br.mark(p)
    assert a["equity"] == pytest.approx(br.s["cash"] + 5 * p.raw["AAA"].iloc[-1])
    br.submit([trader.Order("AAA", "sell", 5, 100.0)], str(nxt.date()))
    br.settle(p)
    assert br.positions() == {}
