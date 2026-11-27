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

SYMS = ["SPY", "SSO", "QQQ", "BIL", "AGG", "EFA", "IWM", "EEM", "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB",
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


@pytest.mark.parametrize("name", ["gem", "etf_rotation", "trend_spy", "stock_mom", "blend", "lev_trend_spy", "vol_managed_spy"])
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
    o = [trader.Order("AAA", "buy", 150, 100.0)]  # 資金の3倍（上限2.1倍）
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


def test_paper_broker_survives_split(cfg):
    """4対1の分割の後も、ペーパー口座の評価額は続いていて、実際の株数は4倍になり、余計な買いは出ない。"""
    p = make_panel()
    split = pd.Timestamp("2008-06-02")
    p.raw.loc[p.raw.index < split, "AAA"] = p.close.loc[p.close.index < split, "AAA"] * 4  # 分割前の実際の値段は4倍
    br = B.make(cfg, "paper")
    br.submit([trader.Order("AAA", "buy", 5, 400.0)], "2008-01-31")
    br.settle(p.upto(pd.Timestamp("2008-05-30")))
    before = br.mark(p.upto(pd.Timestamp("2008-05-30")))["equity"]
    assert br.positions()["AAA"] == pytest.approx(5)
    after = br.mark(p.upto(split))["equity"]
    move = p.close.at[split, "AAA"] / p.close.at[pd.Timestamp("2008-05-30"), "AAA"] - 1
    assert after == pytest.approx(before + br.s["units"]["AAA"] * p.close.at[pd.Timestamp("2008-05-30"), "AAA"] * move)
    assert br.positions()["AAA"] == pytest.approx(20)
    px = {"AAA": float(p.raw.at[split, "AAA"])}
    w = {"AAA": 20 * px["AAA"] / after}
    assert trader.plan_orders(w, br.positions(), px, after, cfg) == []


def test_replay_matches_backtest(cfg):
    """過去をたどる自動売買（毎月 rebalance --execute → ペーパー口座で約定）が、過去の検証と同じ結果になる。"""
    cfg["backtest"]["capital_usd"] = 200000
    p = D.with_cash(make_panel(end="2009-12-31"))
    mem = members_all(p)
    s0 = pd.Timestamp("2007-12-31")
    for name, tol in (("hold_spy", 0.005), ("etf_rotation", 0.03)):
        r = trader.replay(cfg, str(s0.date()), strategy=name, data=(p, mem))
        bt = backtest.run(p, trader._strategy(cfg, name, p, mem, s0), cfg, start="2008-01-01", members=mem).equity
        assert r["months"] >= 20 and not any(x["blocked"] for x in r["log"])
        assert r["final_equity"] == pytest.approx(bt.iloc[-1], rel=tol)


def test_plan_orders_scales_buys_when_cash_short(cfg):
    orders = trader.plan_orders({"AAA": 0.5, "BBB": 0.5}, {}, {"AAA": 10.0, "BBB": 10.0}, 1000.0, cfg, cash=500.0)
    assert sum(o.value for o in orders) <= 500.0 and {o.symbol for o in orders} == {"AAA", "BBB"}
    assert abs(orders[0].qty - orders[1].qty) <= 1


def test_alpaca_requests(monkeypatch):
    """Alpaca のひな形が送る中身（接続はしない。URL・鍵のヘッダ・注文の形だけ確かめる）。"""
    from usscout.broker import alpaca
    calls = []

    class R:
        def __init__(self, body):
            self.body, self.content = body, b"x"

        def raise_for_status(self):
            pass

        def json(self):
            return self.body

    def fake(method, url, headers=None, timeout=None, json=None):
        calls.append((method, url, headers, json))
        if url.endswith("/v2/account"):
            return R({"cash": "100.5", "equity": "250", "status": "ACTIVE"})
        if url.endswith("/v2/positions"):
            return R([{"symbol": "SPY", "qty": "3"}])
        return R({"id": "1", "status": "accepted"})

    monkeypatch.setattr(alpaca.requests, "request", fake)
    a = alpaca.AlpacaBroker(key="KEY", secret="SECRET")
    assert a.account() == {"cash": 100.5, "equity": 250.0, "status": "ACTIVE"}
    assert a.positions() == {"SPY": 3.0}
    a.submit([trader.Order("SPY", "buy", 2.0, 500.0), trader.Order("QQQ", "sell", 1.5, 400.0)], "2026-10-30")
    m, url, h, body = calls[2]
    assert (m, url) == ("POST", alpaca.PAPER_URL + "/v2/orders")
    assert h == {"APCA-API-KEY-ID": "KEY", "APCA-API-SECRET-KEY": "SECRET"}
    assert body == {"symbol": "SPY", "qty": "2", "side": "buy", "type": "market", "time_in_force": "opg",
                    "client_order_id": "usscout-2026-10-30-SPY-buy"}
    assert calls[3][3]["qty"] == "1" and calls[3][3]["side"] == "sell"  # opg は整数の株数だけ


def test_combo_extremes(cfg):
    """米国0%＝日本株のAIだけ（税引き後は metrics.after_tax と同じ）、米国100%（NISA）＝米国株だけ（非課税）。"""
    from usscout import combo
    idx = pd.date_range("2008-01-04", "2012-12-28", freq="W-FRI")
    rng = np.random.default_rng(1)
    jp = pd.Series(1e6 * np.cumprod(1 + rng.normal(0.002, 0.02, len(idx))), index=idx)
    us = pd.Series(1e6 * np.cumprod(1 + rng.normal(0.002, 0.02, len(idx))), index=idx)
    fx = pd.Series(100.0, index=idx)
    a = combo.combine(jp, us, 0.0, us_taxed=False, cfg=cfg, fx=fx)
    assert a.iloc[-1] == pytest.approx(metrics.after_tax(jp, cfg["tax"]["rate"], cfg["tax"]["carry_years"]).iloc[-1], rel=1e-9)
    b = combo.combine(jp, us, 1.0, us_taxed=False, cfg=cfg, fx=fx)
    assert b.iloc[-1] == pytest.approx(us.iloc[-1] / us.iloc[0] * jp.iloc[0], rel=1e-9)
    al = combo.allocation(700000, 2000, 150.0, 0.5)
    assert al["move_jpy"] == 200000 and al["direction"] == "日本株 → 米国株"


def test_vol_managed_weights(cfg):
    """落ち着いた相場ほど2倍の ETF を多く、荒れた直後は少なく。合計は1、上限は2倍（= 2倍の ETF 100%）。"""
    p = D.with_cash(make_panel())
    s = D.month_ends(p.dates)[40]
    st = strategies.build("vol_managed_spy", cfg)
    w = st(p.upto(s))
    assert sum(w.values()) == pytest.approx(1.0) and 0 <= w["SSO"] <= 1.0
    q = D.Panel(*(getattr(p, f).copy() for f in D.FIELDS))
    idx = q.close.index[(q.close.index <= s)][-15:]
    q.close.loc[idx, "SPY"] = q.close.loc[idx, "SPY"].values * np.array([1.06, 0.95] * 7 + [1.06])  # 直近だけ大荒れ
    w2 = strategies.build("vol_managed_spy", cfg)(q.upto(s))
    assert w2["SSO"] < w["SSO"]


def test_lev_trend_uses_signal_index(cfg):
    p = D.with_cash(make_panel())
    s = D.month_ends(p.dates)[40]
    w = strategies.build("lev_trend_spy", cfg)(p.upto(s))
    w1 = strategies.build("trend_spy", cfg, {"cash": "CASH"})(p.upto(s))
    assert (w == {"SSO": 1.0}) == (w1 == {"SPY": 1.0})
