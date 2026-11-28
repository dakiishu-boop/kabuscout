"""autotrade（自動売買）のテスト。作り物のデータだけを使う（ネット・実データ・APIキーなし）。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from autotrade import features as F
from autotrade import model as M
from autotrade import sim
from autotrade.broker import Order, PaperBroker
from autotrade.risk import RiskGuard, RiskLimits
from autotrade.strategy import Params, decide


# ---------------------------------------------------------------- broker
def test_paper_broker_fills_at_open_with_cost_and_lots():
    b = PaperBroker(1_000_000, cost=0.002, lot=100)
    b.submit([Order("1001", "buy", 300, 1000.0)])
    assert b.positions() == {}  # 注文した日にはまだ約定しない
    b.execute_open("2024-01-05", {"1001": 1010.0})
    p = b.positions()["1001"]
    assert p.shares == 300
    assert p.entry_px == pytest.approx(1010.0 * 1.001)
    assert b.cash() == pytest.approx(1_000_000 - 300 * 1010.0 * 1.001)
    b.submit([Order("1001", "sell", 300, 1100.0)])
    b.execute_open("2024-01-06", {"1001": 1100.0})
    assert b.positions() == {}
    f = b.fills[-1]
    assert f.net == pytest.approx(1100 * 0.999 / (1010 * 1.001) - 1)
    assert b.cash() == pytest.approx(1_000_000 + 300 * (1100 * 0.999 - 1010 * 1.001))


def test_paper_broker_reduces_buy_when_cash_short_and_skips_missing_open():
    b = PaperBroker(100_000, cost=0.0)
    b.submit([Order("A", "buy", 200, 400.0), Order("B", "buy", 100, 100.0)])
    b.execute_open("d1", {"A": 600.0})  # 600×200 = 12万 > 10万 → 100株に減らす。B は寄らず取り消し
    assert b.positions()["A"].shares == 100
    assert "B" not in b.positions() and b.open_orders() == []


def test_sell_without_open_price_waits_until_next_day():
    b = PaperBroker(100_000, cost=0.0)
    b.submit([Order("A", "buy", 100, 500.0)])
    b.execute_open("d1", {"A": 500.0})
    b.submit([Order("A", "sell", 100, 500.0)])
    b.execute_open("d2", {})  # 売買停止
    assert b.positions()["A"].shares == 100 and len(b.open_orders()) == 1
    b.execute_open("d3", {"A": 520.0})
    assert "A" not in b.positions()


def test_split_adjusts_shares_and_cost_basis():
    b = PaperBroker(1_000_000, cost=0.0)
    b.submit([Order("A", "buy", 100, 3000.0)])
    b.execute_open("d1", {"A": 3000.0})
    b.apply_splits({"A": 3.0})
    p = b.positions()["A"]
    assert p.shares == 300 and p.entry_px == pytest.approx(1000.0)
    assert b.equity({"A": 1000.0}) == pytest.approx(1_000_000)


def test_broker_state_roundtrip(tmp_path):
    b = PaperBroker(500_000)
    b.submit([Order("A", "buy", 100, 1000.0)])
    b.execute_open("d1", {"A": 1000.0})
    b.submit([Order("A", "sell", 100, 1000.0)])
    b.save(tmp_path / "s.json")
    c = PaperBroker.load(tmp_path / "s.json")
    assert c.cash() == pytest.approx(b.cash())
    assert c.positions()["A"].shares == 100 and len(c.open_orders()) == 1
    json.loads((tmp_path / "s.json").read_text())


# ---------------------------------------------------------------- strategy
def _broker_with(holdings: dict, cash=600_000):
    b = PaperBroker(cash, cost=0.0)
    for code, (n, px) in holdings.items():
        b.submit([Order(code, "buy", n, px)])
    b.execute_open("d0", {c: px for c, (n, px) in holdings.items()})
    return b


def test_decide_buys_top_ranked_affordable_and_keeps_within_buffer():
    b = _broker_with({"H1": (100, 1000.0)})
    scores = {"A": 0.9, "B": 0.8, "C": 0.7, "H1": 0.75, "EXP": 0.95}
    px = {"A": 500.0, "B": 800.0, "C": 300.0, "H1": 1000.0, "EXP": 50_000.0}  # EXP は100株で500万円 → 買えない
    p = Params(n_hold=3, keep_rank=4, depth=10, cost=0.0)
    orders = decide(b, scores, set(scores) - {"H1"}, px, rebalance=True, weak=False, p=p)
    buys = [o.code for o in orders if o.side == "buy"]
    sells = [o.code for o in orders if o.side == "sell"]
    assert sells == []           # H1 は順位4位（EXP, A, B, H1）で keep_rank 以内
    assert buys == ["A", "B"]    # 空き2枠に、買える上位から
    slot = b.equity(px) / 3
    for o in orders:
        assert o.shares % 100 == 0 and o.shares * px[o.code] <= slot + 1e-6


def test_decide_sells_when_rank_drops_and_on_stop_loss():
    b = _broker_with({"H1": (100, 1000.0), "H2": (100, 1000.0)})
    scores = {"A": 0.9, "B": 0.8, "H1": 0.1, "H2": 0.85}
    px = {"A": 500.0, "B": 500.0, "H1": 1000.0, "H2": 850.0}
    p = Params(n_hold=2, keep_rank=2, depth=5, stop_loss=0.1, cost=0.0)
    orders = decide(b, scores, {"A", "B"}, px, rebalance=True, weak=False, p=p)
    reasons = {o.code: o.reason for o in orders if o.side == "sell"}
    assert reasons == {"H1": "順位低下", "H2": "損切"}
    assert [o.code for o in orders if o.side == "buy"] == ["A", "B"]


def test_decide_no_rebalance_only_stops():
    b = _broker_with({"H1": (100, 1000.0)})
    orders = decide(b, {"H1": 0.0}, set(), {"H1": 980.0}, rebalance=False, weak=False, p=Params(stop_loss=0.1))
    assert orders == []


def test_regime_reduces_holdings():
    b = _broker_with({"H1": (100, 1000.0), "H2": (100, 1000.0)})
    scores = {"H1": 0.9, "H2": 0.5}
    p = Params(n_hold=2, keep_rank=10, regime_ma=200, regime_scale=0.5, cost=0.0)
    orders = decide(b, scores, set(), {"H1": 1000.0, "H2": 1000.0}, rebalance=True, weak=True, p=p)
    assert [(o.code, o.reason) for o in orders] == [("H2", "地合い")]


# ---------------------------------------------------------------- risk guard
def test_risk_guard_blocks_bad_orders_and_kill_switch():
    g = RiskGuard(RiskLimits(max_order_value=300_000, max_orders_per_day=3, kill_drawdown=0.3, max_price_move=0.3))
    ok, why = g.check([Order("A", "buy", 100, 1000.0), Order("B", "buy", 1000, 1000.0)],
                      equity=1_000_000, peak=1_000_000, last_close={"A": 1000.0, "B": 1000.0}, data_age_days=0)
    assert [o.code for o in ok] == ["A"] and "B" in why
    ok, why = g.check([Order("A", "buy", 100, 1000.0)], equity=650_000, peak=1_000_000,
                      last_close={"A": 1000.0}, data_age_days=0)
    assert ok == [] and g.halted  # 最高値から −35% で新規の買いを止める
    ok, _ = g.check([Order("A", "sell", 100, 1000.0)], equity=650_000, peak=1_000_000,
                    last_close={"A": 1000.0}, data_age_days=0)
    assert [o.side for o in ok] == ["sell"]  # 売り（手仕舞い）は止めない
    ok, why = g.check([Order("A", "buy", 100, 1000.0)], equity=1e6, peak=1e6, last_close={"A": 1000.0}, data_age_days=5)
    assert ok == [] and "古い" in " ".join(why.values())


def test_risk_guard_price_sanity():
    g = RiskGuard(RiskLimits(max_price_move=0.3))
    ok, why = g.check([Order("A", "buy", 100, 2000.0)], equity=1e6, peak=1e6, last_close={"A": 1000.0}, data_age_days=0)
    assert ok == [] and "A" in why


# ---------------------------------------------------------------- features / model / sim (end to end)
@pytest.fixture(scope="module")
def synth():
    rng = np.random.default_rng(0)
    T, N = 900, 40
    idx = pd.bdate_range("2015-01-01", periods=T)
    codes = [f"{1000 + j}" for j in range(N)]
    drift = rng.normal(0, 0.0006, N)
    r = rng.normal(drift, 0.02, (T, N))
    # 先読みの無い「勝てる」性質を1つ入れる: 前日に大きく下げた銘柄は翌日少し戻る
    for t in range(1, T):
        r[t] += -0.15 * r[t - 1]
    c = pd.DataFrame(800 * np.exp(np.cumsum(r, 0)), index=idx, columns=codes)
    o = c.shift(1).fillna(c) * np.exp(rng.normal(0, 0.003, (T, N)))
    h = np.maximum(o, c) * 1.01
    l = np.minimum(o, c) * 0.99
    v = pd.DataFrame(rng.lognormal(12.5, 0.3, (T, N)), index=idx, columns=codes)
    mkt = pd.Series(30000 * np.exp(np.cumsum(r.mean(1))), index=idx)
    return o, h, l, c, v, mkt


def test_features_have_no_lookahead(synth):
    o, h, l, c, v, mkt = synth
    f1 = F.price_features(o, h, l, c, v, mkt)
    cut = 600
    f2 = F.price_features(o.iloc[:cut], h.iloc[:cut], l.iloc[:cut], c.iloc[:cut], v.iloc[:cut], mkt.iloc[:cut])
    for k in F.PRICE_FEATURES:
        a, b = f1[k].iloc[:cut].to_numpy(), f2[k].to_numpy()
        assert np.allclose(np.nan_to_num(a, nan=-9), np.nan_to_num(b, nan=-9), atol=1e-5), k


def test_end_to_end_walk_forward_and_sim(synth):
    o, h, l, c, v, mkt = synth
    feats = F.price_features(o, h, l, c, v, mkt)
    mask = c.notna() & (c.notna().cumsum() >= 250)
    ds = F.stack(feats, mask, F.PRICE_FEATURES)
    ii, jj = np.nonzero(mask.to_numpy())
    H = 1
    fwd = (o.shift(-1 - H) / o.shift(-1) - 1).where(mask)
    ds["fwd1"] = fwd.to_numpy()[ii, jj]
    ds["y1"] = fwd.rank(axis=1, pct=True).to_numpy()[ii, jj]
    ds["elig"] = True
    sc = M.walk_forward(ds, F.PRICE_FEATURES, "y1", H, 2017, params=dict(max_iter=40, min_samples_leaf=50), sample_every=1, min_train_days=100)
    assert sc[ds["date"] < pd.Timestamp("2017-01-01")].isna().all()
    assert sc[ds["date"] >= pd.Timestamp("2017-01-01")].notna().all()
    # 予測に意味がある（前日の下げの戻りを学んでいる）
    d = ds.assign(s=sc).dropna(subset=["s", "fwd1"])
    ic = d.groupby("date").apply(lambda x: x["s"].corr(x["fwd1"], method="spearman")).mean()
    assert ic > 0.05
    T, N = c.shape
    S = np.full((T, N), np.nan)
    S[ii, jj] = sc.to_numpy()
    market = sim.Market(cal=c.index, codes=list(c.columns), open=o.to_numpy(), close=c.to_numpy(),
                        splits=np.ones((T, N)), div=None, weak=np.zeros(T, dtype=bool))
    res = sim.run(market, S, np.isfinite(S), Params(n_hold=5, rebalance_every=1, keep_rank=8, depth=20, cost=0.0),
                  capital=1_000_000)
    eq = res["equity"]
    assert len(eq) == T and (eq > 0).all()
    assert len(res["trades"]) > 50
    # 口座の整合性: 最後の資産 = 現金 + 保有の時価
    b = res["broker"]
    last = {code: c[code].iloc[-1] for code in b.positions()}
    assert eq.iloc[-1] == pytest.approx(b.equity(last))
    # コストを入れると成績は必ず下がる
    res2 = sim.run(market, S, np.isfinite(S), Params(n_hold=5, rebalance_every=1, keep_rank=8, depth=20, cost=0.01))
    assert res2["equity"].iloc[-1] < eq.iloc[-1]


def test_build_from_price_store(cfg, synth):
    """kabuscout の株価の置き場所（PriceStore）から学習表とバックテスト用の値段の表を作れる。分割は実際の株価に戻す。"""
    from kabuscout.data import PriceStore
    from autotrade import data as D
    o, h, l, c, v, mkt = synth
    split = pd.DataFrame(np.nan, index=c.index, columns=c.columns)
    split.loc[c.index[700], "1000"] = 2.0  # 1000 は 700日目に2分割（それより前の調整後の株価は実際の半分）
    panel = {"open": o, "high": h, "low": l, "close": c, "volume": v, "splits": split, "dividends": pd.DataFrame()}
    for k in ("open", "high", "low", "close", "volume"):
        panel[k] = panel[k].assign(**{"^N225": mkt if k != "volume" else 0.0})
    panel["splits"] = split.assign(**{"^N225": np.nan})
    PriceStore(cfg["data"]["price_dir"]).save(panel)
    ds, market, masks = D.build(cfg, horizons=(5,), start="2015-01-01")
    assert {"date", "code", "elig", "y5", "fwd5", *F.FEATURES} <= set(ds.columns)
    assert ds["elig"].any()
    j = market.codes.index("1000")
    # 分割より前の日の実際の株価は、調整後の株価の2倍
    assert market.close[100, j] == pytest.approx(float(c["1000"].iloc[100]) * 2, rel=1e-5)
    assert market.close[800, j] == pytest.approx(float(c["1000"].iloc[800]), rel=1e-5)
    assert market.splits[700, j] == 2.0


def test_live_daily_places_orders_then_fills_next_day(cfg, synth, tmp_path):
    """毎日の運用: 1日目に注文 → データが1日進んだ2日目に寄付で約定。状態は保存して引き継ぐ。"""
    from kabuscout.data import PriceStore
    from autotrade import live
    o, h, l, c, v, mkt = synth
    def save(n):
        P = {k: x.iloc[:n].assign(**{"^N225": (mkt.iloc[:n] if k != "volume" else 0.0)})
             for k, x in dict(open=o, high=h, low=l, close=c, volume=v).items()}
        P["splits"] = pd.DataFrame(np.nan, index=c.index[:n], columns=list(c.columns) + ["^N225"])
        P["dividends"] = pd.DataFrame()
        PriceStore(cfg["data"]["price_dir"]).save(P)
    s = live.load_settings()
    s["model"].update(params=dict(max_iter=10, min_samples_leaf=100), sample_every=5)
    s["strategy"].update(n_hold=4, keep_rank=8)
    s["risk"]["max_order_value"] = 1e9
    save(850)
    r1 = live.daily(cfg, s, notify=False, today=c.index[849])
    assert r1["positions"] == [] and len(r1["orders"]) >= 1
    assert all(x["side"] == "buy" and x["shares"] % 100 == 0 for x in r1["orders"])
    save(851)
    r2 = live.daily(cfg, s, notify=False, today=c.index[850])
    bought = {x["code"] for x in r2["positions"]}
    assert bought == {x["code"] for x in r1["orders"]}
    st = json.loads((live.state_dir(cfg) / "state.json").read_text())
    assert st["last_date"] == str(c.index[850].date()) and len(st["equity"]) == 2
    # 同じデータでもう一度動かしても、二重に約定・注文しない
    r3 = live.daily(cfg, s, notify=False, today=c.index[850])
    assert r3["positions"] == r2["positions"] and r3["orders"] == []


def test_crash_injection_lowers_returns(synth):
    """生存者バイアスの検証（倒産の混入）: 倒産の確率を上げるほど成績は下がり、倒産した銘柄は −95% 前後で売られる。"""
    o, h, l, c, v, mkt = synth
    T, N = c.shape
    rng = np.random.default_rng(3)
    S = rng.random((T, N))
    market = sim.Market(cal=c.index, codes=list(c.columns), open=o.to_numpy(), close=c.to_numpy(),
                        splits=np.ones((T, N)), div=None, weak=np.zeros(T, dtype=bool))
    p = Params(n_hold=5, rebalance_every=5, keep_rank=10, depth=20)
    base = sim.run(market, S, np.ones((T, N), bool), p)
    hit = sim.run(market, S, np.ones((T, N), bool), p, crash_mult=3000, seed=1)
    assert hit["equity"].iloc[-1] < base["equity"].iloc[-1]
    tr = hit["trades"]
    crash = tr[tr["reason"] == "倒産（検証）"]
    assert len(crash) > 0 and (crash["net"] < -0.9).all()


def test_live_skips_and_sells_delisting_risk(cfg, synth):
    """監理・整理銘柄（alerts.json）は買わず、持っていれば翌朝売る。"""
    from kabuscout.data import PriceStore
    from autotrade import live
    o, h, l, c, v, mkt = synth
    def save(n):
        P = {k: x.iloc[:n].assign(**{"^N225": (mkt.iloc[:n] if k != "volume" else 0.0)})
             for k, x in dict(open=o, high=h, low=l, close=c, volume=v).items()}
        P["splits"] = pd.DataFrame(np.nan, index=c.index[:n], columns=list(c.columns) + ["^N225"])
        P["dividends"] = pd.DataFrame()
        PriceStore(cfg["data"]["price_dir"]).save(P)
    s = live.load_settings()
    s["model"].update(params=dict(max_iter=10, min_samples_leaf=100), sample_every=5)
    s["strategy"].update(n_hold=4, keep_rank=40, regime_ma=0)
    s["risk"]["max_order_value"] = 1e9
    save(850)
    r1 = live.daily(cfg, s, notify=False, today=c.index[849])
    first = r1["orders"][0]["code"]
    save(851)
    live.daily(cfg, s, notify=False, today=c.index[850])
    alerts = Path(cfg["data"]["db_path"]).parent / "alerts.json"
    alerts.write_text(json.dumps({"date": "x", "codes": {first: "監理・整理銘柄", "1039": "上場廃止予定"}}))
    st = live.state_dir(cfg) / "state.json"
    d = json.loads(st.read_text()); d["since_reb"] = 99; st.write_text(json.dumps(d))  # 入れ替え日にする
    save(852)
    r3 = live.daily(cfg, s, notify=False, today=c.index[851])
    sells = {x["code"]: x["reason"] for x in r3["orders"] if x["side"] == "sell"}
    assert sells.get(first) == "監理・整理銘柄"
    assert all(x["code"] not in (first, "1039") for x in r3["orders"] if x["side"] == "buy")


def _mk_market(c, o, **kw):
    T, N = c.shape
    return sim.Market(cal=c.index, codes=list(c.columns), open=o.to_numpy(), close=c.to_numpy(),
                      splits=np.ones((T, N)), div=None, weak=np.zeros(T, dtype=bool), **kw)


def test_locked_days_are_not_tradable_and_lot_schedule(synth):
    o, h, l, c, v, mkt = synth
    T, N = c.shape
    S = np.random.default_rng(5).random((T, N))
    p = Params(n_hold=5, rebalance_every=5, keep_rank=10, depth=20)
    hi, lo, vol = h.to_numpy(), l.to_numpy(), v.to_numpy().copy()
    vol[:, :20] = 0  # 半分の銘柄は売買が一度も成立しない
    m = _mk_market(c, o, high=hi, low=lo, volume=vol)
    res = sim.run(m, S, np.ones((T, N), bool), p, realistic_fills=True)
    bought = {f.code for f in res["broker"].fills if f.side == "buy"}
    assert bought and not (bought & set(c.columns[:20]))
    big = sim.run(m, S, np.ones((T, N), bool), p, lot_schedule=[("2099-01-01", 1000)])
    assert all(f.shares % 1000 == 0 for f in big["broker"].fills if f.side == "buy")


def test_position_with_vanished_prices_is_cashed_out(synth):
    o, h, l, c, v, mkt = synth
    c2, o2 = c.copy(), o.copy()
    c2.iloc[400:, :] = np.nan  # 全銘柄が400日目で上場廃止
    o2.iloc[400:, :] = np.nan
    T, N = c.shape
    S = np.random.default_rng(1).random((T, N))
    m = _mk_market(c2.ffill(limit=5), o2)
    res = sim.run(m, S, np.ones((T, N), bool), Params(n_hold=5, rebalance_every=5, keep_rank=10, depth=20))
    b = res["broker"]
    assert b.positions() == {}
    assert (res["trades"]["reason"] == "値段の途切れ（上場廃止など）").sum() >= 1
    assert res["trades"]["entry_date"].notna().all()


def test_liquidity_cap_and_withdrawals(synth):
    o, h, l, c, v, mkt = synth
    T, N = c.shape
    S = np.random.default_rng(2).random((T, N))
    turn = np.full((T, N), 2_000_000.0)  # どの銘柄も1日の売買代金200万円
    m = _mk_market(c, o, turn20=turn)
    p = Params(n_hold=5, rebalance_every=5, keep_rank=10, depth=20, max_adv_frac=0.05)  # 1注文10万円まで
    res = sim.run(m, S, np.ones((T, N), bool), p)
    for f in res["broker"].fills:
        if f.side == "buy":
            assert f.shares * f.raw_px <= 100_000 * 1.05
    w = sim.run(m, S, np.ones((T, N), bool), Params(n_hold=5, rebalance_every=5, keep_rank=10, depth=20),
                withdraw_above=900_000)
    assert len(w["equity"]) == T and (w["equity"] > 0).all()


def test_live_shadow_account_runs_alongside(cfg, synth):
    """比較用の口座（shadow_accounts）は別の状態ファイルで動き、本番の口座の結果を変えない。"""
    from kabuscout.data import PriceStore
    from autotrade import live
    o, h, l, c, v, mkt = synth
    def save(n):
        P = {k: x.iloc[:n].assign(**{"^N225": (mkt.iloc[:n] if k != "volume" else 0.0)})
             for k, x in dict(open=o, high=h, low=l, close=c, volume=v).items()}
        P["splits"] = pd.DataFrame(np.nan, index=c.index[:n], columns=list(c.columns) + ["^N225"])
        P["dividends"] = pd.DataFrame()
        PriceStore(cfg["data"]["price_dir"]).save(P)
    s = live.load_settings()
    s["model"].update(params=dict(max_iter=10, min_samples_leaf=100), sample_every=5)
    s["strategy"].update(n_hold=4, keep_rank=8, regime_ma=0)
    s["risk"]["max_order_value"] = 1e9
    s["shadow_accounts"] = {"daily": {"rebalance_every": 1, "keep_rank": 12, "risk": {"kill_drawdown": 0.6}}}
    save(850)
    r1 = live.daily(cfg, s, notify=False, today=c.index[849])
    assert "比較用の口座「daily」" in r1["text"] and "daily" in r1["shadows"]
    assert (live.state_dir(cfg) / "state_daily.json").exists()
    save(851)
    r2 = live.daily(cfg, s, notify=False, today=c.index[850])
    assert {x["code"] for x in r2["positions"]} == {x["code"] for x in r1["orders"]}
    assert len(r2["shadows"]["daily"]["positions"]) >= 1


def test_min_hold_days_blocks_early_rank_sells():
    b = _broker_with({"H1": (100, 1000.0)})
    b.mark({"H1": 1000.0})  # 買った日 = 1日目
    scores = {"A": 0.9, "H1": 0.0}
    px = {"A": 500.0, "H1": 1000.0}
    p = Params(n_hold=1, keep_rank=1, depth=5, min_hold_days=3, cost=0.0)
    assert [o for o in decide(b, scores, {"A"}, px, True, False, p) if o.side == "sell"] == []
    b.mark(px); b.mark(px)  # 3日目
    assert [o.code for o in decide(b, scores, {"A"}, px, True, False, p) if o.side == "sell"] == ["H1"]


# ---------------------------------------------------------------- AIと指数を組み合わせた口座（blend）
def test_sleeve_rebalance_tax_carry_and_idempotent():
    from autotrade.blend import Sleeve
    b = PaperBroker(1_000_000, cost=0.0)
    sl = Sleeve(share=0.5, tax=0.2)
    assert sl.new_year(2020, b, b.cash(), 100.0, "2020-01-06")
    assert sl.value(100.0) == pytest.approx(500_000) and b.cash() == pytest.approx(500_000)
    assert not sl.new_year(2020, b, b.cash(), 100.0)  # 同じ年に2回はしない
    # 2020年: AI の部分が 500,000 → 400,000（損 100,000 は繰り越し）、指数は2倍
    b._cash = 400_000
    sl.new_year(2021, b, 400_000, 200.0, "2021-01-04")
    assert sl.taxes_paid == 0 and sl.carry == [[2020, 100_000]]
    total = 400_000 + 1_000_000
    assert sl.value(200.0) + b.cash() == pytest.approx(total)
    assert sl.value(200.0) == pytest.approx(total / 2)
    # 2021年: AI の部分が +300,000 → 繰り越しの 100,000 を引いた 200,000 に 20% = 40,000
    ai = b.cash() + 300_000
    b._cash = ai
    sl.new_year(2022, b, ai, 200.0)
    assert sl.taxes_paid == pytest.approx(40_000) and sl.carry == []
    # まだ払っていない今年の税
    assert sl.pending_tax(sl.ai_start + 50_000, 2022) == pytest.approx(10_000)


def test_sleeve_rebalances_with_cash_only():
    from autotrade.blend import Sleeve
    b = PaperBroker(1_000_000, cost=0.0)
    sl = Sleeve(share=0.5)
    sl.new_year(2020, b, b.cash(), 100.0)
    b.submit([Order("A", "buy", 4000, 100.0)])
    b.execute_open("d", {"A": 100.0})   # 現金をほぼ使い切る
    sl.new_year(2021, b, b.equity({"A": 200.0}), 50.0)  # 指数が半分になっても、AI の保有は売らない
    assert b.positions()["A"].shares == 4000 and b.cash() >= 0


def test_sim_index_only_and_blend(synth):
    o, h, l, c, v, mkt = synth
    T, N = c.shape
    S = np.random.default_rng(4).random((T, N))
    m = _mk_market(c, o)
    lvl = np.linspace(100, 200, T)  # 指数は2倍に
    only = sim.run(m, S, np.ones((T, N), bool), Params(index_share=1.0), index_level=lvl)
    assert only["equity"].iloc[-1] == pytest.approx(2_000_000, rel=1e-6)
    assert len(only["trades"]) == 0
    mix = sim.run(m, S, np.ones((T, N), bool), Params(n_hold=5, rebalance_every=5, keep_rank=10, depth=20, index_share=0.5),
                  index_level=lvl)
    ai = sim.run(m, S, np.ones((T, N), bool), Params(n_hold=5, rebalance_every=5, keep_rank=10, depth=20))
    assert len(mix["trades"]) > 0
    # 組み合わせは、AI だけと指数だけの間に入る（年1回の戻しなので厳密な平均ではない）
    lo, hi = sorted([ai["equity"].iloc[-1], 2_000_000])
    assert lo * 0.8 <= mix["equity"].iloc[-1] <= hi * 1.2
    with pytest.raises(ValueError):
        sim.run(m, S, np.ones((T, N), bool), Params(index_share=0.5))
    taxed = sim.run(m, S, np.ones((T, N), bool), Params(n_hold=5, rebalance_every=5, keep_rank=10, depth=20, tax_ai=0.2))
    if ai["equity"].iloc[-1] > 1_000_000:
        assert taxed["equity"].iloc[-1] < ai["equity"].iloc[-1]


def test_index_levels_lag_and_fee():
    from autotrade.blend import index_levels
    us = pd.bdate_range("2024-01-01", periods=10)
    ix = pd.DataFrame({"spx_tr": np.arange(1, 11, dtype=float) * 100, "acwi": np.nan, "usdjpy": 100.0}, index=us)
    cal = us[1:]
    lvl, src = index_levels(ix, cal, "S&P500")
    assert pd.Timestamp(src[0]) == us[0]            # 日本の営業日には、前の米国の終値が反映される
    assert lvl[0] == pytest.approx(100 * 100)
    assert lvl[5] == pytest.approx(600 * 100 * (1 - 0.001 / 250) ** 5)


def test_live_blend_account_rebalances_once_and_flags_stale_index(cfg, synth, monkeypatch):
    from kabuscout.data import PriceStore
    from autotrade import live
    o, h, l, c, v, mkt = synth
    def save(n):
        P = {k: x.iloc[:n].assign(**{"^N225": (mkt.iloc[:n] if k != "volume" else 0.0)})
             for k, x in dict(open=o, high=h, low=l, close=c, volume=v).items()}
        P["splits"] = pd.DataFrame(np.nan, index=c.index[:n], columns=list(c.columns) + ["^N225"])
        P["dividends"] = pd.DataFrame()
        PriceStore(cfg["data"]["price_dir"]).save(P)
    lag = {"days": 1}
    def fake_levels(cfg_, cal, name, cache):
        lv = 100.0 + (cal - pd.Timestamp("2015-01-01")).days.to_numpy() * 0.05  # 日付で決まる値段（毎日少しずつ上がる）
        src = (cal - pd.Timedelta(days=lag["days"])).to_numpy()
        return lv, src
    monkeypatch.setattr(live, "_index_levels", fake_levels)
    s = live.load_settings()
    s["model"].update(params=dict(max_iter=10, min_samples_leaf=100), sample_every=5)
    s["strategy"].update(n_hold=4, keep_rank=8, regime_ma=0, index_share=0.5, index_name="S&P500")
    s["risk"]["max_order_value"] = 1e9
    s["shadow_accounts"] = {}
    k = next(j for j in range(700, 890) if c.index[j].year != c.index[j - 1].year)  # 年の最初の営業日
    save(k)  # 前の年の最後の日まで
    r1 = live.daily(cfg, s, notify=False, today=c.index[k - 1])
    st = json.loads((live.state_dir(cfg) / "state.json").read_text())
    assert st["sleeve"]["units"] > 0 and st["sleeve"]["year"] == c.index[k - 1].year
    assert "うち指数（S&P500）" in r1["text"]
    save(k + 1)  # 新しい年の最初の日
    r2 = live.daily(cfg, s, notify=False, today=c.index[k])
    assert "指数の比率を戻す" in r2["text"]
    st2 = json.loads((live.state_dir(cfg) / "state.json").read_text())
    assert st2["sleeve"]["year"] == c.index[k].year
    r3 = live.daily(cfg, s, notify=False, today=c.index[k])  # 同じデータでもう一度: 何もしない
    st3 = json.loads((live.state_dir(cfg) / "state.json").read_text())
    assert st3["sleeve"]["units"] == st2["sleeve"]["units"]
    lag["days"] = 20
    save(k + 2)
    r4 = live.daily(cfg, s, notify=False, today=c.index[k + 1])
    assert "データが古い" in r4["text"]


def test_live_receives_dividends_and_taxes_ai_gains_like_sim(cfg, synth):
    """毎日の運用も過去の検証と同じく、配当を受け取り（税引き後）、年の初めに AI の部分の前年の利益に税をかける。"""
    from kabuscout.data import PriceStore
    from autotrade import live
    o, h, l, c, v, mkt = synth
    k = next(j for j in range(700, 890) if c.index[j].year != c.index[j - 1].year)
    div = pd.DataFrame(np.nan, index=c.index, columns=list(c.columns) + ["^N225"])
    div.iloc[k - 2, :-1] = 10.0  # 前の年の終わりに全銘柄が1株10円の配当
    def save(n):
        P = {kk: x.iloc[:n].assign(**{"^N225": (mkt.iloc[:n] if kk != "volume" else 0.0)})
             for kk, x in dict(open=o, high=h, low=l, close=c, volume=v).items()}
        P["splits"] = pd.DataFrame(np.nan, index=c.index[:n], columns=list(c.columns) + ["^N225"])
        P["dividends"] = div.iloc[:n]
        PriceStore(cfg["data"]["price_dir"]).save(P)
    s = live.load_settings()
    s["model"].update(params=dict(max_iter=10, min_samples_leaf=100), sample_every=5)
    s["strategy"].update(n_hold=4, keep_rank=40, regime_ma=0, tax_ai=0.20315)
    s["risk"]["max_order_value"] = 1e9
    s["shadow_accounts"] = {}
    save(k - 4)
    live.daily(cfg, s, notify=False, today=c.index[k - 5])
    save(k - 3)
    live.daily(cfg, s, notify=False, today=c.index[k - 4])   # 買いが約定
    save(k - 1)
    r = live.daily(cfg, s, notify=False, today=c.index[k - 2])  # 配当の日を含む
    assert "配当" in r["text"]
    save(k + 1)
    live.daily(cfg, s, notify=False, today=c.index[k])          # 新しい年の最初の日: 前年の税（または損失の繰り越し）
    st = json.loads((live.state_dir(cfg) / "state.json").read_text())
    sl = st["sleeve"]
    assert sl["year"] == c.index[k].year and sl["tax"] == pytest.approx(0.20315)
    assert sl["taxes_paid"] > 0 or sl["carry"]


def _save_synth(cfg, synth, n, partial_last: int | None = None, proxy_missing: bool = False):
    """合成の株価を n 日分保存する。partial_last=k なら、最終日は先頭の k 銘柄（と指数）しか株価が無い（Yahoo の更新途中の再現）。
    proxy_missing: 最終日に指数（^N225）の株価も無い（2026-11-27 の本番と同じ形）"""
    from kabuscout.data import PriceStore
    o, h, l, c, v, mkt = synth
    P = {k: x.iloc[:n].assign(**{"^N225": (mkt.iloc[:n] if k != "volume" else 0.0)})
         for k, x in dict(open=o, high=h, low=l, close=c, volume=v).items()}
    if partial_last is not None:
        for k in P:
            P[k].iloc[-1, partial_last:len(c.columns) + (1 if proxy_missing else 0)] = np.nan
    P["splits"] = pd.DataFrame(np.nan, index=c.index[:n], columns=list(c.columns) + ["^N225"])
    P["dividends"] = pd.DataFrame()
    PriceStore(cfg["data"]["price_dir"]).save(P)


def _small_settings():
    from autotrade import live
    s = live.load_settings()
    s["model"].update(params=dict(max_iter=10, min_samples_leaf=100), sample_every=5)
    s["strategy"].update(n_hold=4, keep_rank=8, regime_ma=0)
    s["risk"]["max_order_value"] = 1e9
    s["shadow_accounts"] = {}
    return s


def test_live_ignores_incomplete_latest_day(cfg, synth):
    """2026-11-27 の本番の初回: 最新日の株価が3/3694銘柄しか無いのに、その日で判断して候補が1銘柄になった。
    株価が揃っていない最新日は使わず、前の日で判断する。揃ったら次の実行でその日を処理する。"""
    from autotrade import live
    c = synth[3]
    s = _small_settings()
    _save_synth(cfg, synth, 851, partial_last=2)  # 850日目（最終日）は2銘柄しか無い
    r1 = live.daily(cfg, s, notify=False, today=c.index[850])
    assert r1["date"] == str(c.index[849].date())  # 揃っている 849日目で判断
    assert "揃っていない" in r1["text"] and r1["new_day"]
    assert len(r1["orders"]) == 4  # 候補が全銘柄から選ばれる（揃っていない日だと2銘柄しか候補が無い）
    st = json.loads((live.state_dir(cfg) / "state.json").read_text())
    assert st["last_date"] == str(c.index[849].date()) and st["version"] == live.STATE_VERSION
    # 同じ（揃っていない）データでもう一度: 新しい営業日は無い。前回の注文はそのまま残り、通知もしない
    r2 = live.daily(cfg, s, notify=False, today=c.index[850])
    assert not r2["new_day"] and r2["orders"] == r1["orders"] and r2["positions"] == []
    # 揃わないまま日が経ったら、新しい営業日が無くても知らせる
    r2b = live.daily(cfg, s, notify=False, today=c.index[855])
    assert r2b["stale"] and not r2b["new_day"] and "新しい営業日を処理できていません" in r2b["text"]
    # 850日目が揃った: その日の寄付で約定する
    _save_synth(cfg, synth, 851)
    r3 = live.daily(cfg, s, notify=False, today=c.index[850])
    assert r3["new_day"] and r3["date"] == str(c.index[850].date()) and "揃っていない" not in r3["text"]
    assert {x["code"] for x in r3["positions"]} == {x["code"] for x in r1["orders"]}


def test_live_ignores_incomplete_latest_day_even_without_the_index(cfg, synth):
    """2026-11-27 の本番と同じ形: 最新日は指数（ETF）にも株価が無く、数銘柄だけにある。その日は使わず、注意書きを出す。"""
    from autotrade import live
    c = synth[3]
    _save_synth(cfg, synth, 851, partial_last=3, proxy_missing=True)
    r = live.daily(cfg, _small_settings(), notify=False, today=c.index[850])
    assert r["date"] == str(c.index[849].date()) and len(r["orders"]) == 4
    assert f"{c.index[850].date()} の株価がまだ揃っていない（3/41銘柄" in r["text"]


def test_live_recreates_accounts_from_the_old_version_once(cfg, synth):
    """前の版（version が無い）の口座は一度だけ作り直し、元のファイルは .v1.json として残す。新しい版の口座はそのまま続く。"""
    from autotrade import live
    c = synth[3]
    s = _small_settings()
    _save_synth(cfg, synth, 850)
    live.daily(cfg, s, notify=False, today=c.index[849])
    path = live.state_dir(cfg) / "state.json"
    st = json.loads(path.read_text())
    st.pop("version")
    st["last_date"] = str(c.index[845].date())  # 前の版の印として、古い日付の状態にしておく
    path.write_text(json.dumps(st))
    _save_synth(cfg, synth, 851)
    r = live.daily(cfg, s, notify=False, today=c.index[850])
    assert (live.state_dir(cfg) / "state.v1.json").exists()
    st2 = json.loads(path.read_text())
    assert st2["version"] == live.STATE_VERSION and len(st2["equity"]) == 1  # 作り直した口座は最新日の1日分だけ
    assert r["positions"] == [] and len(r["orders"]) >= 1
    _save_synth(cfg, synth, 852)
    live.daily(cfg, s, notify=False, today=c.index[851])
    st3 = json.loads(path.read_text())
    assert len(st3["equity"]) == 2 and len(st3["broker"]["positions"]) > 0  # 作り直した口座の注文が約定して続く
