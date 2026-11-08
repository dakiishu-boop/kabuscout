import numpy as np
import pandas as pd

from kabuscout import backtest, fills


def test_fill_slippage_signs_and_round_trip(tmp_path):
    cfg = {"data": {"db_path": tmp_path / "x.db"}, "research": {"cost": 0.002}}
    fills.add(cfg, "2026-11-10", "7203", "buy", "open", 100, 1002.0)
    fills.add(cfg, "2026-11-12", "7203", "sell", "stop", 100, 948.0, ref=950.0)
    fills.add(cfg, "2026-11-13", "7203", "sell", "limit", 100, 1100.0, ref=1100.0)
    d = fills.load(cfg)
    idx = pd.to_datetime(["2026-11-10", "2026-11-12", "2026-11-13"])
    px = {k: pd.DataFrame({"7203": v}, index=idx) for k, v in
          (("open", [1000.0, 960.0, 1090.0]), ("close", [1000.0, 950.0, 1100.0]), ("volume", [1e5, 1e5, 1e5]))}
    r = fills.with_reference(d, px)
    assert np.isclose(r.loc[0, "slip"], 0.002) and np.isclose(r.loc[0, "ticks"], 2)  # 始値より2円高く買った＝不利
    assert np.isclose(r.loc[1, "ticks"], 2)  # 逆指値950に948で約定＝2ティック不利
    assert np.isclose(r.loc[2, "slip"], 0.0)
    s = fills.summary(r, 0.002)
    assert s["measured"] == 3 and "round_trip" in s


def test_fill_rejects_stop_without_ref(tmp_path):
    cfg = {"data": {"db_path": tmp_path / "x.db"}}
    try:
        fills.add(cfg, "2026-11-10", "7203", "sell", "stop", 100, 948.0)
        assert False
    except ValueError:
        pass


def test_stats_extra_metrics_and_period():
    idx = pd.bdate_range("2018-01-01", "2020-12-31")
    eq = pd.Series(np.linspace(1e6, 1.3e6, len(idx)) * (1 + 0.01 * np.sin(np.arange(len(idx)))), index=idx)
    tr = pd.DataFrame({"date": idx[::20], "net": np.tile([0.05, -0.02], len(idx[::20]))[: len(idx[::20])]})
    tr["pnl"] = tr["net"] * 2e5
    s = backtest._stats(eq, tr, pd.Series(0.5, index=idx))
    for k in ("sharpe", "sortino", "calmar", "profit_factor", "exposure", "per_year"):
        assert s[k] is not None
    p = backtest.period_stats(dict(equity=eq, trades=tr, exposure=pd.Series(0.5, index=idx)), "2019-01-01")
    assert p["trades"] == int((tr["date"] >= "2019-01-01").sum())
