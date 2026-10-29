"""分割の補正: 呼値（1ティック）は分割調整後の株価ではなく、その日の実際の株価で決まる。"""
import numpy as np
import pandas as pd
import pytest

from kabuscout import indicators, learner
from kabuscout.risk import tick_size


def test_tick_table():
    p = [1, 3000, 3001, 5000, 5001, 30000, 30001, 50000, 50001, 300000, 300001]
    assert tick_size(p).tolist() == [1, 1, 5, 5, 10, 10, 50, 50, 100, 100, 500]


def _stop_hit(entry, fac):
    """1日目に損切ライン（entry − 10）を下回る取引。fac = 実際の株価 ÷ 分割調整後の株価（エントリー日）。"""
    h, l = np.array([[entry]]), np.array([[entry - 50]])
    p = dict(entry=np.array([entry]), o=h.copy(), h=h, l=l, c=h.copy(), fac=np.array([fac]))
    r = learner.outcomes(p, np.array([10.0]), 2.0, 1.0, 1)
    return float(r["ret"][0] * entry + entry)   # 約定値（調整後の株価で）


def test_stop_slippage_uses_actual_price():
    # 調整後 1,000円・のちに1:2分割 → 当時の実際の株価は2,000円 → 呼値1円 → 調整後では0.5円の滑り
    assert _stop_hit(1000.0, 2.0) == pytest.approx(990 - 0.5)
    # 分割なし → 1円
    assert _stop_hit(1000.0, 1.0) == pytest.approx(990 - 1.0)
    # 調整後 1,000円・のちに1:4分割 → 実際は4,000円 → 呼値5円 → 調整後では1.25円
    assert _stop_hit(1000.0, 4.0) == pytest.approx(990 - 1.25)
    # 係数が無い（欠けている）ときは分割なしとして扱う
    assert _stop_hit(1000.0, np.nan) == pytest.approx(990 - 1.0)


def test_actual_price_from_split_factor(panel):
    fac = panel["close"] * 0 + 1.0
    fac.loc[: fac.index[100], "1004"] = 3.0          # 1004 は101日目に1:3の分割（それより前の実際の株価は3倍）
    ind = indicators.compute({**panel, "split_factor": fac}, "^N225")
    c = ind["close"]["1004"]
    assert ind["px_actual"]["1004"].iloc[50] == pytest.approx(c.iloc[50] * 3)
    assert ind["px_actual"]["1004"].iloc[200] == pytest.approx(c.iloc[200])
    assert ind["split_factor"]["1004"].iloc[50] == pytest.approx(3.0)
    assert ind["price_log"]["1004"].iloc[50] == pytest.approx(np.log10(c.iloc[50] * 3))


def test_label_passes_split_factor_to_tick():
    idx = pd.bdate_range("2025-01-06", periods=10)
    f = lambda v: pd.DataFrame({"X": np.full(10, v)}, index=idx)
    ind = {"open": f(1000.0), "high": f(1005.0), "low": f(995.0), "close": f(1000.0), "split_factor": f(2.0)}
    ind["low"].iloc[4] = 980.0
    cfg = {"labeling": dict(horizon_days=5, tp_atr=2.0, sl_atr=1.0, stop_mode="intraday")}
    r = learner.label(ind, pd.DataFrame({"_ti": [3], "_ci": [0], "atr": [10.0]}), cfg).iloc[0]
    assert r["ret"] == pytest.approx((990 - 0.5) / 1000 - 1)
