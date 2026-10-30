"""先読みがないこと。
- 特徴量・戦略の判定: ある日より後の株価を書き換えても、その日までの値が変わらない
- ラベル（トリプルバリア）: 翌日の寄付から数えた保有期間の値だけを使う（シグナル日以前と、期間より後の値は結果に影響しない）"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kabuscout import indicators, learner, strategies

PROXY = "^N225"
T = 260  # この日（0始まり）までの値が変わらないことを確かめる


def _future_changed(panel, t):
    rng = np.random.default_rng(7)
    out = {}
    for k, v in panel.items():
        v = v.copy()
        noise = rng.uniform(0.5, 1.5, size=v.iloc[t + 1:].shape)
        v.iloc[t + 1:] = v.iloc[t + 1:].to_numpy() * noise
        out[k] = v
    return out


def _same_until(a: pd.DataFrame, b: pd.DataFrame, t: int) -> bool:
    x, y = a.iloc[: t + 1].to_numpy(dtype="float64"), b.iloc[: t + 1].to_numpy(dtype="float64")
    return np.allclose(x, y, equal_nan=True, rtol=1e-9, atol=1e-9)


def test_features_do_not_use_future_prices(panel):
    sectors = pd.Series({"1001": "A", "1002": "A", "1003": "B", "1004": "B"})
    a = indicators.compute(panel, PROXY, sectors)
    b = indicators.compute(_future_changed(panel, T), PROXY, sectors)
    names = [k for k in dict.fromkeys(indicators.FEATURES + indicators.CANDIDATE_FEATURES + indicators.RAW_FIELDS) if k in a]
    assert len(names) > 60
    changed = [k for k in names if not _same_until(a[k], b[k], T)]
    assert not changed, f"未来の株価で値が変わった特徴量: {changed}"
    # 書き換えが効いていること（未来の値そのものは変わる）
    assert not np.allclose(a["close"].iloc[T + 1:].to_numpy(), b["close"].iloc[T + 1:].to_numpy())


def _with_event_fields(ind: dict) -> dict:
    """決算・適時開示・配当などの値（株価以外）は「何も起きていない」として空欄で足す（株価からの判定だけを確かめる）。"""
    src = Path(strategies.__file__).read_text()
    like = ind["close"]
    for key in set(re.findall(r'i\["(\w+)"\]', src)):
        if key not in ind:
            ind[key] = like * np.nan
    return ind


def test_signals_do_not_use_future_prices(panel):
    a = _with_event_fields(indicators.compute(panel, PROXY))
    b = _with_event_fields(indicators.compute(_future_changed(panel, T), PROXY))
    mask = a["close"].notna()
    sa = strategies.detect_all(a, mask, strategies.loose_params())
    sb = strategies.detect_all(b, mask, strategies.loose_params())
    assert sum(int(v.iloc[: T + 1].to_numpy().sum()) for v in sa.values()) > 0, "判定が1件も出ないと確かめにならない"
    for k in sa:
        assert (sa[k].iloc[: T + 1] == sb[k].iloc[: T + 1]).all().all(), f"{k} の判定が未来の株価で変わった"


# ---------------------------------------------------------------- ラベル
H = 5


def _ind(n=20, price=1000.0):
    idx = pd.bdate_range("2025-01-06", periods=n)
    f = lambda: pd.DataFrame({"X": np.full(n, price)}, index=idx)
    return {"open": f(), "high": f() + 5, "low": f() - 5, "close": f()}


def _label(ind, ti, atr=10.0, tp=2.0, sl=1.0, h=H, stop="intraday"):
    cfg = {"labeling": dict(horizon_days=h, tp_atr=tp, sl_atr=sl, stop_mode=stop)}
    df = pd.DataFrame({"_ti": [ti], "_ci": [0], "atr": [atr]})
    return learner.label(ind, df, cfg).iloc[0]


def test_label_ignores_signal_day_and_after_horizon():
    ti = 3
    base = _label(_ind(), ti)
    ind = _ind()
    for k in ("open", "high", "low", "close"):   # シグナル日とその前の値を大きく変える
        ind[k].iloc[: ti + 1] *= 3
    ind["high"].iloc[ti + 1 + H:] = 99999       # 保有期間より後に利確ラインを大きく超える
    ind["low"].iloc[ti + 1 + H:] = 1
    got = _label(ind, ti)
    for k in ("entry", "label", "ret", "exit_days", "complete"):
        assert got[k] == base[k], k


def test_label_entry_is_next_day_open():
    ind = _ind()
    ind["open"].iloc[4] = 1100.0
    r = _label(ind, 3)
    assert r["entry"] == 1100.0
    assert r["tp"] == 1100.0 + 2 * 10 and r["sl"] == 1100.0 - 10


def test_label_hand_cases():
    # 1日目に利確ライン（1000+20）に届く → 利確・ret=+2%
    ind = _ind(); ind["high"].iloc[4] = 1025
    r = _label(ind, 3)
    assert (r["label"], r["exit_days"], r["complete"]) == (1, 1, True) and r["ret"] == pytest.approx(0.02)
    # 2日目に窓を開けて損切ライン（990）を下回って始まる → 寄付で約定し、さらに1ティック（1円）不利
    ind = _ind(); ind["open"].iloc[5] = 980; ind["low"].iloc[5] = 975
    r = _label(ind, 3)
    assert (r["label"], r["exit_days"]) == (0, 2) and r["ret"] == pytest.approx((980 - 1) / 1000 - 1)
    # 同じ日に両方に触れたら損切（保守的）
    ind = _ind(); ind["high"].iloc[4] = 1030; ind["low"].iloc[4] = 985
    r = _label(ind, 3)
    assert (r["label"], r["exit_days"]) == (0, 1) and r["ret"] == pytest.approx((990 - 1) / 1000 - 1)
    # どちらにも届かなければ H 日目の終値
    ind = _ind(); ind["close"].iloc[3 + H] = 1007
    r = _label(ind, 3)
    assert (r["label"], r["exit_days"]) == (0, H) and r["ret"] == pytest.approx(0.007)
    # 保有期間の分の株価がまだ無ければ未確定
    r = _label(_ind(n=6), 3)
    assert not r["complete"]


def test_label_close_stop_sells_next_open():
    # stop="close": 1日目の終値が損切ラインを割る → 2日目の寄付で売る（滑りなし）
    ind = _ind(); ind["close"].iloc[4] = 985; ind["low"].iloc[4] = 984; ind["open"].iloc[5] = 983
    r = _label(ind, 3, stop="close")
    assert (r["label"], r["exit_days"]) == (0, 2) and r["ret"] == pytest.approx(983 / 1000 - 1)
