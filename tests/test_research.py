"""research/ の検証用の部品（本番では使わない）。作り物の株価だけで確かめる。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))
import earn_timing as et  # noqa: E402
import event_nets as en  # noqa: E402

PROXY = "^N225"


def _args(panel):
    close = panel["close"].drop(columns=[PROXY])
    return close, panel["volume"][close.columns], panel["close"][PROXY]


def _events(close):
    idx = close.index
    d_late, d_early, d_plain = idx[100], idx[150], idx[200]
    ev = pd.DataFrame({"code": ["1001", "1002", "1003"], "date": [d_late, d_early, d_plain], "surprise": [np.nan, 5.0, np.nan]})
    times = {("1001", d_late): d_late + pd.Timedelta(hours=15, minutes=30), ("1002", d_early): d_early + pd.Timedelta(hours=13)}
    return ev, times


def test_timed_base_equals_production(panel):
    close, vol, mkt = _args(panel)
    ev, times = _events(close)
    et.STATE.update(mode="base", times=times)
    a, b = et._ORIG(ev, close, vol, mkt), et.timed_wide(ev, close, vol, mkt)
    for k in ("earn_age", "earn_react", "earn_surprise", "earn_vol", "earn_to_next"):
        assert np.allclose(a[k].to_numpy("float64"), b[k].to_numpy("float64"), equal_nan=True), k
    assert b["earn_early"]["1002"].iloc[151] == 1 and b["earn_early"]["1001"].iloc[101] == 0


def test_timed_early_moves_only_early_announcements(panel):
    close, vol, mkt = _args(panel)
    ev, times = _events(close)
    et.STATE.update(mode="base", times=times)
    base = et.timed_wide(ev, close, vol, mkt)
    et.STATE["mode"] = "early"
    early = et.timed_wide(ev, close, vol, mkt)
    et.STATE["mode"] = "base"
    for code in ("1001", "1003"):  # 15時以降・時刻不明は今と同じ
        assert np.allclose(base["earn_age"][code].to_numpy("float64"), early["earn_age"][code].to_numpy("float64"), equal_nan=True)
    assert np.isnan(base["earn_age"]["1002"].iloc[150]) and base["earn_age"]["1002"].iloc[151] == 0
    assert early["earn_age"]["1002"].iloc[150] == 0  # 場中の発表は発表日の引けから使う
    c, m = close["1002"].to_numpy(), mkt.to_numpy()
    assert np.isclose(early["earn_react"]["1002"].iloc[150], c[150] / c[149] - 1 - (m[150] / m[149] - 1), atol=1e-6)


def test_rev_features_do_not_use_future_prices(panel):
    close, vol, mkt = _args(panel)
    idx = close.index
    rev = pd.DataFrame({"pubdate": [idx[120] + pd.Timedelta(hours=11), idx[180] + pd.Timedelta(hours=16)], "code": ["1001", "1004"]})
    a = en.rev_wide(rev, close, vol, mkt)
    assert a["rev_age"]["1001"].iloc[120] == 0 and np.isnan(a["rev_age"]["1004"].iloc[180]) and a["rev_age"]["1004"].iloc[181] == 0
    T = 185
    c2, v2 = close.copy(), vol.copy()
    c2.iloc[T + 1:] *= 1.7
    v2.iloc[T + 1:] *= 3
    b = en.rev_wide(rev, c2, v2, mkt)
    for k in en.REV_FEATURES:
        assert np.allclose(a[k].iloc[: T + 1].to_numpy("float64"), b[k].iloc[: T + 1].to_numpy("float64"), equal_nan=True), k


def test_title_rules():
    yes = ["自己株式取得に係る事項の決定に関するお知らせ", "自己株式立会外買付取引（ToSTNeT-3）による自己株式の買付けに関するお知らせ",
           "自己株式の取得及び消却に関するお知らせ", "自己株式の取得枠の設定に関するお知らせ"]
    no = ["自己株式の取得状況に関するお知らせ", "自己株式の取得結果及び取得終了に関するお知らせ", "第三者割当による自己株式の処分に関するお知らせ",
          "Ａ社による自己株式の公開買付けへの応募に関するお知らせ", "（訂正）自己株式取得に係る事項の決定に関するお知らせ", "自己株式の消却に関するお知らせ",
          "譲渡制限付株式報酬としての自己株式の処分に関するお知らせ"]
    assert all(en.is_buyback(t) for t in yes) and not any(en.is_buyback(t) for t in no)
    assert en.is_revision("通期業績予想の修正に関するお知らせ") and en.is_revision("業績予想と実績値との差異に関するお知らせ")
    assert not en.is_revision("（訂正）業績予想の修正に関するお知らせ") and not en.is_revision("配当予想の修正に関するお知らせ")
    assert "buyback" not in en.classify_new("自己株式の取得状況に関するお知らせ")
