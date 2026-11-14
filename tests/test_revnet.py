"""比較用の口座その3「修正好反応」（kabuscout/revnet.py、ジャーナル 85）。本番の候補・AI・売買ルールは変えない。"""
import numpy as np
import pandas as pd

from kabuscout import learner, report, revnet, strategies


def _pool():
    return pd.DataFrame({
        "date": pd.to_datetime(["2026-11-10"] * 4), "code": ["1001", "1002", "1003", "1004"],
        "x": [1, 1, 0, 0],                                  # 本番の網（偽物）に当たるか
        "rev_age": [2.0, np.nan, 0.0, 3.0], "rev_react": [0.08, np.nan, 0.06, 0.01], "rev_vol": [2.0, np.nan, 1.6, 3.0],
    }, index=[10, 11, 12, 13])


def _fake_flags(df, params):
    d = df.copy()
    d["s_a"] = d["x"].astype("int8")
    d["n_strat"] = d["s_a"]
    d = d[d["n_strat"] > 0].copy()
    d["strategies"] = "a"
    return d


def test_dataset_adds_rev_rows_without_touching_production_rows(monkeypatch):
    monkeypatch.setattr(revnet, "apply_flags", _fake_flags)
    pl = _pool()
    d = revnet.dataset(pl, {}).set_index("code")
    assert list(d.index) == ["1001", "1002", "1003"]          # 1004 は反応が小さいので入らない
    assert d.loc["1001", "strategies"] == "a,rev_drift" and d.loc["1001", "n_strat"] == 2 and d.loc["1001", "s_rev_drift"] == 1
    assert d.loc["1002", "strategies"] == "a" and d.loc["1002", "n_strat"] == 1 and d.loc["1002", "s_rev_drift"] == 0
    assert d.loc["1003", "strategies"] == "rev_drift" and d.loc["1003", "n_strat"] == 1 and d.loc["1003", "s_a"] == 0
    prod = _fake_flags(pl, {}).set_index("code")             # 本番の候補表はそのまま
    assert list(prod.index) == ["1001", "1002"] and "s_rev_drift" not in prod


def test_rev_rows_use_the_earnings_drift_rule_and_extra_flag():
    cfg = {"labeling": {"tp_atr": 2.0, "sl_atr": 1.0, "horizon_days": 10,
                        "per_strategy": {"earnings_drift": {"tp_atr": 6.0, "sl_atr": 3.0, "horizon_days": 20}}},
           "learning": {"extra_features": []}}
    c = revnet.cfg_for(cfg)
    assert strategies.plan_for(["rev_drift"], c["labeling"])["tp_atr"] == 6.0
    assert strategies.plan_for(["earnings_drift", "rev_drift"], c["labeling"])["source"] == "earnings_drift"
    assert strategies.plan_for(["rev_drift"], cfg["labeling"])["tp_atr"] == 2.0   # 本番の設定には rev_drift のルールが無い
    assert learner.model_features(c)[-1] == "s_rev_drift" and "s_rev_drift" not in learner.model_features(cfg)
    assert "rev_drift" not in strategies.BY_KEY and "s_rev_drift" not in learner.MODEL_FEATURES


def test_wide_features_match_research():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))
    import event_nets as en
    idx = pd.bdate_range("2026-01-05", periods=40)
    rng = np.random.default_rng(0)
    close = pd.DataFrame(100 * np.exp(rng.normal(0, 0.02, (40, 3)).cumsum(0)), index=idx, columns=["1001", "1002", "1003"])
    vol = pd.DataFrame(rng.uniform(1e5, 3e5, (40, 3)), index=idx, columns=close.columns)
    mkt = pd.Series(np.exp(rng.normal(0, 0.01, 40).cumsum()), index=idx)
    rev = pd.DataFrame({"pubdate": [idx[25] + pd.Timedelta(hours=11), idx[30] + pd.Timedelta(hours=15, minutes=30)], "code": ["1001", "1003"]})
    a, b = revnet.wide_features(rev, close, vol, mkt), en.rev_wide(rev, close, vol, mkt)
    for k in revnet.REV_FEATURES:
        pd.testing.assert_frame_equal(a[k], b[k])
    assert a["rev_age"].loc[idx[25], "1001"] == 0 and np.isnan(a["rev_age"].loc[idx[30], "1003"]) and a["rev_age"].loc[idx[31], "1003"] == 0


def test_rows_merge_saved_and_recent_titles(tmp_path):
    cfg = {"data": {"db_path": tmp_path / "kabu.db"}}
    td = pd.DataFrame({"pubdate": pd.to_datetime(["2026-11-10 15:00", "2026-11-11 11:00"]), "code": ["1001", "1002"],
                       "title": ["a", "b"], "events": ["earnings,revision", "buyback"]})
    assert list(revnet.rows(cfg, td)["code"]) == ["1001"]
    pd.DataFrame({"pubdate": pd.to_datetime(["2026-11-10 15:00", "2026-11-13 13:00"]), "code": ["1001", "1003"]}).to_parquet(tmp_path / revnet.RECENT_FILE)
    assert sorted(revnet.rows(cfg, td)["code"]) == ["1001", "1003"]


def test_report_lines():
    pt = dict(equity=1_000_000, rev=dict(equity=1_020_000, diff_orders=3, drawdown=-0.05, events=["注文 1003 テスト (翌寄付)"]))
    L = report._side_lines(pt)
    assert "修正好反応を足した口座" in L[0] and "+20,000円" in L[0] and "本番と違う注文 3回" in L[0] and L[1].startswith("• 注文")
    assert report._side_lines(dict(equity=1, rev_wait="修正好反応の口座: 準備中")) == ["修正好反応の口座: 準備中"]
