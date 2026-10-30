"""前向きの答え合わせ（forward.py）: 全候補の記録は追記だけで、同じ日を2回実行しても最初の記録が使われる。
集計するのは、その日の全候補の結果が出揃った日だけ。"""
from datetime import datetime

import numpy as np
import pandas as pd

from kabuscout import forward


class _Clock:
    """forward.datetime の代わり。now() のたびに決めた時刻を返す。"""

    def __init__(self, *times):
        self.times = list(times)

    def now(self, tz=None):
        return self.times.pop(0)


def _scored(date, tag, n=3):
    return pd.DataFrame({"date": [date] * n, "code": [f"{k}" for k in range(n)], "ai_score": [tag] * n})


def test_same_day_twice_keeps_first_run(cfg, monkeypatch):
    monkeypatch.setattr(forward, "datetime", _Clock(datetime(2026, 10, 29, 16, 40), datetime(2026, 10, 29, 18, 5),
                                                    datetime(2026, 10, 30, 16, 40)))
    forward.save(cfg, _scored("2026-10-29", 0.1), "2026-10-29")
    forward.save(cfg, _scored("2026-10-29", 0.9), "2026-10-29")   # 同じ日の再実行（予測が違う）
    forward.save(cfg, _scored("2026-10-30", 0.5), "2026-10-30")
    assert len(list(forward.log_dir(cfg).glob("*.parquet"))) == 3     # 上書きしない
    first = forward.load(cfg)
    d29 = first[first["date"] == "2026-10-29"]
    assert len(d29) == 3 and (d29["ai_score"] == 0.1).all()           # 最初の実行の予測
    assert set(first["date"].dt.strftime("%Y-%m-%d")) == {"2026-10-29", "2026-10-30"}
    both = forward.load(cfg, first_only=False)
    assert len(both) == 9


def test_same_second_does_not_overwrite(cfg, monkeypatch):
    t = datetime(2026, 10, 29, 16, 40, 0)
    monkeypatch.setattr(forward, "datetime", _Clock(t, t))
    a = forward.save(cfg, _scored("2026-10-29", 0.1), "2026-10-29")
    b = forward.save(cfg, _scored("2026-10-29", 0.2), "2026-10-29")
    assert a != b and b.endswith("_1.parquet")
    assert len(forward.load(cfg, first_only=False)) == 6


def test_wide_log_is_separate(cfg):
    forward.save(cfg, _scored("2026-10-29", 0.3), "2026-10-29", sub="scored_wide")
    assert forward.load(cfg) is None                                  # 本番の記録には混ざらない
    assert len(forward.load(cfg, sub="scored_wide")) == 3


def test_only_fully_settled_dates_are_counted():
    dates = pd.bdate_range("2026-01-05", periods=30)
    ds = pd.DataFrame({"date": np.repeat(dates, 2)})
    logged = pd.DataFrame({"date": dates, "x": range(30)})
    h = 10
    m = forward.matured(logged, ds, h)
    # 最長10営業日の保有が終わっている日 = 最新日から数えて11日目より前だけ
    assert m["date"].max() == dates[-h - 1] and len(m) == 30 - h
    # 日数が足りなければ何も集計しない
    assert forward.matured(logged, ds.head(2 * h), h).empty


def test_daily_metrics_skip_small_days():
    rng = np.random.default_rng(0)
    rows = []
    for d, n in ((pd.Timestamp("2026-01-05"), 30), (pd.Timestamp("2026-01-06"), 10)):
        s = rng.normal(size=n)
        rows.append(pd.DataFrame({"date": d, "score": s, "ret": s * 0.01 + rng.normal(0, 0.001, n)}))
    m = forward.daily_metrics(pd.concat(rows), "score")
    assert len(m) == 1                    # 候補が20件未満の日は除く
    assert m.iloc[0]["ic"] > 0.9
