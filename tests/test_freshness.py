"""静かな失敗を知らせる確認（ジャーナル §72 の優先1）: データの鮮度・監理銘柄の一覧の0件・5分足の取り直しの期間。"""
import json

import pandas as pd

from kabuscout import alerts, freshness, intraday


def _cfg(tmp_path):
    return {"data": {"db_path": tmp_path / "kabuscout.db"}}


def _write(tmp_path, tdnet_last, macro_last, alerts_date):
    pd.DataFrame({"pubdate": [pd.Timestamp("2026-10-01 15:00"), pd.Timestamp(tdnet_last)], "code": ["1001", "1002"],
                  "title": ["a", "b"], "events": ["", ""]}).to_parquet(tmp_path / "tdnet.parquet")
    pd.DataFrame({"usdjpy": [150.0]}, index=pd.DatetimeIndex([macro_last])).to_parquet(tmp_path / "macro.parquet")
    (tmp_path / "alerts.json").write_text(json.dumps({"date": alerts_date, "codes": {"9999": "監理"}}))


CAL = pd.DatetimeIndex(["2026-11-02", "2026-11-04", "2026-11-05"])  # 11-03 は祝日


def test_fresh_data_is_quiet(tmp_path):
    _write(tmp_path, "2026-11-04 18:00", "2026-11-04", "2026-11-05")
    assert freshness.check(_cfg(tmp_path), CAL) == []


def test_stale_tdnet_is_reported_once(tmp_path):
    _write(tmp_path, "2026-10-31 16:00", "2026-11-04", "2026-11-05")
    out = freshness.check(_cfg(tmp_path), CAL)
    assert len(out) == 1 and "適時開示" in out[0] and "10/31" in out[0]


def test_holiday_does_not_count_as_stale(tmp_path):
    # 11-04 の実行: 前の営業日は 11-02（11-03 は祝日）。11-02 の開示まであれば知らせない
    _write(tmp_path, "2026-11-02 17:00", "2026-11-02", "2026-11-04")
    assert freshness.check(_cfg(tmp_path), CAL[:2]) == []


def test_old_alert_list_and_stale_macro(tmp_path):
    _write(tmp_path, "2026-11-04 18:00", "2026-10-28", "2026-10-20")
    out = freshness.check(_cfg(tmp_path), CAL)
    assert any("海外市場" in s for s in out) and any("監理" in s for s in out)


def test_empty_alert_parse_keeps_previous_list(tmp_path, monkeypatch):
    (tmp_path / "alerts.json").write_text(json.dumps({"date": "2026-11-04", "codes": {"9999": "監理"}}))
    monkeypatch.setattr(alerts, "fetch", lambda today: {})
    assert alerts.load(_cfg(tmp_path), pd.Timestamp("2026-11-05")) == {"9999": "監理"}
    assert json.loads((tmp_path / "alerts.json").read_text())["date"] == "2026-11-04"  # 0件では上書きしない


def test_intraday_period_covers_a_long_outage(tmp_path):
    assert intraday.auto_period(tmp_path, pd.Timestamp("2026-11-20 16:45")) == "60d"  # 初回
    pd.DataFrame({"datetime": [pd.Timestamp("2026-11-05 15:20")], "code": ["7203"]}).to_parquet(tmp_path / "2026-11.parquet")
    assert intraday.auto_period(tmp_path, pd.Timestamp("2026-11-06 16:45")) == "8d"   # 毎日動いているとき
    assert intraday.auto_period(tmp_path, pd.Timestamp("2026-11-20 16:45")) == "18d"  # 15日止まった後
    assert intraday.auto_period(tmp_path, pd.Timestamp("2027-03-01 16:45")) == "60d"  # Yahoo に残る上限
