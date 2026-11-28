"""夜中にずれた起動でその日の分を取りこぼさないための仕組み（ジャーナル §91）。"""
import json
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from kabuscout import data, db, pipeline, site

JST = ZoneInfo("Asia/Tokyo")


def test_last_final_day_keeps_unfinished_bars_out():
    assert data.last_final_day(datetime(2026, 11, 26, 2, 39, tzinfo=JST)) == pd.Timestamp("2026-11-25")
    assert data.last_final_day(datetime(2026, 11, 26, 9, 17, tzinfo=JST)) == pd.Timestamp("2026-11-25")
    assert data.last_final_day(datetime(2026, 11, 26, 15, 40, tzinfo=JST)) == pd.Timestamp("2026-11-26")
    idx = pd.to_datetime(["2026-11-25", "2026-11-26"])
    out = data.drop_unfinished({"close": pd.DataFrame({"7203": [1.0, 2.0]}, index=idx), "volume": pd.DataFrame()},
                               pd.Timestamp("2026-11-25"))
    assert list(out["close"].index) == [pd.Timestamp("2026-11-25")] and out["volume"].empty


def test_expected_day_skips_weekends_and_holidays():
    assert pipeline.expected_day(pd.Timestamp("2026-11-28")) == pd.Timestamp("2026-11-27")  # 土曜 → 金曜
    assert pipeline.expected_day(pd.Timestamp("2026-11-23")) == pd.Timestamp("2026-11-20")  # 勤労感謝の日（月）→ 金曜
    assert pipeline.expected_day(pd.Timestamp("2026-11-25")) == pd.Timestamp("2026-11-25")


def _run(monkeypatch, cfg, usable):
    seq = iter(usable)
    monkeypatch.setattr(pipeline, "_usable_date", lambda store, cutoff: pd.Timestamp(next(seq)))
    monkeypatch.setattr(pipeline.time, "sleep", lambda s: None)
    calls = []
    ok = pipeline._refetch_if_incomplete(None, ["7203"], cfg, pd.Timestamp("2026-11-27"),
                                         update=lambda *a: calls.append(1))
    return ok, len(calls)


def test_refetch_until_the_day_is_complete(monkeypatch, cfg):
    assert _run(monkeypatch, cfg, ["2026-11-27"]) == (True, 0)                               # 揃っていれば取り直さない
    assert _run(monkeypatch, cfg, ["2026-11-26", "2026-11-27"]) == (True, 1)                 # 1回の取り直しで揃った
    assert _run(monkeypatch, cfg, ["2026-11-26", "2026-11-26", "2026-11-26"]) == (False, 2)  # 2回まで


def test_status_file_shows_the_last_finished_day(cfg, tmp_path):
    con = db.connect(cfg["data"]["db_path"])
    con.executemany("INSERT INTO runs VALUES (?,?,?,?,?)",
                    [("2026-11-26", "", "", "ok", ""), ("2026-11-27", "", "", "running", "")])
    got = json.loads(site.write_status(con, tmp_path).read_text())
    assert got["data_date"] == "2026-11-26"  # 途中で止まった日は「済んだ」に数えない
