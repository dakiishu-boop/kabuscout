"""夜中にずれた起動でその日の分を取りこぼさないための仕組み（ジャーナル §100）。"""
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


class _Store:
    def load(self, **kw):
        return {"close": pd.DataFrame({"7203": [1.0], "9984": [float("nan")]}, index=pd.to_datetime(["2026-11-27"]))}


def _run(monkeypatch, cfg, usable, probe_ok):
    seq = iter(usable)
    monkeypatch.setattr(pipeline, "_usable_date", lambda store, cutoff: pd.Timestamp(next(seq)))
    calls = []
    ok = pipeline._refetch_if_incomplete(_Store(), ["7203"], cfg, pd.Timestamp("2026-11-27"),
                                         update=lambda *a: calls.append(1), probe=lambda c, w: (probe_ok, "確認"))
    return ok, len(calls)


def test_refetch_only_when_yahoo_has_the_day(monkeypatch, cfg):
    assert _run(monkeypatch, cfg, ["2026-11-27"], True) == (True, 0)                 # 揃っていれば取り直さない
    assert _run(monkeypatch, cfg, ["2026-11-26", "2026-11-27"], True) == (True, 1)   # 主力株に足がある → 1回取り直して揃った
    assert _run(monkeypatch, cfg, ["2026-11-26", "2026-11-26"], True) == (False, 1)  # 取り直しても揃わない → 次の起動へ
    assert _run(monkeypatch, cfg, ["2026-11-26"], False) == (False, 0)               # 今の Yahoo にも無い → 取り直さない


def test_status_file_shows_the_last_finished_day(cfg, tmp_path):
    con = db.connect(cfg["data"]["db_path"])
    con.executemany("INSERT INTO runs VALUES (?,?,?,?,?)",
                    [("2026-11-26", "", "", "ok", ""), ("2026-11-27", "", "", "running", "")])
    got = json.loads(site.write_status(con, tmp_path).read_text())
    assert got["data_date"] == "2026-11-26"  # 途中で止まった日は「済んだ」に数えない
    assert got["next_needed"] == "2026-11-27"
    con.execute("INSERT OR REPLACE INTO runs VALUES ('2026-11-20','','','ok','')")
    con.execute("DELETE FROM runs WHERE data_date>'2026-11-20'")
    assert json.loads(site.write_status(con, tmp_path).read_text())["next_needed"] == "2026-11-24"  # 土日と祝日（11/23）を飛ばす


def test_ties_are_ordered_by_code_whatever_the_input_order():
    from kabuscout import learner
    d = pd.DataFrame(dict(code=["9999", "1111", "5555", "2222"], s=[0.5, 0.5, 0.9, 0.5]))
    a = learner.order_by(d, "s")["code"].tolist()
    b = learner.order_by(d.iloc[::-1], "s")["code"].tolist()
    assert a == b == ["5555", "1111", "2222", "9999"]


def test_pdf_archive_saves_only_new_targets(cfg):
    from kabuscout import tdnet_archive as ta

    class R:
        def __init__(self, j=None, content=b""):
            self.j, self.content, self.status_code = j, content, 200

        def json(self):
            return self.j

    items = [dict(id="1", title="2026年3月期 第2四半期決算短信〔日本基準〕(連結)", pubdate="2026-11-10 15:00", document_url="https://x/1.pdf"),
             dict(id="2", title="業績予想の修正に関するお知らせ", pubdate="2026-11-10 15:00", document_url="https://x/2.pdf"),
             dict(id="3", title="自己株式の取得状況に関するお知らせ", pubdate="2026-11-10 15:00", document_url="https://x/3.pdf"),
             dict(id="4", title="（訂正）業績予想の修正に関するお知らせ", pubdate="2026-11-10 15:00", document_url="https://x/4.pdf")]
    asked = []

    def get(url, **kw):
        if url.endswith(".pdf"):
            asked.append(url)
            return R(content=b"%PDF-1.7 test")
        return R(j=dict(items=[{"Tdnet": i} for i in items]))

    got = ta.update(cfg, days=1, get=get, sleep=lambda s: None)
    assert (got["saved"], got["total"]) == (2, 2) and sorted(asked) == ["https://x/1.pdf", "https://x/2.pdf"]
    assert ta.path_of(cfg, items[0]).read_bytes().startswith(b"%PDF")
    asked.clear()
    assert ta.update(cfg, days=1, get=get, sleep=lambda s: None)["saved"] == 0 and asked == []  # 2回目は取りに行かない
