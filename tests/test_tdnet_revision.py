"""業績予想の修正の題名を毎日保存する（§84）。AI の特徴量・戦略は変えない。"""
import json

import numpy as np
import pandas as pd

from kabuscout import events, tdnet


def test_revision_titles_are_kept():
    c = tdnet.classify
    assert c("業績予想の修正に関するお知らせ") == ["revision"]
    assert c("通期業績予想の上方修正に関するお知らせ") == ["forecast_up", "revision"]
    assert c("2026年3月期 業績予想と実績値との差異に関するお知らせ") == ["revision"]
    assert c("第2四半期累計期間の業績予想値と実績値との差異及び通期業績予想の修正に関するお知らせ") == ["revision"]
    # 「結果」を含む題名は他の分類から外れるが、修正の印は残す（研究の拾い方と同じ。除くのは訂正だけ）
    assert c("公開買付けの結果及び業績予想の修正に関するお知らせ") == ["revision"]
    assert c("（訂正）業績予想の修正に関するお知らせ") == []
    assert c("2026年3月期 決算短信〔日本基準〕（連結）") == ["earnings"]
    assert c("配当予想の修正（増配）に関するお知らせ") == ["dividend_up"]


def test_revision_does_not_change_model_features():
    assert "revision" not in events.EVENT_TYPES
    idx = pd.bdate_range("2026-01-05", periods=10)
    close = pd.DataFrame(1.0, index=idx, columns=["1001"])
    td = pd.DataFrame({"pubdate": [idx[2] + pd.Timedelta(hours=10)], "code": ["1001"], "title": ["業績予想の修正"], "events": ["revision"]})
    out = events.wide_features(td, close)
    assert set(out) == set(events.EVENT_FEATURES) and all(np.isnan(v.to_numpy()).all() for v in out.values())
    assert events.earnings_dates(td).empty


def _fake_api(monkeypatch, calls):
    class R:
        def __init__(self, url):
            self.day = url.split("/")[-1][:8]

        def raise_for_status(self):
            pass

        def json(self):
            if self.day != "20240510":
                return {"items": []}
            items = [dict(pubdate="2024-05-10 15:00:00", company_code="10010", title="2024年3月期 決算短信及び業績予想の修正に関するお知らせ"),
                     dict(pubdate="2024-05-10 15:30:00", company_code="20020", title="業績予想と実績値との差異に関するお知らせ"),
                     dict(pubdate="2024-05-10 16:00:00", company_code="30030", title="自己株式の取得に係る事項の決定に関するお知らせ")]
            return {"items": [{"Tdnet": t} for t in items]}

    monkeypatch.setattr(tdnet.requests, "get", lambda url, **kw: calls.append(url) or R(url))
    monkeypatch.setattr(tdnet.time, "sleep", lambda s: None)


def test_backfill_only_adds_revision_marks(tmp_path, monkeypatch):
    """過去分の取り直しは revision の印を足すだけ（他の分類・保存済みの行は変えない＝本番の特徴量は変わらない）。"""
    calls = []
    _fake_api(monkeypatch, calls)
    cfg = {"data": {"db_path": tmp_path / "kabu.db"}}
    old = pd.DataFrame({"pubdate": pd.to_datetime(["2024-05-10 15:00", "2024-05-10 16:00"]), "code": ["1001", "3003"],
                        "title": ["2024年3月期 決算短信及び業績予想の修正に関するお知らせ", "自己株式の取得に係る事項の決定に関するお知らせ"],
                        "events": ["earnings", "buyback"]})
    old.to_parquet(tmp_path / "tdnet.parquet")
    assert not tdnet.revisions_ready(cfg, start="2024-05-01")
    n = tdnet.backfill_revisions(cfg, start="2024-05-01", today=pd.Timestamp("2024-05-20"))
    assert n == 2 and len(calls) == 20
    td = tdnet.load(cfg).set_index("code")["events"].to_dict()
    assert td == {"1001": "earnings,revision", "3003": "buyback", "2002": "revision"}
    assert tdnet.revisions_ready(cfg, start="2024-05-01") and tdnet.backfill_progress(cfg, start="2024-05-01") == (20, 20)
    calls.clear()
    assert tdnet.backfill_revisions(cfg, start="2024-05-01", today=pd.Timestamp("2024-06-30")) == 0 and not calls  # 最後の日は初回のまま
    assert len(tdnet.load(cfg)) == 3


def test_backfill_resumes_after_timeout(tmp_path, monkeypatch):
    calls = []
    _fake_api(monkeypatch, calls)
    cfg = {"data": {"db_path": tmp_path / "kabu.db"}}
    clock = iter(range(0, 10_000, 30))  # 1日ごとに30秒進む時計
    monkeypatch.setattr(tdnet.time, "time", lambda: next(clock))
    tdnet.backfill_revisions(cfg, start="2024-05-01", max_minutes=5, today=pd.Timestamp("2024-05-20"))
    assert 0 < tdnet.backfill_progress(cfg, start="2024-05-01")[0] < 20 and not tdnet.revisions_ready(cfg, start="2024-05-01")
    tdnet.backfill_revisions(cfg, start="2024-05-01", today=pd.Timestamp("2024-06-30"))
    assert tdnet.revisions_ready(cfg, start="2024-05-01")
