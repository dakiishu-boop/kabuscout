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


def test_new_rule_refetches_last_400_days_once(tmp_path, monkeypatch):
    today = pd.Timestamp.today().normalize()
    start = today - pd.Timedelta(days=410)
    (tmp_path / "tdnet.json").write_text(json.dumps([d.strftime("%Y%m%d") for d in pd.date_range(start, today - pd.Timedelta(days=4))]))
    (tmp_path / "tdnet_rescan_tob_v1").touch()  # 前の取り直しは済んでいる
    calls = []

    class R:
        def raise_for_status(self):
            pass

        def json(self):
            return {"items": [{"Tdnet": {"pubdate": "2026-11-13 15:30:00", "company_code": "10010", "title": "業績予想の修正に関するお知らせ"}}]}

    monkeypatch.setattr(tdnet.requests, "get", lambda url, **kw: calls.append(url) or R())
    monkeypatch.setattr(tdnet.time, "sleep", lambda s: None)
    cfg = {"data": {"db_path": tmp_path / "kabu.db"}}
    tdnet.update(cfg, start=start.strftime("%Y-%m-%d"))
    assert (tmp_path / "tdnet_rescan_rev_v1").exists()
    assert 395 <= len(calls) <= 405
    assert set(tdnet.load(cfg)["events"]) == {"revision"}
    calls.clear()
    tdnet.update(cfg, start=start.strftime("%Y-%m-%d"))
    assert len(calls) <= 5  # 2回目からは取り直さない（直近3日と今日だけ）
