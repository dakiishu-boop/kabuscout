"""影の候補（shadow.py）: 判定日の前は「記録中」、判定日の後は事前登録の基準どおりに合否が出る。"""
import pandas as pd
import yaml

from kabuscout import shadow

PAST, FUTURE = "2000-01-01", "2999-12-31"


def _v(source, judge, **rule):
    base = dict(min_weeks=20, t_min=2.0) if source == "registered" else dict(min_weeks=8, t_floor=-1.0)
    return dict(key="k", label="k", source=source, judge=judge, rule={**base, **rule})


def test_before_judge_date_is_recording():
    assert shadow.verdict(_v("registered", FUTURE), dict(weeks=100, uplift_t=9.0)) == "記録中"
    assert shadow.verdict(_v("research", FUTURE), dict(weeks=100, uplift_t=-9.0)) == "記録中"


def test_registered_candidates_need_t_of_two():
    v = _v("registered", PAST)
    assert shadow.verdict(v, dict(weeks=25, uplift_t=2.0)).startswith("合格")
    assert shadow.verdict(v, dict(weeks=25, uplift_t=1.99)) == "不合格"
    assert shadow.verdict(v, dict(weeks=25, uplift_t=None)) == "不合格"
    assert shadow.verdict(v, dict(weeks=19, uplift_t=5.0)).startswith("判定日を過ぎたが")


def test_research_candidates_are_promoted_unless_clearly_worse():
    v = _v("research", PAST)
    assert shadow.verdict(v, dict(weeks=12, uplift_t=-0.99)) == "昇格"
    assert shadow.verdict(v, dict(weeks=12, uplift_t=None)) == "昇格"
    assert shadow.verdict(v, dict(weeks=12, uplift_t=-1.0)).startswith("見送り")
    assert shadow.verdict(v, dict(weeks=7, uplift_t=3.0)).startswith("判定日を過ぎたが")


def test_registry_is_read_with_dates_as_text(cfg, tmp_path, monkeypatch):
    reg = dict(registered=pd.Timestamp("2026-10-28").date(), start=pd.Timestamp("2026-10-29").date(),
               judge=dict(date=pd.Timestamp("2027-06-30").date(), min_weeks=20, t_min=2.0),
               variants=[dict(key="leaf31", label="葉31", model_params=dict(max_leaf_nodes=31))],
               research_rule=dict(weeks=12, min_weeks=8, t_floor=-1.0))
    p = tmp_path / "shadow.yaml"
    p.write_text(yaml.safe_dump(reg, allow_unicode=True))
    monkeypatch.setattr(shadow, "SHADOW_FILE", p)
    (v,) = shadow.variants(cfg)
    assert v["judge"] == "2027-06-30" and v["start"] == "2026-10-29" and v["rule"]["date"] == "2027-06-30"
    assert v["changes"] == {"model_params": {"max_leaf_nodes": 31}}
    c = shadow.cfg_for(cfg, v)
    assert c["learning"]["model_params"] == {"max_leaf_nodes": 31}
    assert cfg["learning"].get("model_params") != {"max_leaf_nodes": 31}   # 元の設定は変えない
