"""通知・Webの「AI勝率」の表示直し（ジャーナル §89）: 予想リターンの保存、旧DBの移行、実際の売買ルールでの答え合わせ、表示名。"""
import sqlite3

import numpy as np
import pandas as pd

from kabuscout import db, learner, pipeline, report

OLD_SIGNALS = """CREATE TABLE signals(
  date TEXT, code TEXT, name TEXT, strategies TEXT, rule_score REAL, prob REAL,
  final_score REAL, rank INTEGER, model_version TEXT, close REAL, atr REAL,
  tp REAL, sl REAL, reasons TEXT, earnings TEXT,
  label INTEGER, ret REAL, exit_days INTEGER, labeled INTEGER DEFAULT 0,
  PRIMARY KEY(date, code))"""


def test_old_db_gets_new_columns_and_keeps_rows(cfg, tmp_path):
    p = cfg["data"]["db_path"]
    p.parent.mkdir(parents=True, exist_ok=True)
    old = sqlite3.connect(str(p))
    old.execute(OLD_SIGNALS)
    old.execute("INSERT INTO signals(date, code, prob, labeled, label, ret) VALUES ('2026-11-02', '1111', 0.31, 1, 0, -0.01)")
    old.commit(); old.close()
    con = db.connect(p)
    cols = {r[1] for r in con.execute("PRAGMA table_info(signals)")}
    assert {"exp_ret", "label_plan", "ret_plan", "exit_plan", "labeled_plan"} <= cols
    row = con.execute("SELECT prob, label, ret, exp_ret, labeled_plan FROM signals").fetchone()
    assert row == (0.31, 0, -0.01, None, 0)
    db.connect(p)  # 2回目も壊れない


def _top(exp):
    return pd.DataFrame(dict(code=["1111", "2222"], name=["A", "B"], strategies=["earnings_drift", "squeeze"], rule_score=[0.5, 0.4],
                             prob=[0.33, np.nan], final_score=[0.9, 0.8], rank=[1, 2], model_version=["v", "v"], close=[1000.0, 500.0],
                             atr=[20.0, 10.0], tp=[1040.0, 520.0], sl=[980.0, 490.0], reasons=["r", "r"], earnings_days=[np.nan, 5],
                             exp_ret=exp))


def test_store_live_saves_exp_ret(cfg):
    con = db.connect(cfg["data"]["db_path"])
    pipeline._store_live(con, _top([0.012, np.nan]), "2026-11-24")
    got = dict(con.execute("SELECT code, exp_ret FROM signals").fetchall())
    assert got["1111"] == 0.012 and got["2222"] is None


class _Stub:
    def __init__(self, v):
        self.v = v

    def predict(self, X):
        return np.full(len(X), self.v)

    def predict_proba(self, X):
        return np.column_stack([np.full(len(X), 1 - self.v), np.full(len(X), self.v)])


def test_exp_ret_is_mean_of_ret_models_and_not_used_for_order():
    feats = ["f"]
    member = lambda r: dict(win=_Stub(0.3), ret=_Stub(r), rank=_Stub(0.0), features=feats)
    bundle = dict(models=dict(members=[member(0.01), member(0.03)], features=feats), version="v", alpha=1.0)
    df = pd.DataFrame(dict(date=pd.Timestamp("2026-11-24"), code=["1", "2"], f=[0.0, 1.0]))
    assert np.allclose(learner.bundle_exp_ret(bundle, df), 0.02)
    assert np.isnan(learner.bundle_exp_ret(dict(model=None, features=feats), df)).all()  # 旧モデルには無い


def test_label_live_fills_plan_results_only_when_settled(cfg, monkeypatch):
    con = db.connect(cfg["data"]["db_path"])
    pipeline._store_live(con, _top([0.01, 0.02]), "2026-11-02")
    ds = pd.DataFrame(dict(date=pd.Timestamp("2026-11-02"), code=["1111", "2222"], strategies=["earnings_drift", "squeeze"],
                           atr=[20.0, 10.0], ret=[0.04, -0.02], label=[1, 0], exit_days=[3, 2], complete=[True, True]))

    def fake(d, cfg_, store=None):  # 1111 は20日ルールでまだ決着していない、2222 は決着
        return d.assign(ret_plan=[0.0, -0.03], label_plan=[0.0, 0.0], exit_plan=[np.nan, 4.0], complete_plan=[False, True])
    monkeypatch.setattr(learner, "with_plan_outcomes", fake)
    pipeline._label_live(con, ds, cfg)
    got = {r[0]: r[1:] for r in con.execute("SELECT code, labeled, label, labeled_plan, label_plan, ret_plan, exit_plan FROM signals")}
    assert got["1111"] == (1, 1, 0, None, None, None)
    assert got["2222"] == (1, 0, 1, 0, -0.03, 4)


def test_answer_stats_uses_cost_and_pending(cfg):
    con = db.connect(cfg["data"]["db_path"])
    rows = [("2026-11-0%d" % i, "1%03d" % i, "earnings_drift", 0.35, 0.01, 1, 1 if i < 3 else 0, 1, 1 if i < 2 else 0, 0.10 if i < 2 else -0.05)
            for i in range(1, 5)]
    rows += [("2026-11-05", "9999", "earnings_drift", 0.15, None, 0, None, 0, None, None)]  # 未決着
    con.executemany("INSERT INTO signals(date, code, strategies, prob, exp_ret, labeled, label, labeled_plan, label_plan, ret_plan) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
    con.commit()
    a = report.answer_stats(con, cfg)
    s = a["strategies"][0]
    cost = cfg["research"]["cost"]
    assert (s["n"], s["pending"], s["ref"]) == (4, 1, True)
    assert abs(s["win"] - 0.25) < 1e-9 and abs(s["avg"] - ((0.10 - 0.05 * 3) / 4 - cost)) < 1e-9
    assert a["exp"]["n"] == 4 and abs(a["exp"]["actual"] - s["avg"]) < 1e-9
    band = [c for c in a["calib"] if c["lo"] == 0.3][0]
    assert band["n"] == 4 and abs(band["actual"] - 0.5) < 1e-9
    txt = "\n".join(report.answer_lines(a))
    assert "未決着1件" in txt and "参考値" in txt and f"コスト{cost:.1%}引き" in txt


def test_labels_are_named_by_what_they_measure(cfg):
    con = db.connect(cfg["data"]["db_path"])
    name = report.prob_name(cfg)
    assert name == f"+{cfg['labeling']['tp_atr']:g}ATR先着率({cfg['labeling']['horizon_days']}日)"
    pick = dict(rank=1, code="1111", name="A", strategies=["決算好反応"], prob=0.33, exp_ret=0.012, close=1000.0, tp=1270.0, sl=865.0,
                horizon=20, earnings_days=None, reasons="r", flows="")
    rep = dict(date="2026-11-24", date_label="11/24(火)", site_url="", market=dict(mood="弱気", mkt_dev75=-0.05, mkt_ret20=-0.03, breadth25=0.3),
               n_scanned=1, n_signals=1, picks=[pick], model=None, live=None, hot=[], paper=None, regime=None,
               prob_name=name, prob_note=report.prob_note(cfg), answers=report.answer_stats(con, cfg))
    disc = str(report.discord_payload(rep))
    md = report.to_markdown(rep)
    for t in (disc, md):
        assert "AI勝率" not in t and name in t and "予想リターン" in t and "+1.2%" in t
    assert disc.count("先着率＝共通ルール") == 1  # 注記は1回だけ
