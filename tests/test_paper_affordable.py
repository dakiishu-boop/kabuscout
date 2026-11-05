"""100株で買えない候補は注文せず、次の候補を買う（過去の検証 backtest.simulate と同じ）。
以前の本番は上位から注文して翌朝「資金不足」で取り消し、その日の枠が空いたままになっていた。"""
import numpy as np
import pandas as pd

from kabuscout import db, paper

idx = pd.bdate_range("2026-03-02", periods=3)


def ind_for(prices):
    c = pd.DataFrame({k: np.full(len(idx), v) for k, v in prices.items()}, index=idx)
    return {"open": c, "high": c * 1.01, "low": c * 0.99, "close": c}


def pick(code, close, prob):
    return dict(code=code, name="テスト", prob=prob, close=close, atr=close * 0.02, tp_atr=2.0, sl_atr=1.0, horizon=10)


def test_unaffordable_top_pick_is_skipped(cfg):
    cfg["paper_trading"]["picks_per_day"] = 1
    con = db.connect(cfg["data"]["db_path"])
    # 1銘柄の上限は 100万円 ÷ 5 = 20万円。5,000円 × 100株 = 50万円は買えない
    picks = pd.DataFrame([pick("8888", 5000.0, 0.5), pick("9999", 1500.0, 0.4)])
    out = paper.step(con, cfg, ind_for({"8888": 5000.0, "9999": 1500.0}), picks, idx[0])
    pos = pd.read_sql("SELECT code, status FROM paper_positions", con)
    assert pos["code"].tolist() == ["9999"] and pos["status"].tolist() == ["pending"]
    assert out["events"] == ["注文 9999 テスト (翌寄付)"]


def test_affordable_top_pick_is_still_first(cfg):
    con = db.connect(cfg["data"]["db_path"])
    picks = pd.DataFrame([pick("7777", 1200.0, 0.6), pick("8888", 5000.0, 0.5), pick("9999", 1500.0, 0.4)])
    paper.step(con, cfg, ind_for({"7777": 1200.0, "8888": 5000.0, "9999": 1500.0}), picks, idx[0])
    assert pd.read_sql("SELECT code FROM paper_positions ORDER BY id", con)["code"].tolist() == ["7777", "9999"]
