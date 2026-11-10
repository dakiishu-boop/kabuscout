"""比較用の口座 paper_conf: 3つのモデルの合意が高い候補を多めに買う（ジャーナル 82・83）。
倍率は「1銘柄の上限額」と「損切時の損失の上限」の両方に掛け、注文した時に決めて翌朝の約定でも同じ倍率を使う。
本番の口座（sizing なし）は、合意の値があっても今までどおり。"""
import numpy as np
import pandas as pd

from kabuscout import backtest, db, paper, shadow
from kabuscout.risk import size_mult, sized
from tests.conftest import FakeStore
from tests.test_account import _cfg, _preds

TIERS = [[0.9, 1.25], [0.95, 1.5]]
idx = pd.bdate_range("2026-03-02", periods=3)


def test_size_mult_picks_the_highest_tier_reached():
    assert [size_mult(TIERS, a) for a in (0.5, 0.89, 0.9, 0.94, 0.95, 1.0)] == [1.0, 1.0, 1.25, 1.25, 1.5, 1.5]
    assert size_mult(list(reversed(TIERS)), 0.97) == 1.5       # 段の並び順によらない
    assert size_mult(TIERS, float("nan")) == 1.0 and size_mult(TIERS, None) == 1.0   # 合意が分からなければ本番と同じ
    assert size_mult(None, 0.99) == 1.0 and size_mult([], 0.99) == 1.0               # 設定が無ければ本番と同じ


def test_sized_scales_only_the_loss_cap():
    acc = dict(lot_size=100, risk_per_trade=0.02)
    assert sized(acc, 1.0) is acc
    assert sized(acc, 1.5) == dict(lot_size=100, risk_per_trade=0.03) and acc["risk_per_trade"] == 0.02


def test_registered_tiers_are_the_ones_tested():
    # 事前登録した倍率（合意 0.95 以上だけ 2倍。ジャーナル 83）を、結果を見てから変えないための見張り
    assert shadow.account("paper_conf")["size_by_agreement"] == [[0.95, 2.0]]
    assert shadow.account("no_such_account") == {}


def ind_for(prices):
    c = pd.DataFrame({k: np.full(len(idx), v) for k, v in prices.items()}, index=idx)
    return {"open": c, "high": c * 1.01, "low": c * 0.99, "close": c, "mkt_dev75": pd.DataFrame({"m": np.full(len(idx), 0.05)}, index=idx)}


def pick(code, agreement):
    # 500円・損切まで 3 × 25 = 75円。損失の上限 2万円 → 266株 → 200株。1.5倍なら 3万円 → 400株（上限額 30万円の内）
    return dict(code=code, name="テスト", prob=0.5, close=500.0, atr=25.0, tp_atr=6.0, sl_atr=3.0, horizon=20,
                agreement=agreement, final_score=1.0, strategies="earnings_drift")


def run(cfg, acct, sizing, picks):
    con = db.connect(cfg["data"]["db_path"])
    ind = ind_for({p["code"]: 500.0 for p in picks})
    first = paper.step(con, cfg, {k: v.iloc[:1] for k, v in ind.items()}, pd.DataFrame(picks), idx[0], acct=acct, sizing=sizing)
    paper.step(con, cfg, {k: v.iloc[:2] for k, v in ind.items()}, pd.DataFrame(), idx[1], acct=acct, sizing=sizing)
    return first["events"], pd.read_sql(f"SELECT * FROM {acct}_positions ORDER BY id", con)


def test_high_agreement_pick_is_bought_larger_in_the_side_account(cfg):
    ev, pos = run(cfg, "paper_conf", TIERS, [pick("7777", 0.97), pick("8888", 0.5)])
    assert ev == ["注文 7777 テスト (翌寄付・1.5倍)", "注文 8888 テスト (翌寄付)"]
    assert pos["size"].tolist() == [1.5, 1.0] and pos["status"].tolist() == ["open", "open"]
    assert pos["shares"].tolist() == [400, 200]


def test_middle_tier_is_capped_by_the_scaled_loss_cap(cfg):
    _, pos = run(cfg, "paper_conf", TIERS, [pick("7777", 0.92)])
    assert pos["size"].tolist() == [1.25] and pos["shares"].tolist() == [300]   # 2.5万円 ÷ 75円 = 333株 → 300株


def test_production_account_ignores_agreement(cfg):
    ev, pos = run(cfg, "paper", None, [pick("7777", 0.97), pick("8888", 0.5)])
    assert ev == ["注文 7777 テスト (翌寄付)", "注文 8888 テスト (翌寄付)"]
    assert pos["shares"].tolist() == [200, 200] and "size" not in pos.columns


def test_side_account_starts_from_production_and_keeps_old_orders_at_1x(cfg):
    con = db.connect(cfg["data"]["db_path"])
    ind = ind_for({"7777": 500.0})
    paper.step(con, cfg, {k: v.iloc[:1] for k, v in ind.items()}, pd.DataFrame([pick("7777", 0.97)]), idx[0])
    assert paper.start_from(con, "paper", "paper_conf") == ["比較用の口座を、本番の口座の今の状態から始めました"]
    paper.step(con, cfg, {k: v.iloc[:2] for k, v in ind.items()}, pd.DataFrame(), idx[1], acct="paper_conf", sizing=TIERS)
    pos = pd.read_sql("SELECT * FROM paper_conf_positions", con)
    assert pos["shares"].tolist() == [200] and pos["size"].isna().all()   # 本番で注文済みの分は、合意が高くても1倍のまま約定


def test_backtest_setting_equals_research_size_column(cfg):
    # 過去の検証（backtest.simulate の口座設定 size_by_agreement）と、研究で使った _size 列が同じ結果になる
    cal = pd.bdate_range("2026-03-02", periods=8)
    close = pd.DataFrame({"A": 500.0, "B": 500.0}, index=cal)
    rows = [dict(day=0, code="A", close=500.0, _stop=75.0, prob=0.9, ret=0.10, exit_days=3, agreement=0.97),
            dict(day=0, code="B", close=500.0, _stop=75.0, prob=0.8, ret=-0.05, exit_days=3, agreement=0.5)]
    c = _cfg(cfg)
    kw = dict(pick="ai", gate=None, store=FakeStore(close))
    base = backtest.simulate(_preds(rows, cal), cal, c, **kw)
    tier = backtest.simulate(_preds(rows, cal), cal, c, account={"size_by_agreement": TIERS}, **kw)
    col = backtest.simulate(_preds(rows, cal).assign(_size=[1.5, 1.0]), cal, c, **kw)
    assert tier["equity"].equals(col["equity"])
    assert sorted(tier["trades"]["pnl"].round().tolist()) != sorted(base["trades"]["pnl"].round().tolist())
    # A: 400株（1.5倍）× 500円 × +10% ≒ +2万円、今の設定なら 200株で ≒ +1万円
    assert tier["trades"]["pnl"].max() / base["trades"]["pnl"].max() > 1.9


def test_registered_rule_doubles_only_the_top_tier(cfg):
    # 登録した買い方（合意 0.95 以上だけ 2倍）: 損失の上限 4万円 ÷ 75円 = 533株 → 500株（上限額 40万円の内）。0.92 は1倍のまま
    tiers = shadow.account("paper_conf")["size_by_agreement"]
    ev, pos = run(cfg, "paper_conf", tiers, [pick("7777", 0.97), pick("8888", 0.92)])
    assert ev == ["注文 7777 テスト (翌寄付・2倍)", "注文 8888 テスト (翌寄付)"]
    assert pos["size"].tolist() == [2.0, 1.0] and pos["shares"].tolist() == [500, 200]
