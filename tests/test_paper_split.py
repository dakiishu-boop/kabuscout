"""保有中の分割・併合: 記録の買値・利確・損切は買った時の株価のままなので、今のデータの単位に直してから判定する。
Yahoo の株価は分割が起きると過去にさかのぼって調整される（2026-10-29 の3銘柄は権利落ち日のうちに調整済み）。"""
import numpy as np
import pandas as pd
import pytest

from kabuscout import db, paper
from kabuscout.core import ai_capital

idx = pd.bdate_range("2026-03-02", periods=12)
EX = 6                # idx[6] が権利落ち日
CODE = "9999"


def bars(ratio=1.0, raw=False):
    """じわじわ上がる銘柄（分割がなければ利確にも損切にも届かない）。ratio=2 は1→2分割、0.2 は5→1併合。
    raw=True: 権利落ち日以降だけ新しい株価で、過去はまだ調整されていない。"""
    c = pd.Series(1500.0 + 4 * np.arange(len(idx)), index=idx)
    f = pd.Series(1 / ratio, index=idx)
    if raw:
        f[idx < idx[EX]] = 1.0
    return {k: pd.DataFrame({CODE: v * f}) for k, v in {"open": c - 2, "high": c + 8, "low": c - 8, "close": c}.items()}


def play(cfg, days, acct="paper", first=0):
    """days: 各日のデータ（iloc[:i+1] まで見える）を返す関数。本番と同じく1日ずつ step を回す。"""
    con = db.connect(cfg["data"]["db_path"])
    cfg["labeling"]["stop_mode"] = "intraday"
    sd = idx[first].strftime("%Y-%m-%d")
    con.execute(f"INSERT INTO {acct}_positions(signal_date,code,name,atr,tp_atr,sl_atr,horizon,capital,status) "
                "VALUES (?,?,'テスト',30.0,2.0,1.0,10,?,'pending')", (sd, CODE, ai_capital(cfg)))
    con.execute("INSERT INTO signals(date,code,close) VALUES (?,?,?)", (sd, CODE, 1500.0 + 4 * first))
    con.commit()
    out = None
    for i in range(first, len(idx)):
        ind = {k: v.iloc[: i + 1] for k, v in days(i).items()}
        out = paper.step(con, cfg, ind, pd.DataFrame(), idx[i], acct=acct)
    return con, out, pd.read_sql(f"SELECT * FROM {acct}_positions", con).iloc[0]


def same_day(ratio):
    return lambda i: bars(ratio) if i >= EX else bars()


def next_day(ratio):
    return lambda i: bars(ratio) if i > EX else bars(ratio, raw=True) if i == EX else bars()


def test_no_split_is_unchanged(cfg):
    _, out, p = play(cfg, lambda i: bars())
    assert (p["reason"], p["entry"], p["exit"]) == ("期限", 1502.0, 1540.0)
    assert out["equity"] == pytest.approx(1_003_496, abs=1)


@pytest.mark.parametrize("ratio,where", [(2.0, "same"), (2.0, "next"), (0.2, "same"), (0.2, "next")])
def test_split_during_holding_matches_no_split(cfg, ratio, where):
    # 分割しても持っている株の価値は変わらないので、分割がなかった場合と同じ決済・同じ損益になる
    _, out, p = play(cfg, same_day(ratio) if where == "same" else next_day(ratio))
    assert p["reason"] == "期限" and p["exit_date"] == idx[10].strftime("%Y-%m-%d")
    assert p["exit"] == pytest.approx(1540.0)
    assert p["pnl"] == pytest.approx(3_496, abs=1)
    assert out["equity"] == pytest.approx(1_003_496, abs=1)


def test_real_stop_after_split_still_fires(cfg):
    def days(i):
        b = bars(2.0) if i >= EX else bars()
        if i >= EX + 1:                        # 分割後に本当に下がる（調整後の 700 = 分割前の単位で 1,400 → 損切 1,472 を割る）
            b["low"].iloc[EX + 1] = 700.0
        return b
    _, _, p = play(cfg, days)
    assert p["reason"] == "損切" and p["exit_date"] == idx[EX + 1].strftime("%Y-%m-%d")
    assert p["exit"] < 1472                    # 買った時の単位で記録する
    assert p["pnl"] == pytest.approx((p["exit"] * 0.999 - 1502 * 1.001) * 100, rel=0.01)


def test_big_gap_waits_one_day_then_is_judged(cfg):
    # 調整の遅れではなく本当の急落なら、翌日の実行で判定される（その日の日付で）
    def days(i):
        b = bars()
        if i >= EX:
            for f in ("open", "high", "low", "close"):
                b[f].iloc[EX:] = b[f].iloc[EX:] * 0.6
        return b
    con, _, p = play(cfg, days)
    assert p["reason"] == "損切" and p["exit_date"] == idx[EX].strftime("%Y-%m-%d")


def test_split_between_order_and_fill_scales_the_stop(cfg):
    # 注文の翌朝が権利落ち日: ATR は注文した日の単位なので、新しい単位に直してから利確・損切を決める
    _, _, p = play(cfg, same_day(2.0), first=EX - 1)
    assert p["entry"] == pytest.approx(759.0 + 2)    # 寄付（調整後）
    assert (p["tp"] - p["entry"]) / p["entry"] == pytest.approx(2 * 15 / p["entry"])
    assert (p["entry"] - p["sl"]) / p["entry"] == pytest.approx(15 / p["entry"])


def test_swap_sale_after_split(cfg):
    # 比較用の口座: 入れ替えで売る翌朝が権利落ち日でも、半値で売った記録にならない
    con, _, p = play(cfg, lambda i: bars(), acct="paper_swap")
    con.execute("UPDATE paper_swap_positions SET status='selling', exit_date=NULL, exit=NULL, pnl=NULL, reason=NULL, "
                "sell_date=?", (idx[EX - 1].strftime("%Y-%m-%d"),))
    ind = {k: v.iloc[: EX + 1] for k, v in bars(2.0).items()}
    paper._sell_swapped(con, cfg, ind, "paper_swap_positions")
    p = pd.read_sql("SELECT * FROM paper_swap_positions", con).iloc[0]
    assert p["exit"] == pytest.approx(1522.0)
    assert p["pnl"] > 0
