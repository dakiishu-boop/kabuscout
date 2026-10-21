"""ペーパートレード（仮想売買）。

AI の推奨どおりに機械的に売買したらどうなるかを毎日記録する。
ルール: シグナル翌営業日の寄付きで買い → 利確(+tp_atr×ATR) / 損切(-sl_atr×ATR) / horizon 日で手仕舞い。
自動売買に進む前に、ここで数ヶ月分の実績を確認するのが目的。
"""
from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def _simulate(ind: dict, code: str, entry_date: pd.Timestamp, tp: float, sl: float, H: int):
    """entry_date から最新日まで1本ずつ進めて、決済されたら (日付, 価格, 理由, 保有日数) を返す。"""
    idx = ind["close"].index
    start = idx.searchsorted(entry_date)
    for k, i in enumerate(range(start, len(idx)), start=1):
        o, h, l, c = (float(ind[f][code].iat[i]) for f in ("open", "high", "low", "close"))
        if l <= sl:
            return idx[i], (min(o, sl) if k > 1 else sl), "損切", k
        if h >= tp:
            return idx[i], (max(o, tp) if k > 1 else tp), "利確", k
        if k >= H:
            return idx[i], c, "期限", k
    return None, None, None, len(idx) - start


def step(con, cfg: dict, ind: dict, picks: pd.DataFrame, data_date: pd.Timestamp) -> dict:
    pc = cfg["paper_trading"]
    H = cfg["labeling"]["horizon_days"]
    k_tp, k_sl = cfg["labeling"]["tp_atr"], cfg["labeling"]["sl_atr"]
    idx = ind["close"].index
    pos = pd.read_sql("SELECT * FROM paper_positions", con)
    realized = pos.loc[pos["status"] == "closed", "pnl"].sum() if not pos.empty else 0.0
    equity = pc["capital_jpy"] + realized
    slot = equity / pc["max_positions"]
    events = []

    # 1) 約定待ち → 翌営業日の寄付きで約定
    for _, p in pos[pos["status"] == "pending"].iterrows():
        if p["code"] not in ind["close"].columns:
            continue
        sd = pd.Timestamp(p["signal_date"])
        j = idx.searchsorted(sd, side="right")
        if j >= len(idx):
            continue
        ed = idx[j]
        entry = float(ind["open"][p["code"]].iat[j])
        lot = pc["lot_size"]
        shares = int(math.floor(slot / (entry * lot))) * lot if entry > 0 else 0
        if shares == 0 or not np.isfinite(entry):
            con.execute("UPDATE paper_positions SET status='cancelled', reason='資金不足' WHERE id=?", (p["id"],))
            continue
        tp, sl = entry + k_tp * p["atr"], entry - k_sl * p["atr"]
        con.execute("UPDATE paper_positions SET status='open', entry_date=?, entry=?, shares=?, tp=?, sl=? WHERE id=?",
                    (ed.strftime("%Y-%m-%d"), entry, shares, tp, sl, p["id"]))
        events.append(f"買 {p['code']} {p['name']} {shares}株 @{entry:,.0f}")

    # 2) 保有中 → 利確/損切/期限の判定（毎回エントリー日から再生するので実行漏れがあっても正しい）
    pos = pd.read_sql("SELECT * FROM paper_positions WHERE status='open'", con)
    for _, p in pos.iterrows():
        if p["code"] not in ind["close"].columns:  # 上場廃止・データ欠落
            continue
        ex_date, ex_px, why, held = _simulate(ind, p["code"], pd.Timestamp(p["entry_date"]), p["tp"], p["sl"], H)
        if ex_date is None:
            con.execute("UPDATE paper_positions SET days_held=? WHERE id=?", (held, p["id"]))
            continue
        pnl = (ex_px - p["entry"]) * p["shares"]
        con.execute("UPDATE paper_positions SET status='closed', exit_date=?, exit=?, pnl=?, reason=?, days_held=? WHERE id=?",
                    (ex_date.strftime("%Y-%m-%d"), ex_px, pnl, why, held, p["id"]))
        events.append(f"{why} {p['code']} {p['name']} {pnl:+,.0f}円")

    # 3) 新規の注文（翌営業日寄付きで約定予定）
    held = pd.read_sql("SELECT code FROM paper_positions WHERE status IN ('open','pending')", con)["code"].tolist()
    free = pc["max_positions"] - len(held)
    if free > 0 and len(picks):
        cand = picks[~picks["code"].isin(held)]
        if pc.get("min_prob") and cand["prob"].notna().any():
            cand = cand[cand["prob"] >= pc["min_prob"]]
        cand = cand[cand["earnings_days"].isna() | (cand["earnings_days"] > H * 1.5)]  # 決算またぎは避ける
        for _, r in cand.head(min(free, pc["picks_per_day"])).iterrows():
            con.execute("INSERT INTO paper_positions(code,name,signal_date,status,atr,prob) VALUES (?,?,?,?,?,?)",
                        (r["code"], r["name"], data_date.strftime("%Y-%m-%d"), "pending", float(r["atr"]),
                         None if pd.isna(r["prob"]) else float(r["prob"])))
            events.append(f"注文 {r['code']} {r['name']} (翌寄付)")
    con.commit()
    return summary(con, cfg, ind) | {"events": events}


def summary(con, cfg: dict, ind: dict | None = None) -> dict:
    pos = pd.read_sql("SELECT * FROM paper_positions", con)
    closed = pos[pos["status"] == "closed"]
    opened = pos[pos["status"] == "open"]
    unreal = 0.0
    if ind is not None:
        for _, p in opened.iterrows():
            if p["code"] not in ind["close"].columns:
                continue
            last = ind["close"][p["code"]].dropna()
            if len(last):
                unreal += (float(last.iat[-1]) - p["entry"]) * p["shares"]
    realized = float(closed["pnl"].sum()) if len(closed) else 0.0
    return dict(
        open=int(len(opened)), pending=int((pos["status"] == "pending").sum()), closed=int(len(closed)),
        win_rate=float((closed["pnl"] > 0).mean()) if len(closed) else None,
        realized=realized, unrealized=unreal,
        equity=cfg["paper_trading"]["capital_jpy"] + realized + unreal,
    )
