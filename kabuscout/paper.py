"""ペーパートレード（仮想売買）。

AI の推奨どおりに機械的に売買したらどうなるかを毎日記録する。
ルール: シグナル翌営業日の寄付きで買い → 利確(+tp_atr×ATR) / 損切(-sl_atr×ATR) / horizon 日で手仕舞い。
資金・株数の決め方は実用段階（資金100万円・現物）と同じ（risk.py）。損益は売買コスト（research.cost）を引いた値。
自動売買に進む前に、ここで数ヶ月分の実績を確認するのが目的。
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .risk import order_shares

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


def step(con, cfg: dict, ind: dict, picks: pd.DataFrame, data_date: pd.Timestamp, crisis: bool = False) -> dict:
    pc = cfg["paper_trading"]
    H = cfg["labeling"]["horizon_days"]
    k_tp, k_sl = cfg["labeling"]["tp_atr"], cfg["labeling"]["sl_atr"]
    half_cost = cfg["research"]["cost"] / 2
    idx = ind["close"].index
    events = _restart_if_capital_changed(con, pc["capital_jpy"])
    st = summary(con, cfg, ind)
    equity, cash = st["equity"], st["cash"]
    slot = equity / pc["max_positions"]
    pos = pd.read_sql("SELECT * FROM paper_positions", con)

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
        kt = p["tp_atr"] if pd.notna(p.get("tp_atr")) else k_tp
        ks = p["sl_atr"] if pd.notna(p.get("sl_atr")) else k_sl
        shares = order_shares(pc, equity, cash, entry * (1 + half_cost), ks * p["atr"], slot) if np.isfinite(entry) else 0
        if shares == 0:
            con.execute("UPDATE paper_positions SET status='cancelled', reason='資金不足' WHERE id=?", (p["id"],))
            continue
        cash -= shares * entry * (1 + half_cost)
        hz = int(p["horizon"]) if pd.notna(p.get("horizon")) else H
        tp, sl = entry + kt * p["atr"], entry - ks * p["atr"]
        con.execute("UPDATE paper_positions SET status='open', entry_date=?, entry=?, shares=?, tp=?, sl=?, horizon=? WHERE id=?",
                    (ed.strftime("%Y-%m-%d"), entry, shares, tp, sl, hz, p["id"]))
        events.append(f"買 {p['code']} {p['name']} {shares}株 @{entry:,.0f}")

    # 2) 保有中 → 利確/損切/期限の判定（毎回エントリー日から再生するので実行漏れがあっても正しい）
    pos = pd.read_sql("SELECT * FROM paper_positions WHERE status='open'", con)
    for _, p in pos.iterrows():
        if p["code"] not in ind["close"].columns:  # 上場廃止・データ欠落
            continue
        h = int(p["horizon"]) if pd.notna(p.get("horizon")) else H
        ex_date, ex_px, why, held = _simulate(ind, p["code"], pd.Timestamp(p["entry_date"]), p["tp"], p["sl"], h)
        if ex_date is None:
            con.execute("UPDATE paper_positions SET days_held=? WHERE id=?", (held, p["id"]))
            continue
        pnl = (ex_px * (1 - half_cost) - p["entry"] * (1 + half_cost)) * p["shares"]
        con.execute("UPDATE paper_positions SET status='closed', exit_date=?, exit=?, pnl=?, reason=?, days_held=? WHERE id=?",
                    (ex_date.strftime("%Y-%m-%d"), ex_px, pnl, why, held, p["id"]))
        events.append(f"{why} {p['code']} {p['name']} {pnl:+,.0f}円")

    # 3) 新規の注文（翌営業日寄付きで約定予定）
    held = pd.read_sql("SELECT code FROM paper_positions WHERE status IN ('open','pending')", con)["code"].tolist()
    free = pc["max_positions"] - len(held)
    already = con.execute("SELECT COUNT(*) FROM paper_positions WHERE signal_date=? AND status!='archived'",
                          (data_date.strftime("%Y-%m-%d"),)).fetchone()[0]
    if free > 0 and len(picks) and not already:  # 同じ日の再実行で二重注文しない
        cand = picks[~picks["code"].isin(held)]
        if pc.get("min_prob") and cand["prob"].notna().any():
            cand = cand[cand["prob"] >= pc["min_prob"]]
        # 決算をまたいで持つと結果が読めなくなるので避ける（決算好反応のように発表後に買う戦略は次の決算まで余裕がある）
        hz = cand["horizon"] if "horizon" in cand else H
        cand = cand[cand["earnings_days"].isna() | (cand["earnings_days"] > hz * 1.5)]
        n_new = min(free, pc["picks_per_day"])
        if crisis:  # 特例（危機）局面では新規の買いを1銘柄に絞る
            n_new = min(n_new, 1)
        from .backtest import gate_active
        gate = pc.get("gate")
        if gate and gate_active(gate, float(ind["mkt_dev75"].iloc[-1].iloc[0]), float(ind["mkt_ret20"].iloc[-1].iloc[0]), crisis):
            n_new = min(n_new, gate["max_new"])
            events.append(f"地合いが弱いため新規の買いを控えています（{gate['name']}）")
        for _, r in cand.head(n_new).iterrows():
            con.execute("INSERT INTO paper_positions(code,name,signal_date,status,atr,prob,tp_atr,sl_atr,horizon,capital) VALUES (?,?,?,?,?,?,?,?,?,?)",
                        (r["code"], r["name"], data_date.strftime("%Y-%m-%d"), "pending", float(r["atr"]),
                         None if pd.isna(r["prob"]) else float(r["prob"]), float(r.get("tp_atr", k_tp)),
                         float(r.get("sl_atr", k_sl)), int(r.get("horizon", H)), pc["capital_jpy"]))
            events.append(f"注文 {r['code']} {r['name']} (翌寄付)")
    con.commit()
    st = summary(con, cfg, ind)
    con.execute("INSERT OR REPLACE INTO paper_equity VALUES (?,?,?)",
                (data_date.strftime("%Y-%m-%d"), st["equity"], pc["capital_jpy"]))
    con.commit()
    return summary(con, cfg, ind) | {"events": events}


def _restart_if_capital_changed(con, capital: float) -> list[str]:
    """資金の設定が変わったら、それまでの記録を「保管」にして新しい資金で始め直す（削除はしない）。"""
    n = con.execute("SELECT COUNT(*) FROM paper_positions WHERE status!='archived' AND (capital IS NULL OR capital!=?)",
                    (capital,)).fetchone()[0]
    if not n:
        return []
    con.execute("UPDATE paper_positions SET status='archived', reason=COALESCE(reason || ' / ', '') || '資金設定の変更で記録を区切り' "
                "WHERE status!='archived' AND (capital IS NULL OR capital!=?)", (capital,))
    con.commit()
    log.info("資金設定が %s円 に変わったため、ペーパートレードを始め直します（以前の %d件は保管）", f"{capital:,.0f}", n)
    return [f"資金を{capital:,.0f}円に変えたため、ペーパートレードを新しく始めました（以前の記録は保管しています）"]


def summary(con, cfg: dict, ind: dict | None = None) -> dict:
    pc = cfg["paper_trading"]
    cap, half_cost = pc["capital_jpy"], cfg["research"]["cost"] / 2
    pos = pd.read_sql("SELECT * FROM paper_positions WHERE status!='archived'", con)
    closed = pos[pos["status"] == "closed"]
    opened = pos[pos["status"] == "open"]
    unreal, value = 0.0, 0.0
    for _, p in opened.iterrows():
        last = ind["close"][p["code"]].dropna() if ind is not None and p["code"] in ind["close"].columns else None
        px = float(last.iat[-1]) if last is not None and len(last) else p["entry"]
        unreal += (px - p["entry"]) * p["shares"]
        value += px * p["shares"]
    realized = float(closed["pnl"].sum()) if len(closed) else 0.0
    cash = cap + realized - float((opened["entry"] * opened["shares"]).sum() * (1 + half_cost)) if len(opened) else cap + realized
    equity = cash + value
    hist = [r[0] for r in con.execute("SELECT equity FROM paper_equity WHERE capital=?", (cap,))]
    peak = max(hist + [cap, equity])
    dd = equity / peak - 1
    return dict(
        open=int(len(opened)), pending=int((pos["status"] == "pending").sum()), closed=int(len(closed)),
        win_rate=float((closed["pnl"] > 0).mean()) if len(closed) else None,
        realized=realized, unrealized=unreal, cash=cash, equity=equity, capital=cap, drawdown=dd,
        stop_review=bool(pc.get("stop_review_dd") and dd <= -pc["stop_review_dd"]),
    )
