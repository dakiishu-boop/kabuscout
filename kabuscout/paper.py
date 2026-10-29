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

from .risk import order_shares, stop_review_threshold, tick_size

log = logging.getLogger(__name__)


def _simulate(ind: dict, code: str, entry_date: pd.Timestamp, tp: float, sl: float, H: int, stop: str = "intraday"):
    """entry_date から最新日まで1本ずつ進めて、決済されたら (日付, 価格, 理由, 保有日数) を返す。
    stop="intraday": 場中の逆指値（1ティック不利に約定） / stop="close": 終値で割ったら翌日の寄付で売る（過去の検証と同じ）"""
    idx = ind["close"].index
    start = idx.searchsorted(entry_date)
    pending = False
    for k, i in enumerate(range(start, len(idx)), start=1):
        o, h, l, c = (float(ind[f][code].iat[i]) for f in ("open", "high", "low", "close"))
        if stop == "close":
            if pending:
                return idx[i], o, "損切（終値割れの翌朝）", k
        elif l <= sl:  # 逆指値（成行）なので1ティック不利に約定（過去の検証と同じ）
            return idx[i], (min(o, sl) if k > 1 else sl) - float(tick_size(sl)), "損切", k
        if h >= tp:
            return idx[i], (max(o, tp) if k > 1 else tp), "利確", k
        if k >= H:
            return idx[i], c, "期限", k
        pending = stop == "close" and c <= sl
    return None, None, None, int(len(idx) - start)  # numpy の整数のままだと SQLite に文字の塊（blob）で入り、Webに b'\x01…' と出る


def step(con, cfg: dict, ind: dict, picks: pd.DataFrame, data_date: pd.Timestamp, crisis: bool = False,
         acct: str = "paper", swap: bool = False) -> dict:
    """acct: 記録する表（paper = 本番。paper_swap = 比較用の口座）。
    swap: 決算好反応の上位候補が5枠の満杯で見送りになる日は、点が一番低い保有と入れ替える（比較用の口座だけ。ジャーナル 64）。"""
    P, E = f"{acct}_positions", f"{acct}_equity"
    pc = cfg["paper_trading"]
    H = cfg["labeling"]["horizon_days"]
    k_tp, k_sl = cfg["labeling"]["tp_atr"], cfg["labeling"]["sl_atr"]
    half_cost = cfg["research"]["cost"] / 2
    idx = ind["close"].index
    from .core import ai_capital
    events = _restart_if_capital_changed(con, ai_capital(cfg), P)
    if swap:
        events += _sell_swapped(con, cfg, ind, P)
    st = summary(con, cfg, ind, acct)
    equity, cash = st["equity"], st["cash"]
    slot = equity / pc["max_positions"]
    pos = pd.read_sql(f"SELECT * FROM {P}", con)

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
            con.execute(f"UPDATE {P} SET status='cancelled', reason='資金不足' WHERE id=?", (p["id"],))
            continue
        cash -= shares * entry * (1 + half_cost)
        hz = int(p["horizon"]) if pd.notna(p.get("horizon")) else H
        tp, sl = entry + kt * p["atr"], entry - ks * p["atr"]
        con.execute(f"UPDATE {P} SET status='open', entry_date=?, entry=?, shares=?, tp=?, sl=?, horizon=? WHERE id=?",
                    (ed.strftime("%Y-%m-%d"), entry, shares, tp, sl, hz, p["id"]))
        events.append(f"買 {p['code']} {p['name']} {shares}株 @{entry:,.0f}")

    # 2) 保有中 → 利確/損切/期限の判定（毎回エントリー日から再生するので実行漏れがあっても正しい）
    pos = pd.read_sql(f"SELECT * FROM {P} WHERE status='open'", con)
    for _, p in pos.iterrows():
        if p["code"] not in ind["close"].columns:  # 上場廃止・データ欠落
            continue
        h = int(p["horizon"]) if pd.notna(p.get("horizon")) else H
        ex_date, ex_px, why, held = _simulate(ind, p["code"], pd.Timestamp(p["entry_date"]), p["tp"], p["sl"], h,
                                              stop=cfg["labeling"].get("stop_mode", "intraday"))
        if ex_date is None:
            con.execute(f"UPDATE {P} SET days_held=? WHERE id=?", (held, p["id"]))
            continue
        pnl = (ex_px * (1 - half_cost) - p["entry"] * (1 + half_cost)) * p["shares"]
        con.execute(f"UPDATE {P} SET status='closed', exit_date=?, exit=?, pnl=?, reason=?, days_held=? WHERE id=?",
                    (ex_date.strftime("%Y-%m-%d"), ex_px, pnl, why, held, p["id"]))
        events.append(f"{why} {p['code']} {p['name']} {pnl:+,.0f}円")

    # 3) 新規の注文（翌営業日寄付きで約定予定）
    held = pd.read_sql(f"SELECT code FROM {P} WHERE status IN ('open','pending','selling')", con)["code"].tolist()
    free = pc["max_positions"] - len(held)
    already = con.execute(f"SELECT COUNT(*) FROM {P} WHERE signal_date=? AND status!='archived'",
                          (data_date.strftime("%Y-%m-%d"),)).fetchone()[0]
    if free > 0 and len(picks) and not already:  # 同じ日の再実行で二重注文しない
        notes: list[str] = []
        cand = _candidates(con, pc, ind, picks, held, notes)
        events += notes
        # 決算をまたぐ銘柄も避けない: 過去ペーパートレードで避けても年率・最大下落とも改善せず（AIが「次の決算までの日数」を
        # 特徴量として織り込み済み）、本番と過去の検証の売買ルールを揃えるため（ジャーナル ラウンド9）
        n_new = min(free, pc["picks_per_day"])
        if crisis:  # 特例（危機）局面では新規の買いを1銘柄に絞る
            n_new = min(n_new, 1)
        from .backtest import gate_active
        gate = pc.get("gate")
        if gate and gate_active(gate, float(ind["mkt_dev75"].iloc[-1].iloc[0]), float(ind["mkt_ret20"].iloc[-1].iloc[0]), crisis):
            n_new = min(n_new, gate["max_new"])
            events.append(f"地合いが弱いため新規の買いを控えています（{gate['name']}）")
        for _, r in cand.head(n_new).iterrows():
            _order(con, cfg, P, r, data_date, score=swap)
            events.append(f"注文 {r['code']} {r['name']} (翌寄付)")
    elif swap and free <= 0 and len(picks) and not already and not crisis:
        events += _swap(con, cfg, ind, picks, held, data_date, acct)
    con.commit()
    st = summary(con, cfg, ind, acct)
    con.execute(f"INSERT OR REPLACE INTO {E} VALUES (?,?,?)",
                (data_date.strftime("%Y-%m-%d"), st["equity"], ai_capital(cfg)))
    con.commit()
    return summary(con, cfg, ind, acct) | {"events": events}


def _candidates(con, pc: dict, ind: dict, picks: pd.DataFrame, held: list[str], notes: list[str]) -> pd.DataFrame:
    """その日の候補を、本番と同じ順に絞り込む（保有中・AIの下限・合意モード・同じ業種の上限）。"""
    cand = picks[~picks["code"].isin(held)]
    if pc.get("min_prob") and cand["prob"].notna().any():
        cand = cand[cand["prob"] >= pc["min_prob"]]
    if pc.get("consensus") and "agreement" in cand and cand["agreement"].notna().any():
        cand = cand[cand["agreement"] >= pc["consensus"]]  # 合意モード（過去の検証と同じ）
    elif (pc.get("weak_consensus") and "agreement" in cand and cand["agreement"].notna().any()
          and float(ind["mkt_dev75"].iloc[-1].iloc[0]) < 0):
        cand = cand[cand["agreement"] >= pc["weak_consensus"]]
        notes.append(f"相場が弱い（日経平均が75日線割れ）ため、3つのモデルがそろって上位{1 - pc['weak_consensus']:.0%}と評価した銘柄だけを対象にしています")
    cap_sec = pc.get("max_per_sector")
    if cap_sec and "sector" in cand:  # 同じ業種に偏らないように（研究ループで採用された場合）
        sec = dict(con.execute("SELECT code, sector33 FROM universe").fetchall())
        secs = [sec.get(c, "?") for c in held]
        keep = []
        for i, r in cand.iterrows():
            if secs.count(r["sector"]) < cap_sec:
                keep.append(i)
                secs.append(r["sector"])
        cand = cand.loc[keep]
    return cand


def _order(con, cfg: dict, P: str, r, data_date: pd.Timestamp, score: bool = False) -> None:
    """翌営業日の寄付で買う注文を記録する。score: 比較用の口座では、入れ替えに使う「注文した時の点（final_score）」も残す。"""
    lab = cfg["labeling"]
    from .core import ai_capital
    cols = "code,name,signal_date,status,atr,prob,tp_atr,sl_atr,horizon,capital" + (",score" if score else "")
    vals = [r["code"], r["name"], data_date.strftime("%Y-%m-%d"), "pending", float(r["atr"]),
            None if pd.isna(r["prob"]) else float(r["prob"]), float(r.get("tp_atr", lab["tp_atr"])),
            float(r.get("sl_atr", lab["sl_atr"])), int(r.get("horizon", lab["horizon_days"])), ai_capital(cfg)]
    if score:
        vals.append(None if pd.isna(r.get("final_score")) else float(r["final_score"]))
    con.execute(f"INSERT INTO {P}({cols}) VALUES ({','.join('?' * len(vals))})", vals)


def _swap(con, cfg: dict, ind: dict, picks: pd.DataFrame, held: list[str], data_date: pd.Timestamp, acct: str) -> list[str]:
    """比較用の口座だけ: 決算好反応の候補がその日の上位（本番と同じ絞り込みの後の picks_per_day 件）に入り、5枠が埋まっていて
    見送りになるなら、注文した時の点が一番低い保有を翌営業日の寄付で売り、その候補を買う（1日1件まで・危機局面ではしない・
    株数が決まらないなら売らない）。過去の検証（4つの種）では年率 +12.1% → +15.2%、最大下落 −20.8% → −21.2%（ジャーナル 64）。"""
    P = f"{acct}_positions"
    pc = cfg["paper_trading"]
    half_cost = cfg["research"]["cost"] / 2
    top = _candidates(con, pc, ind, picks, held, []).head(pc["picks_per_day"])
    earn = top[top["strategies"].fillna("").map(lambda s: "earnings_drift" in s.split(","))]
    pos = pd.read_sql(f"SELECT * FROM {P} WHERE status='open'", con)
    if earn.empty or pos.empty:
        return []
    r, out = earn.iloc[0], pos.loc[pos["score"].fillna(np.inf).idxmin()]
    st = summary(con, cfg, ind, acct)
    last = ind["close"][out["code"]].dropna() if out["code"] in ind["close"].columns else None
    px_out = float(last.iat[-1]) if last is not None and len(last) else float(out["entry"])
    stop = float(r.get("sl_atr", cfg["labeling"]["sl_atr"])) * float(r["atr"])
    n = order_shares(pc, st["equity"], st["cash"] + px_out * out["shares"] * (1 - half_cost), float(r["close"]) * (1 + half_cost),
                     stop, st["equity"] / pc["max_positions"])
    if n <= 0:
        return []
    con.execute(f"UPDATE {P} SET status='selling', sell_date=?, reason='入れ替えで売却（決算好反応の上位候補）' WHERE id=?",
                (data_date.strftime("%Y-%m-%d"), int(out["id"])))
    _order(con, cfg, P, r, data_date, score=True)
    return [f"入れ替え: {out['code']} {out['name']} を翌寄付で売り、{r['code']} {r['name']}（決算好反応）を翌寄付で買う"]


def _sell_swapped(con, cfg: dict, ind: dict, P: str) -> list[str]:
    """入れ替えで売ると決めた保有を、決めた日の翌営業日の寄付で売る（寄付が無ければその日の終値）。"""
    half_cost = cfg["research"]["cost"] / 2
    idx = ind["close"].index
    events = []
    for _, p in pd.read_sql(f"SELECT * FROM {P} WHERE status='selling'", con).iterrows():
        j = idx.searchsorted(pd.Timestamp(p["sell_date"]), side="right")
        if j >= len(idx) or p["code"] not in ind["close"].columns:
            continue
        px = float(ind["open"][p["code"]].iat[j])
        if not np.isfinite(px):
            px = float(ind["close"][p["code"]].iat[j])
        if not np.isfinite(px):
            continue
        pnl = (px * (1 - half_cost) - p["entry"] * (1 + half_cost)) * p["shares"]
        days = int(j - idx.searchsorted(pd.Timestamp(p["entry_date"])) + 1)
        con.execute(f"UPDATE {P} SET status='closed', exit_date=?, exit=?, pnl=?, days_held=? WHERE id=?",
                    (idx[j].strftime("%Y-%m-%d"), px, pnl, days, int(p["id"])))
        events.append(f"入れ替えで売却 {p['code']} {p['name']} {pnl:+,.0f}円")
    con.commit()
    return events


def start_from(con, src: str, dst: str) -> list[str]:
    """比較用の口座を、本番の口座の今の状態（保有・注文・決済の記録・毎日の評価額）を写して始める（最初の1回だけ）。
    以後は同じ候補を見るので、差はルールの違いの分だけになる。写した保有の「注文した時の点」は、通知した候補の記録の final_score。"""
    if con.execute(f"SELECT COUNT(*) FROM {dst}_positions").fetchone()[0] or con.execute(f"SELECT COUNT(*) FROM {dst}_equity").fetchone()[0]:
        return []
    cols = ",".join(r[1] for r in con.execute(f"PRAGMA table_info({src}_positions)") if r[1] != "id")
    con.execute(f"INSERT INTO {dst}_positions({cols}) SELECT {cols} FROM {src}_positions WHERE status!='archived'")
    con.execute(f"INSERT INTO {dst}_equity SELECT * FROM {src}_equity")
    con.execute(f"UPDATE {dst}_positions SET score=(SELECT s.final_score FROM signals s WHERE s.date={dst}_positions.signal_date "
                f"AND s.code={dst}_positions.code) WHERE score IS NULL")
    con.commit()
    return ["比較用の口座を、本番の口座の今の状態から始めました"]


def _restart_if_capital_changed(con, capital: float, P: str = "paper_positions") -> list[str]:
    """資金の設定が変わったら、それまでの記録を「保管」にして新しい資金で始め直す（削除はしない）。"""
    n = con.execute(f"SELECT COUNT(*) FROM {P} WHERE status!='archived' AND (capital IS NULL OR capital!=?)",
                    (capital,)).fetchone()[0]
    if not n:
        return []
    con.execute(f"UPDATE {P} SET status='archived', reason=COALESCE(reason || ' / ', '') || '資金設定の変更で記録を区切り' "
                "WHERE status!='archived' AND (capital IS NULL OR capital!=?)", (capital,))
    con.commit()
    log.info("資金設定が %s円 に変わったため、ペーパートレードを始め直します（以前の %d件は保管）", f"{capital:,.0f}", n)
    return [f"資金を{capital:,.0f}円に変えたため、ペーパートレードを新しく始めました（以前の記録は保管しています）"]


def summary(con, cfg: dict, ind: dict | None = None, acct: str = "paper") -> dict:
    pc = cfg["paper_trading"]
    from .core import ai_capital
    cap, half_cost = ai_capital(cfg), cfg["research"]["cost"] / 2
    pos = pd.read_sql(f"SELECT * FROM {acct}_positions WHERE status!='archived'", con)
    closed = pos[pos["status"] == "closed"]
    opened = pos[pos["status"].isin(["open", "selling"])]  # selling: 入れ替えで翌寄付に売る予定（比較用の口座だけ）
    unreal, value = 0.0, 0.0
    for _, p in opened.iterrows():
        last = ind["close"][p["code"]].dropna() if ind is not None and p["code"] in ind["close"].columns else None
        px = float(last.iat[-1]) if last is not None and len(last) else p["entry"]
        unreal += (px - p["entry"]) * p["shares"]
        value += px * p["shares"]
    realized = float(closed["pnl"].sum()) if len(closed) else 0.0
    cash = cap + realized - float((opened["entry"] * opened["shares"]).sum() * (1 + half_cost)) if len(opened) else cap + realized
    equity = cash + value
    hist = [r[0] for r in con.execute(f"SELECT equity FROM {acct}_equity WHERE capital=?", (cap,))]
    peak = max(hist + [cap, equity])
    dd = equity / peak - 1
    return dict(
        open=int(len(opened)), pending=int((pos["status"] == "pending").sum()), closed=int(len(closed)),
        win_rate=float((closed["pnl"] > 0).mean()) if len(closed) else None,
        realized=realized, unrealized=unreal, cash=cash, equity=equity, capital=cap, drawdown=dd,
        stop_review=bool(dd <= -stop_review_threshold(cfg)),
    )
