"""過去でのペーパートレード（ウォークフォワード方式）。

「今のAIで過去を売買したら」をそのままやると、AIは答えを知っている期間で採点されてしまう。
そこで1年ずつ進めながら、「その年が始まる前までのデータだけで学習したAI」でその年を売買する。
これを何年分も繰り返すことで、何ヶ月も待たずに、実際に近い形で実力を確かめる。

比較のため、同じシグナルから「ランダムに選んだ場合」と「日経平均を持ち続けた場合」も計算する。

注意（ページにも表示する）:
- 戦略の条件そのものは過去データで調整しているので、その分は成績が良く見えやすい
- 銘柄は現在上場している銘柄のみ（倒産・上場廃止した銘柄が含まれない＝生存者バイアス）
- 売買単位はすべて100株として計算（2018年10月以前は1000株単位の銘柄も多く、当時は実際には買えなかった取引も含む）
- 資金・株数・現金・毎日の時価で資産を追う（最大下落は含み損込み）。株数の決め方は本番と同じ（risk.py）
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from . import learner, regimes
from .data import PriceStore
from .risk import order_shares

log = logging.getLogger(__name__)


def walk_forward(cfg: dict, ds: pd.DataFrame, start_year: int, features: list[str] | None = None,
                 params: dict | None = None) -> pd.DataFrame:
    """1年ずつ、その年より前のデータだけで本番と同じAI（3つの学習目標）を作って判定する。
    返す ret / label / exit_days は、実際に使う売買ルール（戦略ごと）での結果。"""
    H = learner.plan_horizon(cfg)
    d = learner.with_plan_outcomes(ds[ds["complete"]], cfg)
    d = d[d["complete_plan"]].sort_values("date").reset_index(drop=True)
    out = []
    for y in range(start_year, d["date"].dt.year.max() + 1):
        y0 = pd.Timestamp(f"{y}-01-01")
        train_dates = d.loc[d["date"] < y0, "date"].unique()
        if len(train_dates) < 500:
            continue
        purge = np.sort(train_dates)[-H]  # ラベル期間がテスト年にかからないように
        tr = d[d["date"] < purge]
        te = d[d["date"].dt.year == y]
        if te.empty:
            continue
        prob, sc = learner.predict(learner.fit_models(tr, cfg, params=params, features=features), te)
        out.append(te.assign(prob=sc, win_prob=prob))
        log.info("  %d年: %d件で学習 → %d件を判定", y, len(tr), len(te))
    pr = pd.concat(out, ignore_index=True)
    return pr.assign(ret=pr["ret_plan"], label=pr["label_plan"], exit_days=pr["exit_plan"],
                     complete=pr["complete_plan"], _relabeled=1)


def apply_plans(preds: pd.DataFrame, cfg: dict, store=None) -> pd.DataFrame:
    """戦略ごとの売買ルールが採用されていれば、そのルールで各取引の結果（損益・保有日数）を計算し直す
    （過去ペーパートレードを本番と同じ売買ルールで行うため）。"""
    from .strategies import plan_for
    lab = cfg["labeling"]
    if not lab.get("per_strategy") or "_relabeled" in preds.columns:
        return preds
    plans = [plan_for(k.split(","), lab) for k in preds["strategies"]]
    key = [(p["tp_atr"], p["sl_atr"], p["horizon_days"]) for p in plans]
    preds = preds.assign(_plan=key)
    glob = (lab["tp_atr"], lab["sl_atr"], lab["horizon_days"])
    todo = preds[preds["_plan"] != glob]
    if todo.empty:
        return preds.drop(columns="_plan")
    px = (store or PriceStore(cfg["data"]["price_dir"])).load(codes=sorted(todo["code"].unique()))
    c = px["close"].ffill(limit=5)
    o, h, l = (px[k].fillna(c) for k in ("open", "high", "low"))
    tix = {d: i for i, d in enumerate(c.index)}
    cix = {k: i for i, k in enumerate(c.columns)}
    arrs = [x.to_numpy("float32") for x in (o, h, l, c)]
    preds = preds.copy()
    # 新しい pandas は型の違う値の書き込みを拒否するので、書き込む列の型を先に揃える
    preds = preds.astype({"ret": "float64", "exit_days": "float64", "label": "float64", "complete": "bool"})
    for (kt, ks, hz), g in todo.groupby("_plan"):
        p = learner.forward_paths(*arrs, g["date"].map(tix).to_numpy(int), g["code"].map(cix).to_numpy(int), int(hz))
        r = learner.outcomes(p, g["atr"].to_numpy(float), kt, ks, int(hz))
        preds.loc[g.index, "ret"] = np.asarray(r["ret"], dtype="float64")
        preds.loc[g.index, "label"] = np.asarray(r["label"], dtype="float64")
        preds.loc[g.index, "exit_days"] = np.asarray(r["exit_days"], dtype="float64")
        preds.loc[g.index, "complete"] = np.asarray(r["complete"], dtype=bool)
    log.info("  戦略ごとの売買ルールで %d件の結果を計算し直しました", len(todo))
    return preds.drop(columns="_plan").assign(_relabeled=1)


def gate_active(gate: dict | None, mkt_dev75: float, mkt_ret20: float, crisis: bool) -> bool:
    """地合いによる買い控えの条件に当てはまるか。"""
    if not gate:
        return False
    return ((gate.get("dev75") is not None and mkt_dev75 < gate["dev75"])
            or (gate.get("ret20") is not None and mkt_ret20 < gate["ret20"])
            or (gate.get("crisis") and crisis))


_CLOSE: dict = {}


def _closes(cfg: dict, codes, store=None) -> pd.DataFrame:
    """保有中の銘柄を毎日の終値で評価するための終値表（同じ銘柄の組み合わせなら使い回す）。"""
    key = (tuple(sorted(codes)), id(store))
    if _CLOSE.get("key") != key:
        c = (store or PriceStore(cfg["data"]["price_dir"])).load(fields=["close"], codes=list(key[0]))["close"]
        _CLOSE.update(key=key, close=c.ffill(limit=5))
    return _CLOSE["close"]


def with_stops(preds: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """株数を決めるのに使う「損切までの値幅（円）」を各シグナルに付ける（その戦略の売買ルールの損切幅 × ATR）。"""
    from .strategies import plan_for
    if "_stop" in preds:
        return preds
    lab = cfg["labeling"]
    k = {s: plan_for(s.split(","), lab)["sl_atr"] for s in preds["strategies"].unique()}
    return preds.assign(_stop=preds["strategies"].map(k).astype(float) * preds["atr"])


def with_sectors(preds: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """同じ業種の保有数の制限に使う業種（東証33業種、銘柄一覧の現在の分類）。"""
    if "sector" in preds:
        return preds
    import sqlite3
    with sqlite3.connect(str(cfg["data"]["db_path"])) as con:
        sec = dict(con.execute("SELECT code, sector33 FROM universe").fetchall())
    return preds.assign(sector=preds["code"].map(sec).fillna("?"))


def simulate(preds: pd.DataFrame, cal: pd.DatetimeIndex, cfg: dict, pick: str = "ai", seed: int = 0,
             gate: dict | None = "config", max_positions: int | None = None, picks_per_day: int | None = None,
             account: dict | None = None, store=None) -> dict:
    """実際の口座と同じように、現金・株数（100株単位）・毎日の時価で資産を追う（最大下落も含み損込みで測る）。
    シグナル日の終値で株数を決めて注文し、翌営業日の寄付で約定、売買ルール（利確/損切/期限）で決済する。
    株数の決め方は risk.order_shares（本番のペーパートレードと同じ）。account で資金などを差し替えて比較できる。"""
    pc, rc = cfg["paper_trading"], cfg["research"]
    acc = {**pc, **(account or {})}
    cap, half_cost = acc["capital_jpy"], rc["cost"] / 2
    maxp = max_positions or pc["max_positions"]
    per_day = picks_per_day or pc["picks_per_day"]
    scan = cfg["screening"]["top_n"]  # 1日に見る候補の数（本番の通知と同じ上位15銘柄）
    rng = np.random.default_rng(seed)
    if gate == "config":
        gate = pc.get("gate")
    preds = with_sectors(with_stops(preds, cfg), cfg)
    cal = cal[cal >= preds["date"].min()]
    close = _closes(cfg, preds["code"].unique(), store).reindex(cal).ffill(limit=5)
    cix = {k: j for j, k in enumerate(close.columns)}
    C = close.to_numpy("float64")
    by_date = {d: g for d, g in preds.groupby("date")}
    cash = float(cap)
    open_pos: list[dict] = []
    pending: list[dict] = []
    curve, trades = [], []
    skipped = 0
    for i, d in enumerate(cal):
        # 1) 前日の注文を寄付で約定（寄付が想定より高く現金が足りなければ、買える単元まで減らす）
        for o in pending:
            px = o["entry"] * (1 + half_cost)
            n = min(o["shares"], int(cash / px // acc["lot_size"]) * acc["lot_size"])
            if n <= 0:
                continue
            cash -= n * px
            open_pos.append(dict(o, shares=n))
        pending = []
        # 2) 決済
        still = []
        for p in open_pos:
            if p["exit_i"] <= i:
                proceeds = p["shares"] * p["exit_px"] * (1 - half_cost)
                cost_in = p["shares"] * p["entry"] * (1 + half_cost)
                cash += proceeds
                trades.append((d, proceeds / cost_in - 1, proceeds - cost_in))
            else:
                still.append(p)
        open_pos = still
        # 3) 時価評価
        value = 0.0
        for p in open_pos:
            c = C[i, p["ci"]]
            value += p["shares"] * (c if np.isfinite(c) else p["entry"])
        eq = cash + value
        curve.append((d, eq))
        # 4) 新規の注文
        g = by_date.get(d)
        if g is None or i + 1 >= len(cal):
            continue
        n_new = min(maxp - len(open_pos), per_day)
        if g["crisis"].iloc[0] == 1:
            n_new = min(n_new, 1)  # 本番と同じく、危機局面では新規1銘柄まで
        if gate_active(gate, g["mkt_dev75"].iloc[0], g["mkt_ret20"].iloc[0], g["crisis"].iloc[0] == 1):
            n_new = min(n_new, gate["max_new"])
        if n_new <= 0:
            continue
        g = g.sort_values("prob", ascending=False) if pick == "ai" else g.sample(frac=1, random_state=rng.integers(1e9))
        held = {p["code"] for p in open_pos}
        cap_sec = acc.get("max_per_sector") if "sector" in g else None
        secs = [p.get("sector") for p in open_pos]
        avail = cash
        for _, r in g.head(scan).iterrows():
            if n_new <= 0:
                break
            if r["code"] in held or r["code"] not in cix:
                continue
            if cap_sec and secs.count(r["sector"]) >= cap_sec:  # 同じ業種に偏らないように
                continue
            n = order_shares(acc, eq, avail, float(r["close"]) * (1 + half_cost), float(r["_stop"]), eq / maxp)
            if n <= 0:  # 単元（100株）で買えない値段・損失の上限で1単元も買えない
                skipped += 1
                continue
            entry = float(r["entry"])
            if cap_sec:
                secs.append(r["sector"])
            pending.append(dict(code=r["code"], ci=cix[r["code"]], shares=n, entry=entry, stop=float(r["_stop"]), sector=r.get("sector"),
                                exit_i=i + int(r["exit_days"]), exit_px=entry * (1 + float(r["ret"]))))
            held.add(r["code"])
            avail -= n * float(r["close"]) * (1 + half_cost)
            n_new -= 1
    eq = pd.Series(dict(curve))
    tr = pd.DataFrame(trades, columns=["date", "net", "pnl"])
    return dict(equity=eq, trades=tr, skipped=skipped)


def after_tax(eq: pd.Series, rate: float = 0.20315) -> float:
    """税引き後の年率の目安。年ごとの利益に課税（損失は確定申告をすれば3年まで繰り越せる）し、
    納めた税金の分は翌年以降の元手から減らして複利で計算する。"""
    if len(eq) < 2:
        return float("nan")
    after, carry = float(eq.iloc[0]), []  # carry: (年, 繰越損失の割合でなく金額)
    prev = float(eq.iloc[0])
    for y, g in eq.groupby(eq.index.year):
        r = float(g.iloc[-1]) / prev - 1
        prev = float(g.iloc[-1])
        gain = after * r
        after += gain
        carry = [(yy, a) for yy, a in carry if y - yy <= 3]
        if gain < 0:
            carry.append((y, -gain))
            continue
        for k, (yy, a) in enumerate(carry):
            use = min(gain, a)
            gain -= use
            carry[k] = (yy, a - use)
        carry = [(yy, a) for yy, a in carry if a > 0]
        after -= gain * rate
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    return float((after / float(eq.iloc[0])) ** (1 / max(years, 1e-9)) - 1)


def _stats(eq: pd.Series, trades: pd.DataFrame | None = None) -> dict:
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    dd = (eq / eq.cummax() - 1).min()
    ends = eq.groupby(eq.index.year).last()
    yearly = ends / np.r_[eq.iloc[0], ends.to_numpy()[:-1]] - 1
    out = dict(cagr=float((eq.iloc[-1] / eq.iloc[0]) ** (1 / max(years, 1e-9)) - 1), max_dd=float(dd),
               total=float(eq.iloc[-1] / eq.iloc[0] - 1), after_tax=after_tax(eq), worst_year=float(yearly.min()))
    if trades is not None and len(trades):
        out.update(trades=int(len(trades)), win=float((trades["net"] > 0).mean()), avg=float(trades["net"].mean()))
    return out


def run(cfg: dict, ds: pd.DataFrame, preds: pd.DataFrame | None = None) -> dict:
    bc = cfg["backtest"]
    if preds is None:
        preds = walk_forward(cfg, ds, bc["start_year"])
        preds = preds[preds["complete"]]
    preds = with_sectors(with_stops(preds, cfg), cfg)
    proxy = cfg["universe"]["market_proxy"]
    m = PriceStore(cfg["data"]["price_dir"]).load(fields=["close"], codes=[proxy])["close"][proxy].dropna()
    cal = m.index[m.index >= preds["date"].min()]
    ai = simulate(preds, cal, cfg, "ai")
    rnd = [simulate(preds, cal, cfg, "random", seed=s)["equity"] for s in range(5)]
    rnd_eq = pd.concat(rnd, axis=1).mean(axis=1)
    index = m.reindex(cal).ffill() / m.reindex(cal).ffill().iloc[0] * cfg["paper_trading"]["capital_jpy"]

    eq = ai["equity"]
    years = []
    for y, g in eq.groupby(eq.index.year):
        prev = eq[eq.index.year < y]
        start = prev.iloc[-1] if len(prev) else eq.iloc[0]
        t = ai["trades"][ai["trades"]["date"].dt.year == y]
        tp = index[index.index.year == y]
        tprev = index[index.index.year < y]
        rp = rnd_eq[rnd_eq.index.year == y]
        rprev = rnd_eq[rnd_eq.index.year < y]
        crisis_days = preds.loc[preds["date"].dt.year == y].groupby("date")["crisis"].first()
        evs = sorted({e.name for e in regimes.EVENTS if e.kind == "crisis" and e.start[:4] <= str(y) <= e.end[:4]})
        years.append(dict(
            year=int(y), era=str(regimes.era_of([pd.Timestamp(f"{y}-06-30")])[0]),
            ai=float(g.iloc[-1] / start - 1),
            random=float(rp.iloc[-1] / (rprev.iloc[-1] if len(rprev) else rnd_eq.iloc[0]) - 1),
            index=float(tp.iloc[-1] / (tprev.iloc[-1] if len(tprev) else index.iloc[0]) - 1),
            trades=int(len(t)), win=float((t["net"] > 0).mean()) if len(t) else None,
            crisis_share=float(crisis_days.mean()) if len(crisis_days) else 0.0, events=evs))
    # 平常時/危機局面での1取引あたり成績（AIが選んだ取引）
    weekly = lambda s: [[d.strftime("%Y-%m-%d"), round(float(v))] for d, v in s.resample("W-FRI").last().dropna().items()]
    pc = cfg["paper_trading"]
    res = dict(
        start=str(cal[0].date()), end=str(cal[-1].date()), cost=cfg["research"]["cost"],
        account=dict(capital=pc["capital_jpy"], max_positions=pc["max_positions"], picks_per_day=pc["picks_per_day"],
                     risk_per_trade=pc.get("risk_per_trade"), lot_size=pc["lot_size"], skipped=ai["skipped"]),
        stats=dict(ai=_stats(eq, ai["trades"]), random=_stats(rnd_eq), index=_stats(index)),
        years=years, curve=dict(ai=weekly(eq), random=weekly(rnd_eq), index=weekly(index)),
        crisis_periods=[dict(name=e.name, start=e.start, end=e.end) for e in regimes.EVENTS if e.kind == "crisis"],
    )
    res["gate"] = cfg["paper_trading"].get("gate")
    from .health import expectation
    res["expect"] = expectation(ai["trades"])  # 実際のペーパートレードが想定どおりかの判定に使う（health.py）
    (cfg["data"]["db_path"].parent / "backtest.json").write_text(json.dumps(res, ensure_ascii=False))
    res["_preds"], res["_cal"] = preds, cal
    log.info("過去ペーパートレード: 年率 %+.1f%%（ランダム %+.1f%% / 日経平均 %+.1f%%）最大下落 %.1f%%",
             res["stats"]["ai"]["cagr"] * 100, res["stats"]["random"]["cagr"] * 100,
             res["stats"]["index"]["cagr"] * 100, res["stats"]["ai"]["max_dd"] * 100)
    return res
