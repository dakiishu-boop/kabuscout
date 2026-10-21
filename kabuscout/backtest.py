"""過去でのペーパートレード（ウォークフォワード方式）。

「今のAIで過去を売買したら」をそのままやると、AIは答えを知っている期間で採点されてしまう。
そこで1年ずつ進めながら、「その年が始まる前までのデータだけで学習したAI」でその年を売買する。
これを何年分も繰り返すことで、何ヶ月も待たずに、実際に近い形で実力を確かめる。

比較のため、同じシグナルから「ランダムに選んだ場合」と「日経平均を持ち続けた場合」も計算する。

注意（ページにも表示する）:
- 戦略の条件そのものは過去データで調整しているので、その分は成績が良く見えやすい
- 銘柄は現在上場している銘柄のみ（倒産・上場廃止した銘柄が含まれない＝生存者バイアス）
- 2018年10月以前は1000株単位の銘柄も多かったが、ここでは単位を無視して金額で按分している
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from . import learner, regimes
from .data import PriceStore
from .learner import CANDIDATES, MODEL_FEATURES

log = logging.getLogger(__name__)


def walk_forward(cfg: dict, ds: pd.DataFrame, start_year: int) -> pd.DataFrame:
    H = cfg["labeling"]["horizon_days"]
    d = ds[ds["complete"]].sort_values("date").reset_index(drop=True)
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
        m = learner._fit(CANDIDATES[0], tr[MODEL_FEATURES], tr["label"], learner._weights(tr, cfg["learning"]))
        out.append(te.assign(prob=m.predict_proba(te[MODEL_FEATURES])[:, 1]))
        log.info("  %d年: %d件で学習 → %d件を判定", y, len(tr), len(te))
    return pd.concat(out, ignore_index=True)


def simulate(preds: pd.DataFrame, cal: pd.DatetimeIndex, cfg: dict, pick: str = "ai", seed: int = 0) -> dict:
    """シグナル日の翌営業日寄付で買い、ラベル付けと同じ規則（利確/損切/期限）で決済する。"""
    pc, rc = cfg["paper_trading"], cfg["research"]
    cap, maxp, per_day, cost = pc["capital_jpy"], pc["max_positions"], pc["picks_per_day"], rc["cost"]
    rng = np.random.default_rng(seed)
    cal = cal[cal >= preds["date"].min()]
    pos_of = {d: i for i, d in enumerate(cal)}
    by_date = {d: g for d, g in preds.groupby("date")}
    cash_eq = float(cap)
    open_pos: list[tuple[int, float, float]] = []  # (決済日index, 投入額, リターン)
    curve, trades = [], []
    for i, d in enumerate(cal):
        still = []
        for ex_i, size, r in open_pos:
            if ex_i <= i:
                pnl = size * (r - cost)
                cash_eq += pnl
                trades.append((d, r - cost))
            else:
                still.append((ex_i, size, r))
        open_pos = still
        g = by_date.get(d)
        if g is not None and i + 1 < len(cal):
            free = maxp - len(open_pos)
            n_new = min(free, per_day)
            if g["crisis"].iloc[0] == 1:
                n_new = min(n_new, 1)  # 本番と同じく、危機局面では新規1銘柄まで
            if n_new > 0:
                g = g.sort_values("prob", ascending=False) if pick == "ai" else g.sample(frac=1, random_state=rng.integers(1e9))
                for _, r in g.head(n_new).iterrows():
                    size = cash_eq / maxp
                    open_pos.append((i + int(r["exit_days"]), size, float(r["ret"])))
        curve.append((d, cash_eq))
    eq = pd.Series(dict(curve))
    tr = pd.DataFrame(trades, columns=["date", "net"])
    return dict(equity=eq, trades=tr)


def _stats(eq: pd.Series, trades: pd.DataFrame | None = None) -> dict:
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    dd = (eq / eq.cummax() - 1).min()
    out = dict(cagr=float((eq.iloc[-1] / eq.iloc[0]) ** (1 / max(years, 1e-9)) - 1), max_dd=float(dd),
               total=float(eq.iloc[-1] / eq.iloc[0] - 1))
    if trades is not None and len(trades):
        out.update(trades=int(len(trades)), win=float((trades["net"] > 0).mean()), avg=float(trades["net"].mean()))
    return out


def run(cfg: dict, ds: pd.DataFrame) -> dict:
    bc = cfg["backtest"]
    preds = walk_forward(cfg, ds, bc["start_year"])
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
    res = dict(
        start=str(cal[0].date()), end=str(cal[-1].date()), cost=cfg["research"]["cost"],
        stats=dict(ai=_stats(eq, ai["trades"]), random=_stats(rnd_eq), index=_stats(index)),
        years=years, curve=dict(ai=weekly(eq), random=weekly(rnd_eq), index=weekly(index)),
        crisis_periods=[dict(name=e.name, start=e.start, end=e.end) for e in regimes.EVENTS if e.kind == "crisis"],
    )
    (cfg["data"]["db_path"].parent / "backtest.json").write_text(json.dumps(res, ensure_ascii=False))
    log.info("過去ペーパートレード: 年率 %+.1f%%（ランダム %+.1f%% / 日経平均 %+.1f%%）最大下落 %.1f%%",
             res["stats"]["ai"]["cagr"] * 100, res["stats"]["random"]["cagr"] * 100,
             res["stats"]["index"]["cagr"] * 100, res["stats"]["ai"]["max_dd"] * 100)
    return res
