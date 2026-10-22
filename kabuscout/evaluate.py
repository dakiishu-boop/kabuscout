"""新しく取り込んだデータ（特徴量）が予測に役立つかの確認。

1. 診断: 月ごとに、その特徴量で候補を5つに分け、上位1/5と下位1/5の平均リターン（実際の売買ルール）の差を測る。
   何ヶ月のうち何割で上位が勝ったか（一貫性）を見る
2. 過去ペーパートレード: その特徴量をAIに足した場合と足さない場合を、同じ口座シミュレーションで比べる

結果は数字だけをログと data/eval_<名前>.json に出す（J-Quants などの元データは再配布できないので公開しない）。
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from . import backtest, learner, pool
from .data import PriceStore

log = logging.getLogger(__name__)


def spread(d: pd.DataFrame, feature: str, target: str = "ret_plan", period: str = "M", min_n: int = 50) -> dict:
    x = d.dropna(subset=[feature, target])
    x = x.assign(per=x["date"].dt.to_period(period))
    x = x[x.groupby("per")[feature].transform("size") >= min_n]
    if x.empty:
        return dict(n=0)
    x = x.assign(q=x.groupby("per")[feature].transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False)))
    g = x.groupby(["per", "q"])[target].mean().unstack()
    sp = (g[4] - g[0]).dropna()
    t = float(sp.mean() / sp.std() * np.sqrt(len(sp))) if len(sp) > 2 and sp.std() > 0 else float("nan")
    return dict(n=int(len(x)), periods=int(len(sp)), first=str(sp.index.min()), last=str(sp.index.max()),
                spread=float(sp.mean()), positive=float((sp > 0).mean()), t=t,
                quintiles=[float(v) for v in g.mean().to_numpy()])


def run(cfg: dict, name: str, features: list[str], start_year: int) -> dict:
    from .pipeline import dataset, store_of
    from .data import refresh_universe
    from .db import connect
    pl = pool.load(cfg)
    if pl is None:
        pl = pool.build_full(store_of(cfg), cfg, refresh_universe(connect(cfg["data"]["db_path"]), cfg))
    ds = dataset(cfg, pl)
    d = learner.with_plan_outcomes(ds[ds["complete"]], cfg)
    d = d[d["complete_plan"]]
    out = dict(name=name, features=features, diagnostics={}, cost=cfg["research"]["cost"])
    for f in features:
        out["diagnostics"][f] = r = spread(d, f)
        if r.get("n"):
            log.info("診断 %-14s %s〜%s %dヶ月: 上位−下位 %+.2f%%・上位が勝った月 %.0f%%・t=%.2f（五分位 %s）",
                     f, r["first"], r["last"], r["periods"], r["spread"] * 100, r["positive"] * 100, r["t"],
                     " / ".join(f"{v:+.2%}" for v in r["quintiles"]))
        else:
            log.info("診断 %-14s データなし", f)

    proxy = cfg["universe"]["market_proxy"]
    m = PriceStore(cfg["data"]["price_dir"]).load(fields=["close"], codes=[proxy])["close"][proxy].dropna()
    res = {}
    for label, feats in (("現行", learner.MODEL_FEATURES), ("追加", learner.MODEL_FEATURES + features)):
        pr = backtest.walk_forward(cfg, ds, start_year, features=feats)
        pr = backtest.with_stops(pr[pr["complete"]], cfg)
        cal = m.index[m.index >= pr["date"].min()]
        sim = backtest.simulate(pr, cal, cfg, "ai")
        eq = sim["equity"]
        ends = eq.groupby(eq.index.year).last()
        yearly = ends / np.r_[eq.iloc[0], ends.to_numpy()[:-1]] - 1
        res[label] = dict(stats=backtest._stats(eq, sim["trades"]), yearly={int(k): float(v) for k, v in yearly.items()})
        log.info("過去ペーパートレード（%s）: %s", label,
                 " / ".join(f"{y}年 {v:+.1%}" for y, v in res[label]["yearly"].items())
                 + f" ｜ 最大下落 {res[label]['stats']['max_dd']:.1%}")
    out["backtest"] = res
    (cfg["data"]["db_path"].parent / f"eval_{name}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float))
    return out
