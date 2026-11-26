"""研究: ウォークフォワードのAI × 売買の形の組み合わせを、kabuscout と同じ口座の前提で比べる。

決め方の約束（後から都合よく選ばないため）:
- 売買の形・AIの設定は「2008〜2018年（設計期間）」の成績だけで選ぶ
- 2019年以降（検証期間）は選んだ後の確認に使う。kabuscout の本番の比較基準（backtest.BASE_START）も 2019-01-01
- 口座: 資金100万円・100株単位・実際の株価で単元を判定・翌営業日の寄付で約定・往復コスト0.2%（0.3%・0.5% でも確認）
"""
from __future__ import annotations

import argparse
import json
import logging
import time

import numpy as np
import pandas as pd

from . import data as D
from . import features as F
from . import model as M
from . import sim
from .strategy import Params

log = logging.getLogger("autotrade.research")

# kabuscout の本番の過去ペーパートレード（data/backtest.json、2026-11-20 時点。2008-01-04〜2026-11-20）
KABUSCOUT = {"all": dict(cagr=0.088, max_dd=-0.378, sharpe=0.707, worst_year=-0.166),
             "2019": dict(cagr=0.065, max_dd=-0.304, sharpe=0.534, worst_year=-0.167)}
DESIGN = ("2008-01-01", "2018-12-31")
HOLDOUT = "2019-01-01"


def _fmt(s: dict) -> str:
    g = lambda k: s.get(k) if s.get(k) is not None else float("nan")
    return (f"CAGR {g('cagr')*100:+6.1f}% MDD {g('max_dd')*100:6.1f}% Sharpe {g('sharpe'):5.2f} "
            f"worst {g('worst_year')*100:+6.1f}% trades/yr {g('per_year'):5.0f} win {g('win')*100:4.0f}% "
            f"avg {g('avg')*100:+5.2f}% expo {g('exposure')*100:4.0f}%")


def design_stats(res: dict) -> dict:
    eq = res["equity"]
    k = (eq.index >= pd.Timestamp(DESIGN[0])) & (eq.index <= pd.Timestamp(DESIGN[1]))
    sub = dict(equity=eq[k], trades=res["trades"][(res["trades"]["date"] >= pd.Timestamp(DESIGN[0])) &
                                                  (res["trades"]["date"] <= pd.Timestamp(DESIGN[1]))],
               exposure=res["exposure"][k])
    return sim.stats(sub)


def diagnostics(ds: pd.DataFrame, col: str, H: int) -> None:
    """AIの点数そのものの質: 年ごとの順位相関（IC）と、上位1%・上位5%の H 日リターンの全体平均との差。"""
    d = ds[ds["elig"] & ds[col].notna() & ds[f"fwd{H}"].notna()]
    rows = []
    for y, g in d.groupby(d["date"].dt.year):
        days = np.sort(g["date"].unique())[::H]
        g = g[g["date"].isin(set(days))]
        ic = g.groupby("date").apply(lambda x: x[col].rank().corr(x[f"fwd{H}"].rank())).mean()
        q = g.groupby("date")[col].rank(pct=True, ascending=False)
        base = g.groupby("date")[f"fwd{H}"].transform("mean")
        ex = g[f"fwd{H}"] - base
        rows.append((y, ic, ex[q <= 0.01].mean(), ex[q <= 0.05].mean()))
    print(f"DIAG {col} H={H}: year IC top1%excess top5%excess")
    for y, ic, a, b in rows:
        print(f"DIAG {y} IC {ic:+.3f} top1% {a*100:+.2f}% top5% {b*100:+.2f}%")
    arr = np.array([r[1] for r in rows])
    print(f"DIAG mean IC {np.nanmean(arr):+.3f}  t={np.nanmean(arr)/np.nanstd(arr)*np.sqrt(np.isfinite(arr).sum()):.2f}")


def run_grid(market, scores, elig, grid: list[Params], label: str, dividends: bool = False) -> list[dict]:
    out = []
    for p in grid:
        t0 = time.time()
        res = sim.run(market, scores, elig, p, start="2008-01-01", dividends=dividends)
        st_all, st_19, st_d = sim.stats(res), sim.stats(res, HOLDOUT), design_stats(res)
        yr = sim.yearly(res["equity"])
        out.append(dict(label=label, params=p.to_dict(), all=st_all, since2019=st_19, design=st_d,
                        years={int(k): float(v) for k, v in yr.items()}))
        print(f"GRID [{label}] n={p.n_hold} reb={p.rebalance_every} keep={p.keep_rank} sl={p.stop_loss} "
              f"reg={p.regime_ma}x{p.regime_scale} cost={p.cost} div={int(dividends)} ({time.time()-t0:.0f}s)")
        print(f"   design08-18 {_fmt(st_d)}")
        print(f"   all08-26    {_fmt(st_all)}")
        print(f"   since2019   {_fmt(st_19)}")
        print("   years " + " ".join(f"{k}:{v*100:+.0f}" for k, v in yr.items()))
    return out


def default_grid() -> list[Params]:
    g = []
    for n in (5, 8, 12):
        for reb in (5, 10):
            for keep_mult in (3, 6):
                g.append(Params(n_hold=n, rebalance_every=reb, keep_rank=n * keep_mult, depth=n * 5))
    return g


def main(argv: list[str]) -> None:
    import warnings
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--horizons", default="5,20")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--events", type=int, default=1)
    ap.add_argument("--sample-every", type=int, default=3)
    ap.add_argument("--half-life", type=float, default=0)
    ap.add_argument("--grid", default="default")
    ap.add_argument("--features", default="all")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--model-params", default="")
    a = ap.parse_args(argv)
    from pathlib import Path
    from kabuscout.config import load_config
    cfg = load_config(Path(a.config))
    mp = json.loads(a.model_params) if a.model_params else None
    hs = [int(x) for x in a.horizons.split(",")]
    t0 = time.time()
    ds, market, masks = D.build(cfg, horizons=hs, with_events=bool(a.events))
    print(f"BUILD done {time.time()-t0:.0f}s rows={len(ds)} mem={ds.memory_usage().sum()/1e9:.2f}GB")
    feats = F.FEATURES if a.features == "all" else F.PRICE_FEATURES
    elig = np.zeros(masks["shape"], dtype=bool)
    elig[masks["ii"], masks["jj"]] = ds["elig"].to_numpy()
    results = []
    preds = {}
    for H in hs:
        t1 = time.time()
        col = f"s{H}"
        ds[col] = M.walk_forward(ds, feats, f"y{H}", H, 2008, params=mp, seeds=tuple(range(a.seeds)),
                                 half_life_days=a.half_life or None, sample_every=a.sample_every)
        print(f"WF H={H} done {time.time()-t1:.0f}s")
        diagnostics(ds, col, H)
        preds[H] = col
    # 点数の組み合わせ（各 H の順位の平均）
    cols = list(preds.values())
    ds["s_mix"] = ds.groupby("date")[cols].rank(pct=True).mean(axis=1) if len(cols) > 1 else ds[cols[0]]
    for col in cols + (["s_mix"] if len(cols) > 1 else []):
        if col == "s_mix":
            diagnostics(ds, col, hs[-1])
        scores = D.to_wide(ds, col, masks)
        grid = default_grid() if a.grid == "default" else [Params(**json.loads(x)) for x in a.grid.split(";")]
        results += run_grid(market, scores, elig, grid, col)
    # 設計期間のシャープで並べた上位
    results.sort(key=lambda r: -(r["design"].get("sharpe") or -9))
    print("TOP by design-period Sharpe (2008-2018):")
    for r in results[:10]:
        p = r["params"]
        print(f"TOP [{r['label']}] n={p['n_hold']} reb={p['rebalance_every']} keep={p['keep_rank']} | design {_fmt(r['design'])}")
        print(f"      all {_fmt(r['all'])}")
        print(f"      2019+ {_fmt(r['since2019'])}")
    print("KABUSCOUT benchmark:", json.dumps(KABUSCOUT))
    print(f"TOTAL {time.time()-t0:.0f}s")
