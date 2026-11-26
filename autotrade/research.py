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


def feature_ics(ds: pd.DataFrame, H: int, feats: list[str]) -> None:
    """特徴量ごとの単独の予測力（その日の順位と H 日後の順位の相関の平均。設計期間と検証期間）。"""
    d = ds[ds["elig"] & ds[f"y{H}"].notna() & (ds["date"] >= pd.Timestamp("2008-01-01"))]
    days = np.sort(d["date"].unique())[::H]
    d = d[d["date"].isin(set(days))]
    rows = []
    for f in feats:
        ic = d.groupby("date").apply(lambda x: x[f].corr(x[f"y{H}"]))
        rows.append((f, ic[ic.index < pd.Timestamp(HOLDOUT)].mean(), ic[ic.index >= pd.Timestamp(HOLDOUT)].mean()))
    rows.sort(key=lambda r: -abs(r[1]))
    for f, a, b in rows:
        print(f"FEATIC {f:22s} 2008-18 {a:+.3f}  2019+ {b:+.3f}")


def universe_ew(market, elig) -> pd.Series:
    """対象銘柄を毎日すべて同じ金額ずつ持った場合（寄付→翌日の寄付、コストなし）。母集団そのものの上げ幅（生存者バイアス込み）。"""
    o = market.open
    r = o[1:] / o[:-1] - 1
    r = np.where(elig[:-1] & np.isfinite(r) & (np.abs(r) < 1.0), r, np.nan)
    m = np.nanmean(r, axis=1)
    m = np.nan_to_num(m)
    eq = pd.Series(np.r_[1.0, np.cumprod(1 + m)] * 1e6, index=market.cal)
    return eq[eq.index >= pd.Timestamp("2008-01-01")]


def suite_v2(market, scores, elig, label: str) -> None:
    from kabuscout.backtest import _stats
    base = [Params(n_hold=5, rebalance_every=5, keep_rank=30, depth=25),
            Params(n_hold=5, rebalance_every=10, keep_rank=30, depth=25),
            Params(n_hold=8, rebalance_every=5, keep_rank=48, depth=40),
            Params(n_hold=8, rebalance_every=10, keep_rank=48, depth=40),
            Params(n_hold=12, rebalance_every=10, keep_rank=36, depth=60)]
    risk = []
    for b in (base[0], base[2]):
        d = b.to_dict()
        risk += [Params(**{**d, "regime_ma": 200, "regime_scale": 0.5}), Params(**{**d, "regime_ma": 200, "regime_scale": 0.0}),
                 Params(**{**d, "regime_ma": 75, "regime_scale": 0.5}), Params(**{**d, "stop_loss": 0.15}),
                 Params(**{**d, "stop_loss": 0.08})]
    run_grid(market, scores, elig, base + risk, label)
    print("=== COST sensitivity")
    run_grid(market, scores, elig, [Params(**{**b.to_dict(), "cost": c}) for b in base[:3] for c in (0.003, 0.005)], label)
    print("=== DIVIDENDS included (after 20.315% tax)")
    run_grid(market, scores, elig, base[:3], label, dividends=True)
    print("=== RANDOM picks (same mechanics, random scores among eligible)")
    rng = np.random.default_rng(0)
    for sd in range(3):
        R = np.where(np.isfinite(scores), rng.random(scores.shape), np.nan)
        run_grid(market, R, elig, base[:3], f"random{sd}")
    eq = universe_ew(market, elig)
    k19 = eq.index >= pd.Timestamp(HOLDOUT)
    print(f"UNIVERSE-EW all {_fmt(_stats(eq))}")
    print(f"UNIVERSE-EW 2019+ {_fmt(_stats(eq[k19]))}")
    print("=== BANKRUPTCY injection (x2 and x6 the yearly listed-company bankruptcy rate; 10 runs each)")
    for p in base[:3]:
        for mult in (2, 6):
            c_all, d_all, c19 = [], [], []
            for sd in range(10):
                res = sim.run(market, scores, elig, p, start="2008-01-01", crash_mult=mult, seed=sd)
                a, b = sim.stats(res), sim.stats(res, HOLDOUT)
                c_all.append(a["cagr"]); d_all.append(a["max_dd"]); c19.append(b["cagr"])
            print(f"CRASH n={p.n_hold} reb={p.rebalance_every} x{mult}: CAGR all mean {np.mean(c_all)*100:+.1f}% "
                  f"worst {np.min(c_all)*100:+.1f}% | MDD mean {np.mean(d_all)*100:.1f}% worst {np.min(d_all)*100:.1f}% | "
                  f"2019+ CAGR mean {np.mean(c19)*100:+.1f}% worst {np.min(c19)*100:+.1f}%")


CHOSEN = Params(n_hold=8, rebalance_every=5, keep_rank=48, depth=40, regime_ma=75, regime_scale=0.5)


def suite_v3(market, scores_by_offset: dict, elig) -> None:
    """採用候補（CHOSEN）の頑健性: 学習データの間引き方・約定の遅れ・コスト・倒産・配当・資金の大きさ。"""
    p = CHOSEN
    d = p.to_dict()
    S = scores_by_offset[0]
    print("=== V3 chosen config")
    run_grid(market, S, elig, [p], "chosen")
    for off, Sx in scores_by_offset.items():
        if off:
            run_grid(market, Sx, elig, [p], f"offset{off}")
    print("=== V3 execution delayed by one day (decide with yesterday's scores)")
    Sd = np.vstack([np.full((1, S.shape[1]), np.nan), S[:-1]])
    run_grid(market, Sd, elig, [p], "delay1")
    print("=== V3 cost")
    run_grid(market, S, elig, [Params(**{**d, "cost": c}) for c in (0.003, 0.005)], "chosen")
    print("=== V3 dividends")
    run_grid(market, S, elig, [p], "chosen", dividends=True)
    print("=== V3 capital")
    for cap in (3_000_000, 10_000_000):
        res = sim.run(market, S, elig, p, capital=cap, start="2008-01-01")
        print(f"CAPITAL {cap:,}: all {_fmt(sim.stats(res))}")
        print(f"CAPITAL {cap:,}: 2019+ {_fmt(sim.stats(res, HOLDOUT))}")
    print("=== V3 bankruptcy injection")
    for mult in (2, 6):
        c_all, d_all, c19 = [], [], []
        for sd in range(10):
            res = sim.run(market, S, elig, p, start="2008-01-01", crash_mult=mult, seed=sd)
            a, b = sim.stats(res), sim.stats(res, HOLDOUT)
            c_all.append(a["cagr"]); d_all.append(a["max_dd"]); c19.append(b["cagr"])
        print(f"CRASH chosen x{mult}: CAGR all mean {np.mean(c_all)*100:+.1f}% worst {np.min(c_all)*100:+.1f}% | "
              f"MDD mean {np.mean(d_all)*100:.1f}% worst {np.min(d_all)*100:.1f}% | 2019+ CAGR mean {np.mean(c19)*100:+.1f}% worst {np.min(c19)*100:+.1f}%")
    res = sim.run(market, S, elig, p, start="2008-01-01")
    tr = res["trades"]
    print("REASONS", tr["reason"].value_counts().to_dict())
    eq = res["equity"]
    dd = eq / eq.cummax() - 1
    print("MDD date", dd.idxmin().date(), "peak", eq[:dd.idxmin()].idxmax().date())


def suite_v4(market, S, elig) -> None:
    """本番の形（3つのモデルの平均 × CHOSEN）の最終確認。kabuscout と年ごとに並べる。"""
    p = CHOSEN
    d = p.to_dict()
    out = run_grid(market, S, elig, [p], "ensemble")[0]
    run_grid(market, S, elig, [Params(**{**d, "regime_ma": 0, "regime_scale": 1.0}), Params(**{**d, "n_hold": 5, "keep_rank": 30, "depth": 25})],
             "ensemble-alt")
    Sd = np.vstack([np.full((1, S.shape[1]), np.nan), S[:-1]])
    run_grid(market, Sd, elig, [p], "ensemble-delay1")
    run_grid(market, S, elig, [Params(**{**d, "cost": c}) for c in (0.003, 0.005)], "ensemble-cost")
    for mult in (6,):
        c_all, d_all = [], []
        for sd in range(10):
            res = sim.run(market, S, elig, p, start="2008-01-01", crash_mult=mult, seed=sd)
            st = sim.stats(res)
            c_all.append(st["cagr"]); d_all.append(st["max_dd"])
        print(f"CRASH ensemble x{mult}: CAGR mean {np.mean(c_all)*100:+.1f}% worst {np.min(c_all)*100:+.1f}% MDD worst {np.min(d_all)*100:.1f}%")
    rng = np.random.default_rng(7)
    for sd in range(3):
        R = np.where(np.isfinite(S), rng.random(S.shape), np.nan)
        run_grid(market, R, elig, [p], f"random{sd}")
    print("YEARS-JSON " + json.dumps(out["years"]))
    print("STATS-JSON " + json.dumps(dict(all=out["all"], since2019=out["since2019"], design=out["design"])))


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
    ap.add_argument("--suite", default="")
    ap.add_argument("--offsets", default="0")
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
    if a.suite == "v4":
        H = hs[0]
        S = [D.to_wide(ds, preds[H], masks)]
        for off in (1, 2):
            ds["_tmp"] = M.walk_forward(ds, feats, f"y{H}", H, 2008, params=mp, sample_every=a.sample_every, offset=off)
            S.append(D.to_wide(ds, "_tmp", masks))
        ens = np.nanmean(np.stack(S), axis=0)  # 3つのモデルの予測の平均（毎日の運用と同じ）
        ds["s_ens"] = ens[masks["ii"], masks["jj"]]
        diagnostics(ds, "s_ens", H)
        suite_v4(market, ens, elig)
        print(f"TOTAL {time.time()-t0:.0f}s")
        return
    if a.suite == "v3":
        H = hs[0]
        S = {0: D.to_wide(ds, preds[H], masks)}
        for off in [int(x) for x in a.offsets.split(",") if int(x)]:
            sc = M.walk_forward(ds, feats, f"y{H}", H, 2008, params=mp, seeds=tuple(range(a.seeds)),
                                sample_every=a.sample_every, offset=off)
            ds["_tmp"] = sc
            S[off] = D.to_wide(ds, "_tmp", masks)
        suite_v3(market, S, elig)
        print(f"TOTAL {time.time()-t0:.0f}s")
        return
    if a.suite == "v2":
        feature_ics(ds, hs[0], feats)
        suite_v2(market, D.to_wide(ds, preds[hs[0]], masks), elig, preds[hs[0]])
        print(f"TOTAL {time.time()-t0:.0f}s")
        return
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
