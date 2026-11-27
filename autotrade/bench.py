"""大勝ちは読めるか・S&P500 / オルカンに勝てるか（ユーザーの質問、2026-11-27）。結果はログに出す。

1. AIの点数の順位ごとに、5営業日後（翌日の寄付→6日後の寄付）に +10%以上・+20%以上上がる割合、−10%以上下がる割合
   （対象の全銘柄の割合と比べる。2008〜2018年と2019年以降に分けて）
2. 利益の偏り: 本番の口座・凍結した版・ランダムに選んだ場合（同じ仕組み）の、1回ごとの損益率の分布
3. 指数との比較: S&P500（配当込み）・全世界株式（ACWI・配当込み）を円に換算し、信託報酬 年0.1% を引いたもの（kabuscout.core と同じ）。
   日経平均（配当なし）も。autotrade は配当を受け取る場合も出す（指数は配当込みなので）
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import replace

import numpy as np
import pandas as pd

from . import data as D
from . import features as F
from . import sim
from .research import HOLDOUT, _fmt
from .strategy import Params
from .verify import ensemble_scores

log = logging.getLogger("autotrade.bench")


def big_moves(ds: pd.DataFrame, col: str, H: int) -> None:
    d = ds[ds["elig"] & ds[col].notna() & ds[f"fwd{H}"].notna() & (ds["date"] >= pd.Timestamp("2008-01-01"))].copy()
    days = np.sort(d["date"].unique())[::H]  # 重ならないように H 日おき
    d = d[d["date"].isin(set(days))]
    d["rank"] = d.groupby("date")[col].rank(ascending=False, method="first")
    d["n"] = d.groupby("date")[col].transform("count")
    d["pct"] = d["rank"] / d["n"]
    f = d[f"fwd{H}"]
    bins = [("1-5位", d["rank"] <= 5), ("6-30位", (d["rank"] > 5) & (d["rank"] <= 30)),
            ("上位1%", d["pct"] <= 0.01), ("上位10%", d["pct"] <= 0.10), ("中位40-60%", (d["pct"] > 0.4) & (d["pct"] <= 0.6)),
            ("下位10%", d["pct"] > 0.90), ("全銘柄", d["pct"] > -1)]
    for period, k in (("2008-18", d["date"] < pd.Timestamp(HOLDOUT)), ("2019+", d["date"] >= pd.Timestamp(HOLDOUT))):
        print(f"BIG [{period}] 5日後の値動き（{int(k.sum()):,}件・{d.loc[k, 'date'].nunique()}日）")
        base = None
        for name, m in bins:
            x = f[m & k]
            r = dict(n=len(x), mean=x.mean(), med=x.median(), up10=(x >= 0.10).mean(), up20=(x >= 0.20).mean(),
                     dn10=(x <= -0.10).mean(), sd=x.std())
            if name == "全銘柄":
                base = r
            print(f"BIG   {name:10s} n={r['n']:>8,} 平均 {r['mean'] * 100:+.2f}% 中央値 {r['med'] * 100:+.2f}% "
                  f"+10%以上 {r['up10'] * 100:5.2f}% +20%以上 {r['up20'] * 100:5.2f}% −10%以下 {r['dn10'] * 100:5.2f}% "
                  f"ばらつき {r['sd'] * 100:.1f}%")
        # 実際に大きく上がった銘柄（その日の上位1%）のうち、AIが前日に上位30位に入れていた割合
        top_real = d[k].groupby("date")[f"fwd{H}"].rank(ascending=False, pct=True) <= 0.01
        caught = (d.loc[k][top_real]["rank"] <= 30).mean()
        share30 = (d.loc[k, "rank"] <= 30).mean()
        print(f"BIG   実際の上位1%（大勝ち）のうち、AIの上位30位に入っていた割合 {caught * 100:.2f}%（ランダムなら {share30 * 100:.2f}%）")


def trade_dist(tag: str, res: dict) -> None:
    n = res["trades"]["net"].dropna()
    if not len(n):
        return
    lr = np.log1p(n).sort_values(ascending=False)
    k = max(1, int(len(lr) * 0.05))
    pos = lr[lr > 0].sum()
    print(f"DIST {tag:18s} 取引 {len(n):5d} 平均 {n.mean() * 100:+.2f}% 勝率 {(n > 0).mean() * 100:.0f}% "
          f"+20%以上 {(n >= 0.2).mean() * 100:.1f}% −20%以下 {(n <= -0.2).mean() * 100:.1f}% | "
          f"上位5%の平均 {np.expm1(lr.iloc[:k].mean()) * 100:+.1f}% 残り95%の平均 {np.expm1(lr.iloc[k:].mean()) * 100:+.2f}% | "
          f"上位5%が利益（勝ちの合計）に占める割合 {lr.iloc[:k].sum() / pos * 100:.0f}%")


def main(argv: list[str]) -> None:
    import warnings
    from pathlib import Path
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")
    from kabuscout import core
    from kabuscout.backtest import _stats
    from kabuscout.config import load_config
    from . import live
    cfg = load_config(Path(argv[argv.index("--config") + 1]) if "--config" in argv else Path("config.yaml"))
    mp = json.loads(argv[argv.index("--model-params") + 1]) if "--model-params" in argv else None
    s = live.load_settings()
    p_main = Params(**s["strategy"])
    p_frozen = Params(**{**s["strategy"], **{k: v for k, v in s["shadow_accounts"]["frozen"].items() if k != "risk"}})
    H = s["model"]["horizons"][0]
    t0 = time.time()
    ds, market, masks = D.build(cfg, horizons=(H,))
    elig = np.zeros(masks["shape"], dtype=bool)
    elig[masks["ii"], masks["jj"]] = ds["elig"].to_numpy()
    S, _ = ensemble_scores(ds, masks, F.FEATURES, H, params=mp)
    ds["s_ens"] = S[masks["ii"], masks["jj"]]
    print(f"BENCH built+scored {time.time() - t0:.0f}s")
    # 1. 大勝ちは読めるか
    big_moves(ds, "s_ens", H)
    # 2. 利益の偏り（ランダムと比べる）
    runs = {"本番(減らさない)": sim.run(market, S, elig, p_main, start="2008-01-01"),
            "凍結した版": sim.run(market, S, elig, p_frozen, start="2008-01-01")}
    rng = np.random.default_rng(11)
    for k in range(3):
        R = np.where(np.isfinite(S), rng.random(S.shape), np.nan)
        runs[f"ランダム{k}(減らさない)"] = sim.run(market, R, elig, p_main, start="2008-01-01")
    for tag, r in runs.items():
        trade_dist(tag, r)
    # 3. 指数との比較
    ix = core.load_indices(cfg)
    if ix is None:
        print("INDEX 指数のデータを取れませんでした")
        return
    rows = {}
    for tag, p in (("autotrade 本番(減らさない)", p_main), ("autotrade 凍結した版", p_frozen)):
        rows[tag] = sim.run(market, S, elig, p, start="2008-01-01")
        rows[tag + " 配当込み"] = sim.run(market, S, elig, p, start="2008-01-01", dividends=True)
        rows[tag + " 配当込み・コスト0.5%・不成立日なし"] = sim.run(market, S, elig, replace(p, cost=0.005), start="2008-01-01",
                                                          dividends=True, realistic_fills=True)
    cal = rows["autotrade 本番(減らさない)"]["equity"].index
    R = core.index_returns_jpy(ix, cal)
    first_acwi = ix["acwi"].first_valid_index()
    mk = masks["mkt"].reindex(cal).ffill()
    curves = {k: v["equity"] for k, v in rows.items()}
    for name in R.columns:
        curves[name + "（円・配当込み・信託報酬0.1%）"] = (1 + R[name]).cumprod() * 1e6
    curves["日経平均（配当なし）"] = mk / mk.iloc[0] * 1e6
    for start in ("2008-04-01", "2019-01-01"):
        print(f"INDEX [{start}〜{cal[-1].date()}]（ACWI は {first_acwi.date()} から）")
        for k, eq in curves.items():
            e = eq[eq.index >= pd.Timestamp(start)].dropna()
            if len(e) < 50:
                continue
            st = _stats(e)
            print(f"INDEX   {k:44s} 年率 {st['cagr'] * 100:+6.1f}% 最大下落 {st['max_dd'] * 100:6.1f}% "
                  f"シャープ {st['sharpe']:.2f} 最悪の年 {st['worst_year'] * 100:+6.1f}%")
    # 100万円で2019年から（autotrade は配当込み・同じ口座の大きさ）
    print("INDEX [100万円で2019-01から]")
    for tag, p in (("autotrade 本番(減らさない)", p_main), ("autotrade 凍結した版", p_frozen)):
        for ctag, kw in (("コスト0.2%", dict(p=p)), ("コスト0.5%・不成立日なし", dict(p=replace(p, cost=0.005), realistic_fills=True))):
            r = sim.run(market, S, elig, kw.pop("p"), start="2019-01-01", capital=1_000_000, dividends=True, **kw)
            st = sim.stats(r)
            yr = sim.yearly(r["equity"])
            print(f"INDEX   {tag + ' 配当込み ' + ctag:44s} 年率 {st['cagr'] * 100:+6.1f}% 最大下落 {st['max_dd'] * 100:6.1f}% "
                  f"シャープ {st['sharpe']:.2f} | " + " ".join(f"{a}:{b * 100:+.0f}" for a, b in yr.items()))
            curves[f"1M {tag} {ctag}"] = r["equity"]
    for name in R.columns:
        e = curves[name + "（円・配当込み・信託報酬0.1%）"]
        e = e[e.index >= pd.Timestamp("2019-01-01")]
        yr = sim.yearly(e)
        st = _stats(e)
        print(f"INDEX   {name:44s} 年率 {st['cagr'] * 100:+6.1f}% 最大下落 {st['max_dd'] * 100:6.1f}% シャープ {st['sharpe']:.2f} | "
              + " ".join(f"{a}:{b * 100:+.0f}" for a, b in yr.items()))
    # 週ごとの差の t（2019年〜、本番 vs S&P500）と相関、半分ずつ
    a = curves["1M autotrade 本番(減らさない) コスト0.2%"].resample("W-FRI").last().pct_change().dropna()
    for name in R.columns:
        b = curves[name + "（円・配当込み・信託報酬0.1%）"]
        b = b[b.index >= pd.Timestamp("2019-01-01")].resample("W-FRI").last().pct_change().dropna()
        dd = (a - b).dropna()
        print(f"INDEX   2019〜 本番 − {name}: 週の差の平均 {dd.mean() * 100:+.3f}% t={dd.mean() / dd.std() * np.sqrt(len(dd)):.2f} "
              f"相関 {a.corr(b):+.2f}")
        hr = 0.5 * a.reindex(dd.index) + 0.5 * b.reindex(dd.index)
        half = (1 + hr).cumprod()
        yrs = (half.index[-1] - half.index[0]).days / 365.25
        print(f"INDEX   2019〜 本番50% + {name}50%（毎週戻す）: 年率 {(half.iloc[-1] ** (1 / yrs) - 1) * 100:+.1f}% "
              f"最大下落 {(half / half.cummax() - 1).min() * 100:.1f}%（週の値で） シャープ（週） {hr.mean() / hr.std() * np.sqrt(52):.2f}")
    print(f"BENCH TOTAL {time.time() - t0:.0f}s")
