"""AIと指数を組み合わせた口座の正式な検証と、損益分岐のコスト（2026-11-27）。条件は docs/AUTOTRADE.md に実行前に記載。

判定: 指数50%（S&P500 / オルカン）＋AI50%（本番の形）が、100万円で2019年から・AI の部分は税引き後で、
コスト0.2% と コスト0.5%＋不成立日なし の両方で、同じ指数だけ（NISA・非課税）よりシャープが高いこと。
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
from .blend import index_levels
from .research import _fmt
from .strategy import Params
from .verify import ensemble_scores

log = logging.getLogger("autotrade.blendcheck")
TAX = 0.20315
SP, AC = "S&P500", "全世界株式（オルカン）"


def wt(a: pd.Series, b: pd.Series) -> float:
    ra = a.resample("W-FRI").last().pct_change().dropna()
    rb = b.resample("W-FRI").last().pct_change().dropna()
    d = (ra - rb).dropna()
    return float(d.mean() / d.std() * np.sqrt(len(d))) if len(d) > 10 and d.std() > 0 else float("nan")


def main(argv: list[str]) -> None:
    import warnings
    from pathlib import Path
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")
    from kabuscout import core
    from kabuscout.config import load_config
    from . import live
    cfg = load_config(Path(argv[argv.index("--config") + 1]) if "--config" in argv else Path("config.yaml"))
    mp = json.loads(argv[argv.index("--model-params") + 1]) if "--model-params" in argv else None
    s = live.load_settings()
    p_main = Params(**{**s["strategy"], "tax_ai": 0.0})  # 税は run() の中で行ごとに付ける
    p_frozen = Params(**{**s["strategy"], "tax_ai": 0.0, **{k: v for k, v in s["shadow_accounts"]["frozen"].items() if k != "risk"}})
    H = s["model"]["horizons"][0]
    t0 = time.time()
    ds, market, masks = D.build(cfg, horizons=(H,))
    elig = np.zeros(masks["shape"], dtype=bool)
    elig[masks["ii"], masks["jj"]] = ds["elig"].to_numpy()
    S, parts = ensemble_scores(ds, masks, F.FEATURES, H, params=mp)
    del ds
    ix = core.load_indices(cfg)
    if ix is None:
        print("BLEND 指数のデータを取れませんでした")
        return
    L = {n: index_levels(ix, market.cal, n)[0] for n in (SP, AC)}
    first = {n: market.cal[np.argmax(np.isfinite(L[n]))].date() for n in L}
    print(f"BLEND built+scored {time.time() - t0:.0f}s; index data from {first}; last US date "
          f"{pd.Timestamp(index_levels(ix, market.cal, SP)[1][-1]).date()}")
    Sd = np.vstack([np.full((1, S.shape[1]), np.nan), S[:-1]])

    def run(p: Params, start: str, cost: float = 0.002, real: bool = False, tax: bool = True, lot1000: bool = False,
            scores=None, name=SP):
        p = replace(p, cost=cost, tax_ai=TAX if tax else 0.0, index_name=name)
        kw = dict(start=start, capital=1_000_000, dividends=True, realistic_fills=real,
                  index_level=L[name] if p.index_share > 0 else None)
        if lot1000:
            kw["lot_schedule"] = [("2018-10-01", 1000)]
        return sim.run(market, S if scores is None else scores, elig, p, **kw)

    periods = [("2019-01〜", "2019-01-01", False), ("2008-04〜", "2008-04-01", False), ("2008-04〜 1000株<2018-10", "2008-04-01", True)]
    costs = [("コスト0.2%", 0.002, False), ("コスト0.5%＋不成立日なし", 0.005, True)]
    accounts = [(f"指数だけ {SP}", replace(p_main, index_share=1.0), SP), (f"指数だけ {AC}", replace(p_main, index_share=1.0), AC),
                ("AIだけ 本番", p_main, SP), ("AIだけ 凍結した版", p_frozen, SP)]
    for sh in (0.3, 0.5, 0.7):
        for nm in (SP, AC):
            accounts.append((f"指数{int(sh * 100)}% {nm} + 本番", replace(p_main, index_share=sh), nm))
            accounts.append((f"指数{int(sh * 100)}% {nm} + 凍結した版", replace(p_frozen, index_share=sh), nm))
    worst = {}
    gate = {}
    for ptag, start, lot1000 in periods:
        for ctag, cost, real in costs:
            for tax in (True, False):
                ttag = "AIは税引き後" if tax else "税引き前"
                print(f"BLEND === {ptag} {ctag} {ttag}（100万円から・指数は NISA で非課税）")
                base = {}
                for tag, p, nm in accounts:
                    r = run(p, start, cost, real, tax, lot1000, name=nm)
                    st = sim.stats(r)
                    eq = r["equity"]
                    if tag.startswith("指数だけ"):
                        base[nm] = (st, eq)
                    ref = base.get(nm, base.get(SP))
                    t = wt(eq, ref[1]) if ref and not tag.startswith("指数だけ") else float("nan")
                    tx = r["sleeve"].taxes_paid if r.get("sleeve") is not None else 0.0
                    print(f"BLEND {tag:28s} 年率 {st['cagr'] * 100:+6.1f}% 最大下落 {st['max_dd'] * 100:6.1f}% "
                          f"シャープ {st['sharpe']:.2f} 最悪の年 {st['worst_year'] * 100:+6.1f}% "
                          f"指数だけとの週の差 t={t:+.2f} 税 {tx:,.0f}円")
                    if tag in (f"指数50% {SP} + 本番", f"指数50% {AC} + 本番"):
                        worst[tag] = min(worst.get(tag, 0.0), st["max_dd"])
                        if ptag == "2019-01〜" and tax:
                            gate.setdefault(tag, []).append((ctag, st["sharpe"], ref[0]["sharpe"], st["cagr"], ref[0]["cagr"]))
    print("BLEND === 判定する2つの頑健性（2019-01〜・コスト0.2%・AIは税引き後）")
    for nm in (SP, AC):
        p = replace(p_main, index_share=0.5)
        ref = run(replace(p_main, index_share=1.0), "2019-01-01", name=nm)
        rs = sim.stats(ref)
        for k, Sx in enumerate([*parts, Sd]):
            tag = f"間引き方{k}" if k < 3 else "1日遅れ"
            r = run(p, "2019-01-01", scores=Sx, name=nm)
            st = sim.stats(r)
            worst[f"指数50% {nm} + 本番"] = min(worst[f"指数50% {nm} + 本番"], st["max_dd"])
            print(f"BLEND 指数50% {nm} + 本番 [{tag}] {_fmt(st)} | 指数だけ シャープ {rs['sharpe']:.2f}・年率 {rs['cagr'] * 100:+.1f}%")
    print("BLEND === 判定")
    for tag, rows in gate.items():
        ok = all(sc > si for _, sc, si, _, _ in rows)
        print(f"BLEND GATE {tag}: " + " / ".join(f"{c}: シャープ {sc:.2f} 対 指数だけ {si:.2f}（年率 {cc * 100:+.1f}% 対 {ci * 100:+.1f}%）"
                                                for c, sc, si, cc, ci in rows) + f" => {'PASS' if ok else 'FAIL'}")
    for tag, w in worst.items():
        print(f"BLEND 安全装置の目安 {tag}: 最悪の最大下落 {w * 100:.1f}% → しきい値 {min(0.9, max(0.35, -w + 0.10)):.2f}")
    # 損益分岐のコスト（ユーザーの依頼）: コストを変えて、指数だけ（コストに依らない）と並ぶ値
    print("BREAKEVEN === 往復コストを変えたときの年率（100万円から・配当込み）。指数だけはコストに依らない")
    grid = [round(x, 4) for x in np.arange(0.001, 0.00601, 0.0005)]
    for ptag, start in (("2019-01〜", "2019-01-01"), ("2008-04〜", "2008-04-01")):
        for tax in (False, True):
            ttag = "AIは税引き後" if tax else "税引き前"
            refs = {nm: sim.stats(run(replace(p_main, index_share=1.0), start, name=nm, tax=False))["cagr"] for nm in (SP, AC)}
            print(f"BREAKEVEN [{ptag} {ttag}] 指数だけ: {SP} {refs[SP] * 100:+.1f}% / {AC} {refs[AC] * 100:+.1f}%")
            for real in (False, True):
                curves = {}
                for tag, p, nm in (("AIだけ 本番", p_main, SP), ("AIだけ 凍結した版", p_frozen, SP),
                                   (f"指数50% {SP} + 本番", replace(p_main, index_share=0.5), SP),
                                   (f"指数50% {AC} + 本番", replace(p_main, index_share=0.5), AC)):
                    curves[tag] = [sim.stats(run(p, start, c, real, tax, name=nm))["cagr"] for c in grid]
                rtag = "不成立日なし" if real else "始値で約定"
                for tag, ys in curves.items():
                    cells = " ".join(f"{c * 100:.2f}%:{y * 100:+.1f}" for c, y in zip(grid, ys))
                    xs = []
                    for nm in (SP, AC):
                        target = refs[nm]
                        be = None
                        for (c0, y0), (c1, y1) in zip(zip(grid, ys), zip(grid[1:], ys[1:])):
                            if (y0 - target) * (y1 - target) <= 0 and y0 != y1:
                                be = c0 + (target - y0) * (c1 - c0) / (y1 - y0)
                                break
                        if be is None:
                            be_txt = "どのコストでも上" if ys[-1] > target else "どのコストでも下"
                        else:
                            be_txt = f"{be * 100:.2f}%"
                        xs.append(f"{nm}と並ぶコスト {be_txt}")
                    print(f"BREAKEVEN {rtag} {tag:22s} " + " | ".join(xs))
                    print(f"BREAKEVEN     {cells}")
                # 本番と凍結した版が並ぶコスト（どちらもコストで変わる）
                dif = [a - b for a, b in zip(curves["AIだけ 本番"], curves["AIだけ 凍結した版"])]
                be = next((c0 + (0 - d0) * (c1 - c0) / (d1 - d0) for c0, d0, c1, d1 in zip(grid, dif, grid[1:], dif[1:])
                           if d0 * d1 <= 0 and d0 != d1), None)
                print(f"BREAKEVEN {rtag} 本番と凍結した版が並ぶコスト: " +
                      (f"{be * 100:.2f}%" if be is not None else ("どのコストでも本番が上" if dif[-1] > 0 else "どのコストでも凍結した版が上")))
    print(f"BLEND TOTAL {time.time() - t0:.0f}s")
