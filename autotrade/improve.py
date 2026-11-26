"""もっと年率の高い形を探す（ユーザーの依頼、2026-11-26）。採用ルールは docs/AUTOTRADE.md に試す前に書いたもの:
 1. 2008〜2018年・2019年以降の両方で、凍結した版よりシャープ比が高い（または最大下落が同程度=2ポイント以内で年率が高い）
 2. 学習に使う日の間引き方3通りのどれでも凍結した版を上回る（各間引き方のモデル同士で比べる）
 3. 凍結した版との週ごとの損益の差の t ≥ 2.5
 4. 投資する量を増やしただけの案は数えない（ここでは試さない）

試す案（どれも検証で分かった事実から）:
 A. 入れ替えを早くする（1日遅れると効果が半減した → 新しい情報ほど価値がある）: 1・2・3営業日ごと
 B. 短い予測期間（H=2・3）と、H=5 との平均
 C. 特徴量を間引いた（max_features=0.6）モデルを、乱数の種2つ × 間引き方3つの平均にする
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
from . import model as M
from . import sim
from .research import HOLDOUT, _fmt
from .strategy import Params

log = logging.getLogger("autotrade.improve")
DESIGN_END = "2018-12-31"


def _sub(res, a=None, b=None):
    eq = res["equity"]
    k = np.ones(len(eq), bool)
    if a:
        k &= eq.index >= pd.Timestamp(a)
    if b:
        k &= eq.index <= pd.Timestamp(b)
    tr = res["trades"]
    tk = np.ones(len(tr), bool)
    if a:
        tk &= tr["date"] >= pd.Timestamp(a)
    if b:
        tk &= tr["date"] <= pd.Timestamp(b)
    return dict(equity=eq[k], trades=tr[tk], exposure=res["exposure"][k])


def summary(res) -> dict:
    return dict(design=sim.stats(_sub(res, None, DESIGN_END)), hold=sim.stats(_sub(res, HOLDOUT)), all=sim.stats(res))


def better(c: dict, f: dict) -> bool:
    """規則1の判定（1つの期間）: シャープが高い、または最大下落が2ポイント以内で年率が高い。"""
    return (c["sharpe"] > f["sharpe"]) or (c["max_dd"] >= f["max_dd"] - 0.02 and c["cagr"] > f["cagr"])


def paired_t(a: pd.Series, b: pd.Series) -> float:
    ra = a.resample("W-FRI").last().pct_change().dropna()
    rb = b.resample("W-FRI").last().pct_change().dropna()
    d = (ra - rb).dropna()
    return float(d.mean() / d.std() * np.sqrt(len(d))) if d.std() > 0 else 0.0


def judge(name: str, cand: list[dict], frozen: list[dict]) -> bool:
    """cand / frozen: [平均の結果, 間引き方0, 1, 2 の結果]（それぞれ sim.run の戻り値）。"""
    sc, sf = summary(cand[0]), summary(frozen[0])
    r1 = better(sc["design"], sf["design"]) and better(sc["hold"], sf["hold"])
    r2 = all(better(summary(c)["all"], summary(f)["all"]) for c, f in zip(cand[1:], frozen[1:])) if len(cand) > 1 else False
    t = paired_t(cand[0]["equity"], frozen[0]["equity"])
    ok = r1 and r2 and t >= 2.5
    print(f"JUDGE {name}: rule1(both periods) {'OK' if r1 else 'NG'} | rule2(all offsets) {'OK' if r2 else 'NG'} | "
          f"rule3 t={t:+.2f} {'OK' if t >= 2.5 else 'NG'} => {'PASS' if ok else 'fail'}")
    print(f"   cand  design {_fmt(sc['design'])}")
    print(f"   cand  2019+  {_fmt(sc['hold'])}")
    print(f"   cand  all    {_fmt(sc['all'])}")
    for k, c in enumerate(cand[1:]):
        print(f"   cand  offset{k} all {_fmt(summary(c)['all'])}")
    return ok


def confirm(market, S, elig, p0: Params, p1: Params) -> bool:
    """凍結した版と候補を同じ条件で並べる。採用の条件（実行前に docs/AUTOTRADE.md に書いたもの）:
    すべての行で、2019年以降のシャープ比か年率のどちらかが候補の方が高いこと。"""
    Sd = np.vstack([np.full((1, S.shape[1]), np.nan), S[:-1]])
    rows = [("base", S, {}, {}), ("cost0.3%", S, {"cost": 0.003}, {}), ("cost0.5%", S, {"cost": 0.005}, {}),
            ("delay1", Sd, {}, {}), ("realfills", S, {}, {"realistic_fills": True}),
            ("lot1000<2018-10", S, {}, {"lot_schedule": [("2018-10-01", 1000)]}),
            ("bounded3M", S, {}, {"withdraw_above": 3_000_000}), ("bounded10M", S, {}, {"withdraw_above": 10_000_000}),
            ("adv3%", S, {"max_adv_frac": 0.03}, {}),
            ("bounded3M+cost0.5%+realfills", S, {"cost": 0.005}, {"withdraw_above": 3_000_000, "realistic_fills": True})]
    ok_all = True
    for name, Sx, pk, rk in rows:
        rf = sim.run(market, Sx, elig, replace(p0, **pk), start="2008-01-01", **rk)
        rc = sim.run(market, Sx, elig, replace(p1, **pk), start="2008-01-01", **rk)
        f19, c19 = sim.stats(rf, HOLDOUT), sim.stats(rc, HOLDOUT)
        fa, ca = sim.stats(rf), sim.stats(rc)
        win = c19["sharpe"] > f19["sharpe"] or c19["cagr"] > f19["cagr"]
        ok_all &= win
        print(f"CONFIRM {name:30s} {'cand wins' if win else 'CAND LOSES'}")
        print(f"   frozen 2019+ {_fmt(f19)} | all CAGR {fa['cagr']*100:+.1f}% MDD {fa['max_dd']*100:.1f}% Sharpe {fa['sharpe']:.2f}")
        print(f"   cand   2019+ {_fmt(c19)} | all CAGR {ca['cagr']*100:+.1f}% MDD {ca['max_dd']*100:.1f}% Sharpe {ca['sharpe']:.2f}")
    print(f"CONFIRM RESULT: {'ADOPT' if ok_all else 'KEEP FROZEN'}")
    return ok_all


def small(market, S, elig, p0: Params, p1: Params) -> None:
    """ユーザーの実際の資金（100万円）での成績。kabuscout のジャーナル §82 の「100万円で2019年に開始」と同じ見方。"""
    rows = [("1M from 2019", "2019-01-01", {}, {}),
            ("1M from 2019, kept at 1M (withdraw above)", "2019-01-01", {}, {"withdraw_above": 1_000_000}),
            ("1M from 2019, cost0.3%+realfills", "2019-01-01", {"cost": 0.003}, {"realistic_fills": True}),
            ("1M from 2019, cost0.5%+realfills", "2019-01-01", {"cost": 0.005}, {"realistic_fills": True}),
            ("1M from 2019, kept at 1M, cost0.5%+realfills", "2019-01-01", {"cost": 0.005},
             {"withdraw_above": 1_000_000, "realistic_fills": True}),
            ("1M from 2008, kept at 1M", "2008-01-01", {}, {"withdraw_above": 1_000_000}),
            ("1M from 2008, kept at 1M, lot1000<2018-10", "2008-01-01", {}, {"withdraw_above": 1_000_000,
                                                                               "lot_schedule": [("2018-10-01", 1000)]})]
    for name, start, pk, rk in rows:
        for tag, p in (("frozen", p0), ("shadow", p1)):
            r = sim.run(market, S, elig, replace(p, **pk), start=start, capital=1_000_000, **rk)
            st = sim.stats(r)
            yr = sim.yearly(r["equity"])
            print(f"SMALL {name:48s} {tag:6s} {_fmt(st)} | years " + " ".join(f"{k}:{v * 100:+.0f}" for k, v in yr.items()))


def main(argv: list[str]) -> None:
    import argparse
    import warnings
    from pathlib import Path
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--model-params", default="")
    ap.add_argument("--only", default="A,B,C")
    ap.add_argument("--small", default="", help='比較用の口座の Params の差分（JSON）。100万円の口座で凍結した版と並べる')
    ap.add_argument("--confirm", default="", help='候補の Params の差分（JSON）。凍結した版と同じ条件で並べて確認する')
    a = ap.parse_args(argv)
    only = set(a.only.split(","))
    from kabuscout.config import load_config
    from . import live
    cfg = load_config(Path(a.config))
    mp = json.loads(a.model_params) if a.model_params else {}
    s = live.load_settings()
    p0 = Params(**s["strategy"])
    t0 = time.time()
    hs = (2, 3, 5) if "B" in only else (5,)
    ds, market, masks = D.build(cfg, horizons=hs)
    elig = np.zeros(masks["shape"], dtype=bool)
    elig[masks["ii"], masks["jj"]] = ds["elig"].to_numpy()
    feats = F.FEATURES

    def scores(H, params=None, seeds=(0,)):
        parts = []
        for off in range(3):
            ds["_tmp"] = M.walk_forward(ds, feats, f"y{H}", H, 2008, params={**mp, **(params or {})}, seeds=seeds,
                                        sample_every=3, offset=off)
            parts.append(D.to_wide(ds, "_tmp", masks))
        return [np.nanmean(np.stack(parts), axis=0)] + parts

    def runs(Sl, p):
        return [sim.run(market, S, elig, p, start="2008-01-01") for S in Sl]

    S5 = scores(5)
    if a.small:
        small(market, S5[0], elig, p0, replace(p0, **json.loads(a.small)))
        print(f"IMPROVE TOTAL {time.time() - t0:.0f}s")
        return
    if a.confirm:
        confirm(market, S5[0], elig, p0, replace(p0, **json.loads(a.confirm)))
        print(f"IMPROVE TOTAL {time.time() - t0:.0f}s")
        return
    frozen = runs(S5, p0)
    print(f"FROZEN {json.dumps(p0.to_dict())} ({time.time() - t0:.0f}s)")
    print(f"   design {_fmt(summary(frozen[0])['design'])}")
    print(f"   2019+  {_fmt(summary(frozen[0])['hold'])}")
    print(f"   all    {_fmt(summary(frozen[0])['all'])}")
    # 規模の現実性: 複利で口座が大きくなると、1注文が小型株の売買代金に対して大きすぎる（値段を動かす）。上限を付けて測る
    for frac in (0.01, 0.03):
        r = sim.run(market, S5[0], elig, replace(p0, max_adv_frac=frac), start="2008-01-01")
        print(f"LIQ cap {frac:.0%} of 20d turnover: design {_fmt(summary(r)['design'])}")
        print(f"LIQ cap {frac:.0%} of 20d turnover: 2019+  {_fmt(summary(r)['hold'])}")
        print(f"LIQ cap {frac:.0%} of 20d turnover: all    {_fmt(summary(r)['all'])}")
    for cap in (3_000_000, 10_000_000):  # 複利をやめ、この額を超えた分は毎年末に引き出すとみなす（1注文の大きさを抑える）
        r = sim.run(market, S5[0], elig, p0, start="2008-01-01", capital=1_000_000, withdraw_above=cap)
        print(f"LIQ withdraw above {cap:,}: all {_fmt(summary(r)['all'])} | 2019+ {_fmt(summary(r)['hold'])}")
    tr = frozen[0]["trades"]
    lr = np.log1p(tr["net"]).sort_values(ascending=False)
    for q in (0.01, 0.05):
        k = max(1, int(len(lr) * q))
        print(f"CONC top {q:.0%} trades by return: share of total log-return {lr.iloc[:k].sum() / lr.sum():.1%}; "
              f"rest mean {np.expm1(lr.iloc[k:].mean()):+.2%} per trade")
    passed = []
    if "A" in only:
        for reb, keep in ((1, 30), (1, 50), (2, 30), (3, 30)):
            p = replace(p0, rebalance_every=reb, keep_rank=keep)
            if judge(f"A reb={reb} keep={keep}", runs(S5, p), frozen):
                passed.append(f"A reb={reb} keep={keep}")
    if "B" in only:
        for H in (2, 3):
            SH = scores(H)
            if judge(f"B H={H}", runs(SH, p0), frozen):
                passed.append(f"B H={H}")
            # H と H=5 の平均（間引き方ごとに平均）
            mix = [np.nanmean(np.stack([x, y]), axis=0) for x, y in zip(SH, S5)]
            if judge(f"B H={H}+5", runs(mix, p0), frozen):
                passed.append(f"B H={H}+5")
            for reb in (1, 2):
                if judge(f"B H={H} reb={reb}", runs(SH, replace(p0, rebalance_every=reb)), frozen):
                    passed.append(f"B H={H} reb={reb}")
            print(f"  ({time.time() - t0:.0f}s)")
    if "C" in only:
        SC = scores(5, params=dict(max_features=0.6), seeds=(0, 1))
        if judge("C max_features=0.6 x2 seeds", runs(SC, p0), frozen):
            passed.append("C")
    print("PASSED " + json.dumps(passed, ensure_ascii=False))
    print(f"IMPROVE TOTAL {time.time() - t0:.0f}s")
