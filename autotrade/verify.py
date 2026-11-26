"""儲けが本物かの検証（ユーザーの依頼「儲けが確かなものか確認する」、2026-11-26）。

1回のジョブで、本番の形（3つのモデルの平均 × config.yaml の売買の形）について次を確かめ、結果をログに出す:
 1. 取引の点検: 大きく勝った・負けた取引にデータの誤り（分割の調整漏れ・売買不成立の日の約定）が無いか。利益の偏り
 2. 売買が成立しない日（出来高0・高値=安値）には約定しない、とした場合
 3. 単純な並べ替え（値動きの小ささ・配当利回り）との比較 = AIが何を足しているか
 4. 2018年10月より前は全銘柄1,000株単位、とした場合（下限の見積もり）
 5. 株価の異常値の除去を「過去の値だけ」で行った場合（先読みを外す）
 6. 実際に上場廃止になった銘柄（J-Quants・直近約2年）を混ぜた場合
 週ごとの資産の推移を WEEKLY-JSON としてログに出す（kabuscout との対の検定は手元で行う）。
"""
from __future__ import annotations

import json
import logging
import time

import numpy as np
import pandas as pd

from . import data as D
from . import features as F
from . import model as M
from . import sim
from .research import HOLDOUT, _fmt
from .strategy import Params

log = logging.getLogger("autotrade.verify")


def _line(tag: str, res: dict) -> None:
    print(f"{tag} all   {_fmt(sim.stats(res))}")
    print(f"{tag} 2019+ {_fmt(sim.stats(res, HOLDOUT))}")
    d08 = res["equity"][res["equity"].index <= pd.Timestamp("2018-12-31")]
    if len(d08) > 10:
        print(f"{tag} 08-18 {_fmt(sim.stats(dict(equity=d08, trades=res['trades'][res['trades']['date'] <= pd.Timestamp('2018-12-31')], exposure=res['exposure'][d08.index])))}")


def ensemble_scores(ds, masks, feats, H, offsets=(0, 1, 2), params=None):
    S = []
    for off in offsets:
        sc = M.walk_forward(ds, feats, f"y{H}", H, 2008, params=params, sample_every=3, offset=off)
        ds["_tmp"] = sc
        S.append(D.to_wide(ds, "_tmp", masks))
    return np.nanmean(np.stack(S), axis=0), S


def audit(market, res: dict) -> None:
    """取引の点検。"""
    tr = res["trades"].copy()
    cix = {c: j for j, c in enumerate(market.codes)}
    cal = market.cal
    tr = tr[tr["entry_date"].notna()]
    pnl = tr["pnl"].sort_values(ascending=False)
    tot = tr["pnl"].sum()
    n = len(pnl)
    print(f"AUDIT trades {n} total pnl {tot:,.0f}")
    for q in (0.01, 0.05):
        k = max(1, int(n * q))
        print(f"AUDIT top {q:.0%} ({k} trades) pnl share {pnl.iloc[:k].sum() / tot:.1%}; "
              f"without them total pnl {(tot - pnl.iloc[:k].sum()):,.0f}")
    print(f"AUDIT |net|>40%: {(tr['net'].abs() > 0.4).sum()} trades, pnl {tr.loc[tr['net'].abs() > 0.4, 'pnl'].sum():,.0f}")

    def flags(r):
        j = cix[r["code"]]
        a, b = cal.searchsorted(r["entry_date"]), cal.searchsorted(r["date"])
        sp = market.splits[a:b + 1, j]
        spl = [float(x) for x in sp if x == x and x not in (0, 1)]
        out = []
        if spl:
            out.append(f"split{spl}")
        if market.volume is not None:
            if market.volume[a, j] <= 0:
                out.append("vol0@entry")
            if market.high[a, j] <= market.low[a, j]:
                out.append("H=L@entry")
            if market.volume[b, j] <= 0 or market.high[b, j] <= market.low[b, j]:
                out.append("locked@exit")
        c = market.close[a:b + 1, j]
        rr = np.abs(np.diff(np.log(c[np.isfinite(c)]))) if np.isfinite(c).sum() > 1 else np.array([0])
        if len(rr) and rr.max() > np.log(1.3):
            out.append(f"jump{np.exp(rr.max()) - 1:+.0%}")
        return ",".join(out)

    tr["flags"] = tr.apply(flags, axis=1)
    for title, part in (("TOP", tr.nlargest(20, "net")), ("BOTTOM", tr.nsmallest(20, "net"))):
        for _, r in part.iterrows():
            print(f"AUDIT {title} {r['code']} {r['entry_date'].date()}→{r['date'].date()} net {r['net']:+.1%} "
                  f"pnl {r['pnl']:+,.0f} exit_px {r['px']:,.1f} {r['reason']} {r['flags']}")
    fl = tr[tr["flags"].str.contains("split|vol0|H=L|locked")]
    print(f"AUDIT flagged (split/vol0/H=L/locked) {len(fl)} trades, pnl {fl['pnl'].sum():,.0f} ({fl['pnl'].sum() / tot:.1%} of total)")
    print(f"AUDIT flagged by kind: " + json.dumps({k: int(tr['flags'].str.contains(k).sum()) for k in ('split', 'vol0', 'H=L', 'locked', 'jump')}))
    sp = tr[tr["flags"].str.contains("split")]
    print(f"AUDIT trades holding through a split: {len(sp)}, mean net {sp['net'].mean():+.2%} vs all {tr['net'].mean():+.2%}")


def trailing_clean(panel):
    """kabuscout.data.clean_panel の先読みなし版（前後ではなく、その日までの7日の中央値と比べる）。"""
    c = panel["close"]
    med = c.rolling(7, min_periods=3).median().shift(1)
    ratio = (c / med).abs()
    bad = (ratio > 2.5) | (ratio < 0.4) | (c <= 0)
    n = int(bad.to_numpy().sum())
    log.info("異常値を除去（先読みなし）: %d 本", n)
    return {k: v.mask(bad) for k, v in panel.items()} if n else panel


def delisted_check(cfg, ds_main, feats, H, p: Params) -> None:
    """実際に上場廃止になった銘柄（J-Quants・直近約2年）を混ぜて、2025年以降を売買し直す。"""
    from pathlib import Path
    bars_p = Path(cfg["data"]["db_path"]).parent / "jquants" / "delisted_bars.parquet"
    lst_p = Path(cfg["data"]["db_path"]).parent / "jquants" / "delisted_list.parquet"
    if not bars_p.exists():
        print("DELIST no data")
        return
    b = pd.read_parquet(bars_p)
    b = b[b["Date"].notna()]
    b["code"] = b["Code"].astype(str).str[:4]
    b["date"] = pd.to_datetime(b["Date"])
    for k in ("O", "H", "L", "C", "Vo"):
        b[k] = pd.to_numeric(b[k], errors="coerce")
    names = {}
    if lst_p.exists():
        l = pd.read_parquet(lst_p)
        names = dict(zip(l["Code"].astype(str).str[:4], l["CoName"]))
    y_last = pd.Timestamp(ds_main["date"].max()).year
    years = (y_last - 1, y_last)
    start = f"{y_last - 3}-01-01"
    px, mkt, fac, splits, div = D.load_panel(cfg, start)
    cal = px["close"].index
    gone = sorted(set(b["code"]) - set(px["close"].columns))
    print(f"DELIST listed in J-Quants delisted list: {b['code'].nunique()} codes; not in Yahoo data: {len(gone)}; "
          f"bars {b['date'].min().date()}〜{b['date'].max().date()}")
    w = {k: b[b["code"].isin(gone)].pivot_table(index="date", columns="code", values=v).reindex(cal)
         for k, v in (("open", "O"), ("high", "H"), ("low", "L"), ("close", "C"), ("volume", "Vo"))}
    days = w["close"].notna().sum()
    print(f"DELIST codes with >=225 days of bars (can become eligible): {(days >= 225).sum()} / {len(gone)}")
    aug = {k: pd.concat([px[k], w[k].astype("float32")], axis=1) for k in w}
    fac2 = pd.concat([fac, pd.DataFrame(1.0, index=cal, columns=gone, dtype="float32")], axis=1)
    sp2 = pd.concat([splits, pd.DataFrame(np.nan, index=cal, columns=gone)], axis=1) if splits is not None else None
    div2 = pd.concat([div, pd.DataFrame(np.nan, index=cal, columns=gone)], axis=1) if div is not None else None
    ds2, mk2, ms2 = D.build_core(cfg, aug, mkt, fac2, sp2, div2, horizons=(H,))
    # 2025年・2026年の予測に使うモデル（ウォークフォワードと同じく、その年より前のデータだけで学ぶ・3つの平均）
    dates = np.sort(ds_main["date"].unique())
    sc = pd.Series(np.nan, index=ds2.index)
    for y in years:
        before = dates[dates < np.datetime64(f"{y}-01-01")]
        purge = before[-(H + 2)]
        tr = ds_main[(ds_main["date"] < purge) & ds_main["elig"]]
        models = [m for off in range(3) for m in M.fit(tr, feats, f"y{H}", sample_every=3, offset=off)]
        te = (ds2["date"] >= np.datetime64(f"{y}-01-01")) & (ds2["date"] < np.datetime64(f"{y + 1}-01-01"))
        sc[te] = M.predict(models, ds2[te], feats)
    ds2["s"] = sc
    S2 = D.to_wide(ds2, "s", ms2)
    E2 = np.zeros(ms2["shape"], bool)
    E2[ms2["ii"], ms2["jj"]] = ds2["elig"].to_numpy()
    gi = [mk2.codes.index(c) for c in gone]
    S_surv, E_surv = S2.copy(), E2.copy()
    S_surv[:, gi], E_surv[:, gi] = np.nan, False
    s0 = f"{years[0]}-01-01"
    print(f"DELIST delisted rows eligible in {years[0]}+: {int(E2[:, gi][cal >= pd.Timestamp(s0)].sum())}")
    d2 = ds2[ds2["code"].isin(gone) & ds2["s"].notna()]
    if len(d2):
        rk = ds2[ds2["s"].notna()].groupby("date")["s"].rank(ascending=False)
        top = ds2.loc[rk.index[rk <= 30]]
        top = top[top["code"].isin(gone)]
        for code, g in top.groupby("code"):
            j = mk2.codes.index(code)
            last = w["close"][code].dropna()
            first = g["date"].min()
            print(f"DELIST top30 {code} {names.get(code, '')}: {len(g)} days in top30 from {first.date()}, "
                  f"close then {w['close'][code].get(first):,.1f} → last {last.iloc[-1]:,.1f} on {last.index[-1].date()} "
                  f"({last.iloc[-1] / w['close'][code].get(first) - 1:+.1%})")
    for tag, S_, E_ in (("with delisted", S2, E2), ("survivors only", S_surv, E_surv)):
        r = sim.run(mk2, S_, E_, p, start=s0)
        st = sim.stats(r)
        bought = {f.code for f in r["broker"].fills if f.side == "buy"} & set(gone)
        print(f"DELIST sim {s0}〜 [{tag}] {_fmt(st)} | delisted names bought: {sorted(bought)}")
        for f in r["broker"].fills:
            if f.code in gone and f.side == "sell":
                print(f"DELIST   trade {f.code} {names.get(f.code, '')} {f.entry_date}→{f.date} net {f.net:+.1%} {f.reason}")


def main(argv: list[str]) -> None:
    import argparse
    import warnings
    from pathlib import Path
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--model-params", default="")
    ap.add_argument("--skip", default="")
    a = ap.parse_args(argv)
    skip = set(a.skip.split(","))
    from kabuscout.config import load_config
    from . import live
    cfg = load_config(Path(a.config))
    mp = json.loads(a.model_params) if a.model_params else None
    s = live.load_settings()
    p = Params(**s["strategy"])
    H = s["model"]["horizons"][0]
    feats = F.FEATURES
    t0 = time.time()
    ds, market, masks = D.build(cfg, horizons=(H,))
    elig = np.zeros(masks["shape"], dtype=bool)
    elig[masks["ii"], masks["jj"]] = ds["elig"].to_numpy()
    S, parts = ensemble_scores(ds, masks, feats, H, params=mp)
    print(f"VERIFY config {json.dumps(p.to_dict())} H={H} built+scored {time.time() - t0:.0f}s")
    base = sim.run(market, S, elig, p, start="2008-01-01")
    _line("BASE", base)
    eq = base["equity"].resample("W-FRI").last().dropna()
    print("WEEKLY-JSON " + json.dumps([[d.strftime("%Y-%m-%d"), round(float(v))] for d, v in eq.items()]))
    for k, Sx in enumerate(parts):
        _line(f"OFFSET{k}", sim.run(market, Sx, elig, p, start="2008-01-01"))
    # 1. 取引の点検
    audit(market, base)
    # 2. 売買が成立しない日には約定しない
    rf = sim.run(market, S, elig, p, start="2008-01-01", realistic_fills=True)
    _line("REALFILL", rf)
    # 3. 単純な並べ替え
    for name, cols, sign in (("lowvol(ivol120)", ["ivol120"], [-1]),
                             ("lowvol+div", ["ivol120", "div_yield"], [-1, 1]),
                             ("lowvol4+div", ["ivol120", "vol60", "atr_pct", "max_ret20", "div_yield"], [-1, -1, -1, -1, 1])):
        x = sum(sg * ds[c].fillna(0.5 if c != "div_yield" else 0.0) for c, sg in zip(cols, sign))
        ds["_f"] = x.astype("float32")
        Sf = D.to_wide(ds, "_f", masks)
        _line(f"FACTOR[{name}]", sim.run(market, Sf, elig, p, start="2008-01-01"))
    # 4. 2018年10月より前は1,000株単位
    _line("LOT1000<2018-10", sim.run(market, S, elig, p, start="2008-01-01", lot_schedule=[("2018-10-01", 1000)]))
    _line("LOT1000+REALFILL", sim.run(market, S, elig, p, start="2008-01-01", lot_schedule=[("2018-10-01", 1000)],
                                      realistic_fills=True))
    # 6. 実際の上場廃止銘柄
    if "delist" not in skip:
        try:
            delisted_check(cfg, ds, feats, H, p)
        except Exception as e:
            log.exception("上場廃止銘柄の検証に失敗: %s", e)
    # 5. 異常値の除去を先読みなしで（作り直し・モデル1つ = 間引き方0 と比べる）
    if "clean" not in skip:
        import kabuscout.data as kd
        del ds
        orig = kd.clean_panel
        kd.clean_panel = trailing_clean
        try:
            ds2, market2, masks2 = D.build(cfg, horizons=(H,))
        finally:
            kd.clean_panel = orig
        e2 = np.zeros(masks2["shape"], dtype=bool)
        e2[masks2["ii"], masks2["jj"]] = ds2["elig"].to_numpy()
        S2, _ = ensemble_scores(ds2, masks2, feats, H, offsets=(0,), params=mp)
        _line("TRAILCLEAN offset0", sim.run(market2, S2, e2, p, start="2008-01-01"))
        print("(compare with OFFSET0 above)")
    print(f"VERIFY TOTAL {time.time() - t0:.0f}s")
