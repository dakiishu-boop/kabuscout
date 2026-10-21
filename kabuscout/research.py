"""研究ループ: 過去データで戦略の条件を自動調整する（市場が動くのを待たずに学習を回す部分）。

同じ過去データで何百通りも試すと「たまたま過去にだけ当てはまった条件」が必ず見つかる。
それを本番に持ち込まないため、次の4つの安全装置を通った条件だけを採用する。

1. 金庫（vault）: 直近 vault_days 営業日は研究中に一切見ない。最後に1回だけ、
   研究で選ばれた条件を金庫のデータで試し、現行の条件に負けたら不採用。
   金庫を開けた回数も記録し、開けるほど合格ラインを上げる（覗き見の積み重ねを罰する）。
2. 年ごとの安定性: 「全期間の平均」ではなく「平常時の年ごとの成績の中央値」で比べ、
   さらに過半の年でプラスであることを求める（一部の年だけの大当たりを排除）。
3. 多重検定の補正: これまでに試した条件の累計数が多いほど、現行条件との差の
   統計的な有意性（年ごとの差のt値）に厳しい基準を課す（Bonferroni補正）。
4. 危機局面は特例: リーマン・コロナ等の危機局面（regimes.py）は年ごとの採点から外し、
   別枠で「現行より大きく悪化していないこと」だけを確認する。危機時の極端な値動きに合わせて
   条件を最適化しないため。

取引コスト（手数料・スプレッド・滑り）は1往復 cost として差し引いて評価する。

売買ルール（利確・損切の幅と最長保有日数）も同じ安全装置で探す（plan_search）。
保有日数が違うと1回あたりの損益は比べられないので、「1枠あたりの年率換算」
（期間中の損益合計 ÷ 保有日数合計 × 250）で比べる。
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from statistics import NormalDist

import numpy as np
import pandas as pd

from . import learner, pool, regimes
from .data import PriceStore
from .strategies import STRATEGIES, apply_flags, load_params

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS research_log(
  run_at TEXT, strategy TEXT, trials INTEGER, cum_trials INTEGER, incumbent TEXT, candidate TEXT,
  inc_score REAL, cand_score REAL, t_stat REAL, t_needed REAL, vault_inc REAL, vault_cand REAL,
  vault_n INTEGER, vault_uses INTEGER, adopted INTEGER, reason TEXT
);
"""


def _yearly(net: np.ndarray, year_idx: np.ndarray, n_years: int, sel: np.ndarray, min_n: int):
    cnt = np.bincount(year_idx[sel], minlength=n_years)
    tot = np.bincount(year_idx[sel], weights=net[sel], minlength=n_years)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = tot / cnt
    mean[cnt < min_n] = np.nan
    return mean, cnt


def _sample(space: dict, rng, around: dict | None = None, scale: float = 0.25) -> dict:
    p = {}
    for k, (lo, hi, _d, _l) in space.items():
        if around is None:
            v = rng.uniform(lo, hi)
        else:
            v = float(np.clip(around[k] + rng.normal(0, (hi - lo) * scale), lo, hi))
        p[k] = round(v) if isinstance(lo, int) and isinstance(hi, int) else round(v, 4)
    return p


def _score(yearly: np.ndarray, min_years: int):
    valid = yearly[~np.isnan(yearly)]
    if len(valid) < min_years:
        return None
    return dict(median=float(np.median(valid)), pos=float((valid > 0).mean()), years=int(len(valid)))


def run(con, cfg: dict, trials: int | None = None, budget_min: float | None = None, seed: int | None = None) -> dict:
    rc = cfg["research"]
    trials = trials or rc["trials_per_strategy"]
    budget = 60 * (budget_min or rc["budget_min"])
    con.executescript(SCHEMA)
    t0 = time.time()
    rng = np.random.default_rng(seed if seed is not None else int(datetime.now().timestamp()))
    from .pipeline import params_path
    ppath = params_path(cfg)
    params = load_params(ppath)

    pl = pool.load(cfg)
    if pl is None:
        raise RuntimeError("シグナル表がありません。先に daily を実行してください")
    H = cfg["labeling"]["horizon_days"]
    d = pl[pl["complete"]].reset_index(drop=True)
    d["net"] = d["ret"] - rc["cost"]
    dates = np.array(sorted(d["date"].unique()))
    vault_start = dates[-rc["vault_days"]]
    search_end = dates[-rc["vault_days"] - H]  # ラベル期間が金庫にかからないように間を空ける
    S = d[d["date"] < search_end].reset_index(drop=True)
    V = d[d["date"] >= vault_start].reset_index(drop=True)
    years = sorted(S["date"].dt.year.unique())
    yidx = (S["date"].dt.year - years[0]).to_numpy()
    normal = S["crisis"].to_numpy() == 0
    crisis = ~normal
    log.info("研究ループ: 探索期間 %s〜%s（%d件）/ 金庫 %s〜（%d件）", str(dates[0])[:10], str(search_end)[:10],
             len(S), str(vault_start)[:10], len(V))

    prev = pd.read_sql("SELECT strategy, SUM(trials) AS n FROM research_log GROUP BY strategy", con)
    cum_prev = dict(zip(prev["strategy"], prev["n"]))
    vault_uses = con.execute("SELECT COUNT(*) FROM research_log WHERE vault_n > 0 AND run_at >= ?",
                             (str(vault_start)[:10],)).fetchone()[0]
    results, changed = [], False
    per_strategy_budget = budget / len(STRATEGIES)

    for s in STRATEGIES:
        st0 = time.time()
        loose = s.loose()
        sub_m = s.detect(S, loose).fillna(False).to_numpy()
        Ssub = S[sub_m]
        ys, net, nm, cr = yidx[sub_m], Ssub["net"].to_numpy(), normal[sub_m], crisis[sub_m]
        Vsub = V[s.detect(V, loose).fillna(False).to_numpy()]

        def evaluate(p):
            m = s.detect(Ssub, p).fillna(False).to_numpy()
            yearly, cnt = _yearly(net, ys, len(years), m & nm, rc["min_trades_per_year"])
            sc = _score(yearly, rc["min_years"])
            cm = net[m & cr]
            return sc, yearly, (float(cm.mean()) if len(cm) >= 30 else None), int(m.sum())

        inc = params[s.key]
        inc_sc, inc_y, inc_cr, inc_n = evaluate(inc)
        best = None
        n_done = 0
        for t in range(trials):
            if time.time() - st0 > per_strategy_budget:
                break
            around = best[0] if (best and t % 2) else (inc if t % 4 == 1 else None)
            cand = _sample(s.space, rng, around)
            sc, yearly, cr_mean, n = evaluate(cand)
            n_done += 1
            # 年ごとの安定性: 現行より「プラスの年の割合」が下がる条件は不採用（現行が基準以上なら基準も満たすこと）
            if sc is None or sc["pos"] < min(rc["min_positive_years"], inc_sc["pos"] if inc_sc else 1):
                continue
            if inc_cr is not None and cr_mean is not None and cr_mean < inc_cr - rc["crisis_tolerance"]:
                continue  # 危機局面で現行より大きく悪化する条件は不採用
            if best is None or sc["median"] > best[1]["median"]:
                best = (cand, sc, yearly, cr_mean)

        cum = int(cum_prev.get(s.key, 0) + n_done)
        row = dict(strategy=s.key, name=s.name, trials=n_done, cum_trials=cum, incumbent=inc,
                   profitable=bool(inc_sc and inc_sc["median"] > 0 and inc_sc["pos"] >= rc["min_positive_years"]),
                   inc_score=inc_sc and inc_sc["median"], inc_pos=inc_sc and inc_sc["pos"], inc_n=inc_n,
                   candidate=None, cand_score=None, t_stat=None, t_needed=None,
                   vault_inc=None, vault_cand=None, vault_n=0, adopted=False, reason="")
        if best is None or inc_sc is None:
            row["reason"] = "条件を満たす候補なし" if inc_sc else "現行条件のデータ不足"
        else:
            cand, sc, yearly, _ = best
            both = ~np.isnan(yearly) & ~np.isnan(inc_y)
            diff = yearly[both] - inc_y[both]
            t_needed = NormalDist().inv_cdf(1 - rc["alpha"] / max(cum, 1))
            t = float(diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff)))) if len(diff) >= 3 and diff.std(ddof=1) > 0 else 0.0
            row.update(candidate=cand, cand_score=sc["median"], t_stat=t, t_needed=t_needed)
            if t < t_needed:
                row["reason"] = f"改善が偶然の範囲（t={t:.2f} < 必要{t_needed:.2f}）"
            else:
                # 金庫を開ける（この条件で1回だけ）
                vm_c = s.detect(Vsub, cand).fillna(False).to_numpy()
                vm_i = s.detect(Vsub, inc).fillna(False).to_numpy()
                vn = Vsub["net"].to_numpy()
                vc = float(vn[vm_c].mean()) if vm_c.sum() else None
                vi = float(vn[vm_i].mean()) if vm_i.sum() else 0.0
                margin = rc["vault_margin"] * vault_uses
                row.update(vault_inc=vi, vault_cand=vc, vault_n=int(vm_c.sum()))
                vault_uses += 1
                if vm_c.sum() < rc["vault_min_trades"]:
                    row["reason"] = f"金庫期間の取引が少なすぎる（{int(vm_c.sum())}件）"
                elif vc < vi + margin:
                    row["reason"] = f"金庫期間で現行に負け（{vc:+.2%} vs {vi:+.2%}）"
                else:
                    row.update(adopted=True, reason=f"採用: 年ごとの中央値 {inc_sc['median']:+.2%}→{sc['median']:+.2%}、金庫でも {vi:+.2%}→{vc:+.2%}"
                                                    + ("" if sc["median"] > 0 else "（ただしコスト控除後はまだマイナス）"))
                    params[s.key] = cand
                    changed = True
        log.info("  %s: %d通り試行（累計%d）→ %s", s.name, n_done, cum, row["reason"])
        con.execute("INSERT INTO research_log VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
            datetime.now().isoformat(timespec="seconds"), s.key, n_done, cum, json.dumps(inc),
            json.dumps(row["candidate"]) if row["candidate"] else None, row["inc_score"], row["cand_score"],
            row["t_stat"], row["t_needed"], row["vault_inc"], row["vault_cand"], row["vault_n"], vault_uses,
            int(row["adopted"]), row["reason"]))
        results.append(row)
    con.commit()

    saved = json.loads(ppath.read_text()) if ppath.exists() else {}
    plan_res = plan_search(con, cfg, pl, params, vault_start, rng, vault_uses)
    plan_changed = plan_res["adopted"]
    if changed or plan_changed:
        hist = saved.get("history", [])
        adopted = {r["strategy"]: r["candidate"] for r in results if r["adopted"]}
        if plan_changed:
            adopted["売買ルール"] = plan_res["candidate"]
        hist.append(dict(at=datetime.now().isoformat(timespec="seconds"), adopted=adopted))
        out = dict(params=params, history=hist)
        plan = plan_res["candidate"] if plan_changed else saved.get("plan")
        if plan:
            out["plan"] = plan
        ppath.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    summary = dict(run_at=datetime.now().isoformat(timespec="seconds"), search_from=str(dates[0])[:10],
                   search_to=str(search_end)[:10], vault_from=str(vault_start)[:10], cost=rc["cost"],
                   minutes=round((time.time() - t0) / 60, 1), changed=changed, plan_changed=plan_changed,
                   plan=plan_res, results=results,
                   eras=era_table(S, params))
    (cfg["data"]["db_path"].parent / "research.json").write_text(json.dumps(summary, ensure_ascii=False, default=float))
    return summary


def era_table(S: pd.DataFrame, params: dict) -> list[dict]:
    """現行条件の戦略ごと・時代ごとの成績（平常時と危機局面を分けて）。"""
    era = regimes.era_of(S["date"])
    out = []
    for s in STRATEGIES:
        m = s.detect(S, params[s.key]).fillna(False).to_numpy()
        for e in dict.fromkeys(era):
            for flag, lab in ((0, "平常時"), (1, "危機局面")):
                sel = m & (era == e) & (S["crisis"].to_numpy() == flag)
                if sel.sum() < 20:
                    continue
                out.append(dict(strategy=s.key, era=e, regime=lab, n=int(sel.sum()),
                                win=float(S["label"].to_numpy()[sel].mean()), net=float(S["net"].to_numpy()[sel].mean())))
    return out


# ---------------------------------------------------------------- 売買ルールの研究
def _annual(net, days, yidx, n_years, sel, min_n):
    """年ごとの「1枠あたりの年率換算」= 損益合計 ÷ 保有日数合計 × 250。"""
    cnt = np.bincount(yidx[sel], minlength=n_years)
    tot = np.bincount(yidx[sel], weights=net[sel], minlength=n_years)
    dsum = np.bincount(yidx[sel], weights=days[sel], minlength=n_years)
    with np.errstate(invalid="ignore", divide="ignore"):
        a = tot / dsum * 250
    a[cnt < min_n] = np.nan
    return a


def _pooled(net, days, sel):
    return float(net[sel].sum() / days[sel].sum() * 250) if sel.sum() and days[sel].sum() > 0 else None


def plan_search(con, cfg: dict, pl: pd.DataFrame, params: dict, vault_start, rng, vault_uses: int) -> dict:
    rc, lab = cfg["research"], cfg["labeling"]
    grid = rc["plan_grid"]
    inc = dict(tp_atr=lab["tp_atr"], sl_atr=lab["sl_atr"], horizon_days=lab["horizon_days"])
    plans = [dict(tp_atr=t, sl_atr=sl, horizon_days=h) for t in grid["tp_atr"] for sl in grid["sl_atr"] for h in grid["horizon_days"]]
    if inc not in plans:
        plans.append(inc)
    Hmax = max(p["horizon_days"] for p in plans)

    # 現在の戦略条件で拾われるシグナル全体で評価（売買ルールは全戦略共通）
    sig = apply_flags(pl[["date", "code", "atr", "crisis"] + _flag_inputs(pl)], params)[["date", "code", "atr", "crisis"]]
    codes = sorted(sig["code"].unique())
    px = PriceStore(cfg["data"]["price_dir"]).load(codes=codes)
    c = px["close"].ffill(limit=5)
    o, h, l = (px[k].fillna(c) for k in ("open", "high", "low"))
    tix = {d: i for i, d in enumerate(c.index)}
    cix = {k: i for i, k in enumerate(c.columns)}
    keep = sig["date"].map(tix).notna() & sig["code"].map(cix).notna()
    sig = sig[keep].reset_index(drop=True)
    ti = sig["date"].map(tix).to_numpy(int)
    ci = sig["code"].map(cix).to_numpy(int)
    paths = learner.forward_paths(*(x.to_numpy("float32") for x in (o, h, l, c)), ti, ci, Hmax)
    atr = sig["atr"].to_numpy("float64")

    dates = np.array(sorted(sig["date"].unique()))
    search_end = dates[max(0, np.searchsorted(dates, vault_start) - Hmax)]
    in_s = (sig["date"] < search_end).to_numpy()
    in_v = (sig["date"] >= vault_start).to_numpy()
    years = sorted(sig.loc[in_s, "date"].dt.year.unique())
    yidx = (sig["date"].dt.year - years[0]).clip(lower=0).to_numpy()
    normal = sig["crisis"].to_numpy() == 0
    log.info("売買ルールの研究: %d通り × %d件のシグナル", len(plans), int(in_s.sum()))

    evals = []
    for p in plans:
        r = learner.outcomes(paths, atr, p["tp_atr"], p["sl_atr"], p["horizon_days"])
        ok = r["complete"]
        net = np.where(ok, r["ret"] - rc["cost"], 0.0)
        days = np.where(ok, r["exit_days"], 0.0)
        yearly = _annual(net, days, yidx, len(years), ok & in_s & normal, rc["min_trades_per_year"])
        sc = _score(yearly, rc["min_years"])
        evals.append(dict(plan=p, score=sc, yearly=yearly,
                          crisis=_pooled(net, days, ok & in_s & ~normal),
                          vault=_pooled(net, days, ok & in_v), vault_n=int((ok & in_v).sum()),
                          win=float(r["label"][ok & in_s].mean()) if (ok & in_s).any() else None,
                          avg_days=float(days[ok & in_s].mean()) if (ok & in_s).any() else None))
    inc_e = next(e for e in evals if e["plan"] == inc)
    inc_sc = inc_e["score"]

    prev = con.execute("SELECT COALESCE(SUM(trials),0) FROM research_log WHERE strategy='_plan'").fetchone()[0]
    cum = int(prev + len(plans))
    cands = [e for e in evals if e["plan"] != inc and e["score"]
             and e["score"]["pos"] >= min(rc["min_positive_years"], inc_sc["pos"] if inc_sc else 1)
             and not (inc_e["crisis"] is not None and e["crisis"] is not None
                      and e["crisis"] < inc_e["crisis"] - rc["plan_crisis_tolerance"])]
    best = max(cands, key=lambda e: e["score"]["median"]) if cands else None
    res = dict(incumbent=inc, inc_score=inc_sc and inc_sc["median"], inc_pos=inc_sc and inc_sc["pos"],
               inc_win=inc_e["win"], inc_days=inc_e["avg_days"], trials=len(plans), cum_trials=cum,
               candidate=None, cand_score=None, t_stat=None, t_needed=None, vault_inc=None, vault_cand=None,
               vault_n=0, adopted=False, reason="",
               top=[dict(plan=e["plan"], median=e["score"]["median"], pos=e["score"]["pos"], win=e["win"],
                         days=e["avg_days"], crisis=e["crisis"])
                    for e in sorted([e for e in evals if e["score"]], key=lambda e: -e["score"]["median"])[:12]])
    if best is None or inc_sc is None:
        res["reason"] = "条件を満たす候補なし"
    else:
        both = ~np.isnan(best["yearly"]) & ~np.isnan(inc_e["yearly"])
        diff = best["yearly"][both] - inc_e["yearly"][both]
        t_needed = NormalDist().inv_cdf(1 - rc["alpha"] / max(cum, 1))
        t = float(diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff)))) if len(diff) >= 3 and diff.std(ddof=1) > 0 else 0.0
        res.update(candidate=best["plan"], cand_score=best["score"]["median"], t_stat=t, t_needed=t_needed)
        if t < t_needed:
            res["reason"] = f"改善が偶然の範囲（t={t:.2f} < 必要{t_needed:.2f}）"
        else:
            vc, vi = best["vault"], inc_e["vault"] or 0.0
            margin = rc["plan_vault_margin"] * vault_uses
            res.update(vault_inc=vi, vault_cand=vc, vault_n=best["vault_n"])
            vault_uses += 1
            if best["vault_n"] < rc["vault_min_trades"] or vc is None:
                res["reason"] = f"金庫期間の取引が少なすぎる（{best['vault_n']}件）"
            elif vc < vi + margin:
                res["reason"] = f"金庫期間で現行に負け（年率換算 {vc:+.1%} vs {vi:+.1%}）"
            else:
                res.update(adopted=True, reason=f"採用: 年率換算の中央値 {inc_sc['median']:+.1%}→{best['score']['median']:+.1%}、"
                                                 f"金庫でも {vi:+.1%}→{vc:+.1%}")
    log.info("  売買ルール: %d通り（累計%d）→ %s", len(plans), cum, res["reason"])
    con.execute("INSERT INTO research_log VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
        datetime.now().isoformat(timespec="seconds"), "_plan", len(plans), cum, json.dumps(inc),
        json.dumps(res["candidate"]) if res["candidate"] else None, res["inc_score"], res["cand_score"],
        res["t_stat"], res["t_needed"], res["vault_inc"], res["vault_cand"], res["vault_n"], vault_uses,
        int(res["adopted"]), res["reason"]))
    con.commit()
    return res


def _flag_inputs(pl: pd.DataFrame) -> list[str]:
    """戦略判定に必要な列（シグナル表のうち日付・銘柄・ATR・危機フラグ以外の特徴量）。"""
    return [c for c in pl.columns if c not in ("date", "code", "atr", "crisis")]
