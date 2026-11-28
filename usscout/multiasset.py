"""§97: 株以外も使うアイデアの検証（D: 17資産の上昇トレンド、C: S&P500＋金＋日本株のAI）。docs/JOURNAL.md §97 の計画どおり。

D（17資産の上昇トレンド）:
- 本命（買いだけ）: usscout の口座のシミュレーション（strategies.TSMomentum・寄付で約定・コスト込み）
- 研究だけ（買いと売り）: 毎日の値動きから計算。各資産を年40%の値動きにそろえ（研究の原文では未確認の設定）、N で割る。
  ETF の配当込みの騰落 − 短期金利 を「先物の騰落」とみなす。判定の翌日の終値で売買（1日遅らせる＝厳しめ）
- 物差し: S&P500 を持ち続ける／値動きの大きさをそろえた S&P500／重ね合わせ（S&P500＋トレンドの超過リターン）と同じ値動きまでレバレッジをかけた S&P500
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D
from . import metrics, research
from .levtrend import stats, synth

WINDOWS = {"full": "2008-04-01", "post_paper": "2012-06-01"}


def _rf(p: D.Panel) -> pd.Series:
    """短期金利（日次）。^IRX − 0（先物の担保の利回り）。"""
    return (p.close["^IRX"].ffill() / 100 / 252).fillna(0.0)


def longshort(p: D.Panel, universe: list[str], cost: float, target_vol: float = 0.40, lookback: int = 252, vol_days: int = 63) -> dict:
    cal = D.calendar(p)
    c = p.close.reindex(cal)[universe]
    r = c.pct_change()
    rf = _rf(p).reindex(cal).fillna(0.0)
    ex = r.sub(rf, axis=0)                                      # 先物とみなした超過リターン
    me = D.month_ends(cal)
    r12 = c / c.shift(lookback) - 1
    cash12 = (1 + rf).rolling(lookback).apply(np.prod, raw=True) - 1
    sig = np.sign(r12.sub(cash12, axis=0))
    vol = r.rolling(vol_days, min_periods=vol_days).std() * np.sqrt(252)
    n = c.notna().sum(axis=1).clip(lower=1)
    w_me = (sig * (target_vol / vol)).div(n, axis=0).loc[me]
    w = w_me.reindex(cal).ffill().shift(2).fillna(0.0)          # 月末の判定 → 翌日の終値で売買 → その次の日から効く
    turn = w.diff().abs().sum(axis=1).fillna(0.0)
    R_ex = (w * ex.fillna(0.0)).sum(axis=1) - turn.shift(-1).fillna(0.0) * cost
    return {"excess": R_ex, "total": R_ex + rf, "gross": w.abs().sum(axis=1), "turnover": turn}


def _vol_matched(r_spy: pd.Series, rf: pd.Series, target_sd: float) -> pd.Series:
    k = target_sd / float(r_spy.std())
    if k <= 1:
        return k * r_spy + (1 - k) * rf
    return synth(r_spy, k, (rf * 252 + 0.005), er=0.0)       # 借りる金利 = 短期金利＋0.5%（先物並みで信託報酬なし）


def evaluate_d(cfg: dict) -> dict:
    lab = research.Lab(cfg)
    p = lab.p
    uni = list(cfg["strategies"]["ts_momentum"]["universe"])
    cal = D.calendar(p)
    rf = _rf(p).reindex(cal).fillna(0.0)
    r_spy = p.close["SPY"].reindex(cal).pct_change().fillna(0.0)
    out = {"universe": uni}
    for wn, start in WINDOWS.items():
        res = {}
        # 本命（買いだけ）: usscout の口座のシミュレーション
        for label, integer, cap in (("long_only_fractional", False, None), ("long_only_whole_shares_50k", True, 50000)):
            bt = lab.run("ts_momentum", "net_mmf", start=start, integer=integer, capital=cap)
            m = metrics.summary(bt.equity, lab.spy, bt.trades)
            r = bt.equity.pct_change().dropna()
            m.update(vol=float(r.std() * np.sqrt(252)), exposure=float(bt.exposure.mean()) if bt.exposure is not None else None,
                     after_tax=metrics.cagr(metrics.after_tax(bt.equity)))
            vm = _vol_matched(r_spy.loc[bt.equity.index[1]:], rf.loc[bt.equity.index[1]:], float(r.std()))
            m["spy_vol_matched"] = stats(vm)
            res[label] = m
        win = cal >= pd.Timestamp(start)
        res["spy_buy_hold"] = stats(r_spy[win])
        # 研究だけ（買いと売り）: 先物並みのコストと ETF のコスト
        for cl, cost in (("futures_cost", 0.0005), ("etf_cost", 0.00545)):
            ls = longshort(p, uni, cost)
            tot, exr = ls["total"][win], ls["excess"][win]
            s = stats(tot)
            s.update(avg_gross=float(ls["gross"][win].mean()), turnover_per_year=float(ls["turnover"][win].sum() / metrics.years(metrics_curve(tot))),
                     corr_spy_monthly=float(metrics_curve(tot).resample("ME").last().pct_change().corr(metrics_curve(r_spy[win]).resample("ME").last().pct_change())))
            s["spy_vol_matched"] = stats(_vol_matched(r_spy[win], rf[win], float(tot.std())))
            ov = r_spy[win] + exr                                # 重ね合わせ: S&P500 100% ＋ トレンドの超過リターン
            s["overlay"] = stats(ov)
            s["overlay_spy_vol_matched"] = stats(_vol_matched(r_spy[win], rf[win], float(ov.std())))
            s["overlay_after_tax"] = metrics.cagr(metrics.after_tax(metrics_curve(ov)))
            res[f"long_short_{cl}"] = s
        out[wn] = dict(res, start=start)
    return out


def metrics_curve(R: pd.Series) -> pd.Series:
    return (1 + R.fillna(0.0)).cumprod()


# ---- C: S&P500＋金＋日本株のAI ----
def gold_jpy(cfg: dict, refresh: bool = False) -> tuple[pd.Series, pd.Series]:
    """(金先物 − 年0.25% の円建て, 実際の 1540 の値)。data/us_scout/gold.parquet に保存。"""
    path = cfg["data"]["dir"] / "gold.parquet"
    if refresh or not path.exists():
        import yfinance as yf
        d = yf.download(["GC=F", "1540.T", "JPY=X"], start="2005-01-01", progress=False, auto_adjust=True, group_by="ticker")
        g = pd.DataFrame({k: d[k]["Close"] for k in ("GC=F", "1540.T", "JPY=X")})
        g.index = pd.to_datetime(g.index).tz_localize(None).normalize()
        g.to_parquet(path)
    g = pd.read_parquet(path).ffill()
    fee = np.exp(-0.0025 * np.arange(len(g)) / 252)
    return (g["GC=F"] * g["JPY=X"] * fee).dropna(), g["1540.T"].dropna()


def validate_gold(proxy: pd.Series, actual: pd.Series) -> dict:
    a = actual.loc[actual.index >= proxy.index[0]]
    pr = proxy.reindex(a.index, method="ffill")
    yp, ya = metrics.yearly(pr / pr.iloc[0]), metrics.yearly(a / a.iloc[0])
    gap = (yp - ya).dropna()
    return {"start": str(a.index[0].date()), "cagr_proxy": metrics.cagr(pr), "cagr_1540": metrics.cagr(a),
            "gap_mean": float(gap.mean()), "gap_abs_max": float(gap.abs().max()), "by_year": {int(k): round(float(v), 4) for k, v in gap.items()}}


def combine_n(curves: dict[str, pd.Series], weights: dict[str, float], taxed: set[str], cost_rate: dict[str, float], cfg: dict) -> pd.Series:
    """複数の口座（円・同じ日付）を毎年初めに weights へ戻しながら合わせる。taxed の口座だけ毎年末に 20.315%（損失3年繰越）。
    動かす金額に、その口座の cost_rate（片道）。コストは受け取る側から引く（combo.combine と同じ扱い）。"""
    rate, carry = cfg["tax"]["rate"], cfg["tax"]["carry_years"]
    names = [k for k in weights if weights[k] > 0] or list(weights)
    idx = next(iter(curves.values())).index
    rets = {k: curves[k].reindex(idx).ffill().pct_change().fillna(0.0) for k in names}
    total = 1.0
    v = {k: weights[k] * total for k in names}
    base = dict(v)
    losses = {k: [] for k in names}
    year, out = idx[0].year, []

    def tax(k, yr):
        gain = v[k] - base[k]
        losses[k] = [[y, l] for y, l in losses[k] if yr - y <= carry and l > 0]
        if k not in taxed:
            return 0.0
        if gain < 0:
            losses[k].append([yr, -gain])
            return 0.0
        left = gain
        for e in losses[k]:
            t = min(e[1], left)
            e[1] -= t
            left -= t
        return left * rate

    for d in idx:
        if d.year != year:
            for k in names:
                v[k] -= tax(k, year)
            tot = sum(v.values())
            moves = {k: weights[k] * tot - v[k] for k in names}
            cost = sum(abs(m) * cost_rate.get(k, 0.0) for k, m in moves.items())
            recv = sum(m for m in moves.values() if m > 0)
            for k, m in moves.items():
                v[k] += m - (cost * m / recv if m > 0 and recv > 0 else 0.0)
            base = dict(v)
            year = d.year
        for k in names:
            v[k] *= 1 + rets[k].at[d]
        out.append(sum(v.values()))
    s = pd.Series(out, index=idx)
    final_tax = sum(tax(k, year) for k in names)
    s.iloc[-1] -= final_tax
    return s


def evaluate_c(cfg: dict, jp_path: Path) -> dict:
    from .combo import load_jp_curve
    lab = research.Lab(cfg)
    jp = load_jp_curve(jp_path)
    end = min(jp.index[-1], lab.p.dates[-1])
    jp = jp.loc[:end]
    fx = lab.p.close["JPY=X"]
    gproxy, g1540 = gold_jpy(cfg)
    val = validate_gold(gproxy, g1540)
    use_1540 = abs(val["gap_mean"]) > 0.01 or val["gap_abs_max"] > 0.03
    gold = gproxy.copy()
    if use_1540:  # 1540 がある期間は実際の値をつなぐ
        a = g1540.loc[g1540.index >= gold.index[0]]
        gold = pd.concat([gold.loc[:a.index[0]] / gold.loc[:a.index[0]].iloc[-1] * a.iloc[0], a.iloc[1:]])
    c = cfg["costs"]
    out = {"validate_gold": val, "gold_series": "1540 を使用" if use_1540 else "金先物−年0.25% を使用", "windows": {}}
    mixes = {"S&P500だけ": {"sp": 1.0}, "AIだけ": {"jp": 1.0}, "金だけ": {"gold": 1.0},
             "1/3ずつ（新）": {"sp": 1 / 3, "gold": 1 / 3, "jp": 1 / 3}, "40/20/40（新）": {"sp": 0.4, "gold": 0.2, "jp": 0.4},
             "S&P80/金20（既に見た）": {"sp": 0.8, "gold": 0.2}, "S&P60/金40（既に見た）": {"sp": 0.6, "gold": 0.4},
             "S&P50/AI50（既に見た）": {"sp": 0.5, "jp": 0.5}}
    for wn, start in (("2008", None), ("2016", "2016-01-01")):
        j = jp.loc[start:] if start else jp
        eq = lab.run("hold_spy", "net_mmf", start=str(j.index[0].date()), end=str(end.date())).equity
        sp = metrics.to_jpy(eq, fx, c.get("fx_spread_jpy", 0)).reindex(j.index, method="ffill")
        gd = gold.reindex(j.index, method="ffill")
        ok = sp.notna() & gd.notna()  # 全部の値がそろった日から始める（米国の休みで最初の日が欠けることがある）
        j, sp, gd = j[ok], sp[ok], gd[ok]
        curves = {"sp": sp / sp.iloc[0], "gold": gd / gd.iloc[0], "jp": j / j.iloc[0]}
        f0 = float(fx.reindex(j.index, method="ffill").iloc[0])
        costs = {"jp": 0.001, "gold": 0.001, "sp": c.get("commission", 0) + c.get("spread", 0) + c.get("fx_spread_jpy", 0) / f0}
        rows = {}
        for name, w in mixes.items():
            w = {k: w.get(k, 0.0) for k in ("sp", "gold", "jp")}
            pre = combine_n(curves, w, set(), costs, cfg)
            post = combine_n(curves, w, {"jp"}, costs, cfg)
            r = post.pct_change().dropna()
            sh = float(r.mean() / r.std() * np.sqrt(52)) if r.std() > 0 else np.nan
            g, dd = metrics.cagr(post), metrics.max_dd(post)
            rows[name] = {"cagr_pre_tax": metrics.cagr(pre), "cagr_after_tax": g, "max_dd": dd, "worst_year": float(metrics.yearly(post).min()),
                          "sharpe": sh, "calmar": g / abs(dd) if dd < 0 else np.nan}
        out["windows"][wn] = {"start": str(j.index[0].date()), "end": str(j.index[-1].date()), "rows": rows}
    return out


def run_all(cfg: dict, jp_path: Path | None = None) -> dict:
    from .config import ROOT
    rep = {"D": evaluate_d(cfg)}
    jp = Path(jp_path or ROOT / "data" / "backtest.json")
    if jp.exists():
        rep["C"] = evaluate_c(cfg, jp)
    root = cfg["data"]["dir"]
    (root / "multiasset.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float))
    (root / "multiasset.md").write_text(markdown(rep))
    return rep


def _p(v):
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:+.1%}"


def markdown(rep: dict) -> str:
    out = ["# §97 株以外も使うアイデア", ""]
    d = rep["D"]
    for wn in WINDOWS:
        w = d[wn]
        b = w["spy_buy_hold"]
        out += [f"## D: 17資産の上昇トレンド（{w['start']}〜）", "", "| | 年率 | 最大下落 | 年率の標準偏差 | 比べる S&P500 |", "|---|---|---|---|---|",
                f"| S&P500 を持ち続ける | {_p(b['cagr'])} | {_p(b['max_dd'])} | {b['vol']:.0%} | |"]
        for k in ("long_only_fractional", "long_only_whole_shares_50k"):
            m = w[k]
            out.append(f"| 本命・買いだけ（{'端数株' if 'fractional' in k else '1株単位・5万ドル'}）| {_p(m['cagr'])} | {_p(m['max_dd'])} | {m['vol']:.0%} | "
                       f"同じ値動きの S&P500 {_p(m['spy_vol_matched']['cagr'])}・{_p(m['spy_vol_matched']['max_dd'])}（税引き後 {_p(m['after_tax'])}・投資比率 {m['exposure']:.0%}・コスト/年 {m.get('cost_per_year', 0):.2%}） |")
        for k in ("long_short_futures_cost", "long_short_etf_cost"):
            m = w[k]
            out.append(f"| 研究: 買いと売り（{'先物並みのコスト' if 'futures' in k else 'ETFのコスト'}）| {_p(m['cagr'])} | {_p(m['max_dd'])} | {m['vol']:.0%} | "
                       f"同じ値動きの S&P500 {_p(m['spy_vol_matched']['cagr'])}（平均の総額 {m['avg_gross']:.1f}倍・S&P500との相関 {m['corr_spy_monthly']:+.2f}） |")
            o, ov = m["overlay"], m["overlay_spy_vol_matched"]
            out.append(f"| 研究: S&P500＋トレンドの重ね合わせ（{'先物' if 'futures' in k else 'ETF'}）| {_p(o['cagr'])} | {_p(o['max_dd'])} | {o['vol']:.0%} | "
                       f"同じ値動きまでレバレッジの S&P500 {_p(ov['cagr'])}・{_p(ov['max_dd'])} |")
        out.append("")
    if "C" in rep:
        c = rep["C"]
        v = c["validate_gold"]
        out += ["## C: S&P500＋金＋日本株のAI（円建て・毎年初めに戻す・S&P500 と金は NISA、AI は特定口座）", "",
                f"金の値の確かめ（{v['start']}〜）: 先物−0.25% 年率 {_p(v['cagr_proxy'])}、実際の 1540 {_p(v['cagr_1540'])}、年ごとの差の平均 {_p(v['gap_mean'])}・最大 {_p(v['gap_abs_max'])} → {c['gold_series']}", ""]
        for wn, w in c["windows"].items():
            out += [f"### {w['start']}〜{w['end']}", "", "| 割合 | 年率（税引き前） | 税引き後 | 最大下落 | 最悪の年 | シャープ | 年率÷最大下落 |", "|---|---|---|---|---|---|---|"]
            for name, m in w["rows"].items():
                out.append(f"| {name} | {_p(m['cagr_pre_tax'])} | {_p(m['cagr_after_tax'])} | {_p(m['max_dd'])} | {_p(m['worst_year'])} | {m['sharpe']:.2f} | {m['calmar']:.2f} |")
            out.append("")
    return "\n".join(out)
