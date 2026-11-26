"""過去の口座のシミュレーション一式（docs/JOURNAL.md §90 の表を作る）。

日本株版の研究ループと同じ安全装置:
- 金庫: backtest.vault_start 以降は研究の表に使わず、最後の表（final）でだけ開ける
- 年ごとの比較: SPY との年ごとの差の中央値・SPY に勝った年の割合
- コストの場面を全戦略にそろえて当てる（手数料なし / ETF の短期国債 BIL / 米ドルMMF）
- ランダムに選んだ場合・判定日をずらした場合・条件を少し変えた場合（感度）の幅も出す
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from . import backtest, data, metrics, strategies, universe

log = logging.getLogger(__name__)

# 物差し（hold_*）と戦略。trend_qqq と hold_qqq は「QQQ が良かったのを見た後に足した」＝後知恵の参考
NAMES = ["hold_spy", "hold_qqq", "gem", "etf_rotation", "trend_spy", "trend_qqq", "stock_mom", "blend"]
HINDSIGHT = {"hold_qqq", "trend_qqq"}
LABEL = {"hold_spy": "S&P500を持ち続ける（SPY）", "hold_qqq": "NASDAQ100を持ち続ける（QQQ・後知恵）",
         "gem": "デュアルモメンタム（GEM）", "etf_rotation": "ETFローテーション（上位3本）",
         "trend_spy": "SPY＋10か月平均の逃避", "trend_qqq": "QQQ＋10か月平均の逃避（後知恵）",
         "stock_mom": "S&P500個別株モメンタム上位10", "blend": "ETFローテーション50%＋個別株モメンタム50%",
         "equal_weight": "S&P500構成銘柄の均等持ち（参考）"}


def scenarios(cfg: dict) -> dict:
    c = dict(cfg["costs"])
    return {
        "gross": ({}, "BIL"),                 # 手数料・気配差なし（戦略そのものの力）
        "net_bil": (c, "BIL"),                # 0.495%＋0.05%、逃避先は ETF の BIL（厳しめの既定）
        "net_mmf": (c, data.CASH),            # 同じ手数料、逃避先は米ドルMMF（売買の手数料なし）
    }


def cash_overrides(name: str, cfg: dict, cash: str, extra: dict | None = None) -> dict | None:
    if name.startswith("hold_") or name == "equal_weight":
        return None
    if name == "blend":
        return {k: dict({"cash": cash}, **((extra or {}).get(k, {}))) for k in cfg["strategies"]["blend"]["parts"]}
    return dict({"cash": cash}, **(extra or {}))


class Lab:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        root = cfg["data"]["dir"]
        self.p = data.with_cash(data.load(root))
        cur, ch = universe.fetch_sp500(root)
        self.members = universe.membership(cur, ch, data.month_ends(data.calendar(self.p)))
        self.spy = self.p.close["SPY"].dropna()
        self.fx = self.p.close["JPY=X"] if "JPY=X" in self.p.close else None
        self.vault = pd.Timestamp(cfg["backtest"]["vault_start"])
        self.research_end = self.vault - pd.Timedelta(days=1)

    def run(self, name, scen="net_mmf", start=None, end=None, extra=None, seed=None, **kw):
        costs, cash = scenarios(self.cfg)[scen]
        s = strategies.build(name, self.cfg, cash_overrides(name, self.cfg, cash, extra), seed=seed)
        return backtest.run(self.p, s, self.cfg, start=start, end=end, members=self.members, costs=costs, **kw)

    def stats(self, r) -> dict:
        m = metrics.summary(r.equity, self.spy, r.trades)
        if r.exposure is not None and len(r.exposure):
            m["exposure"] = float(r.exposure.mean())
        if r.coverage is not None and len(r.coverage):
            m["coverage"] = float(r.coverage.mean())
        return m


def run_all(cfg: dict, quick: bool = False) -> dict:
    lab = Lab(cfg)
    R = lab.research_end
    rep: dict = {"research_period": [cfg["backtest"]["start"], str(R.date())], "vault_start": str(lab.vault.date()),
                 "data_end": str(lab.p.dates[-1].date()), "labels": LABEL, "hindsight": sorted(HINDSIGHT)}

    # 1. 研究期間（金庫の前まで）: コストの場面 × 戦略
    rep["research"] = {}
    for sc in scenarios(cfg):
        rep["research"][sc] = {n: lab.stats(lab.run(n, sc, end=R)) for n in NAMES}
        log.info("研究期間 %s 完了", sc)
    rep["research"]["gross"]["equal_weight"] = lab.stats(lab.run("equal_weight", "gross", end=R, integer=False))

    if not quick:
        # 2. ランダムに選んだ場合（個別株モメンタムと同じ条件を満たす銘柄から、順位の代わりに乱数）
        rnd = [metrics.cagr(lab.run("stock_mom", "gross", end=R, extra={"score": "random"}, seed=sd).equity) for sd in range(20)]
        mom = rep["research"]["gross"]["stock_mom"]["cagr"]
        rep["random_stock_mom_gross"] = {"cagr_median": float(np.median(rnd)), "cagr_min": float(min(rnd)),
                                         "cagr_max": float(max(rnd)), "momentum_beats": int(sum(x < mom for x in rnd)), "n": len(rnd)}
        # 3. 判定日をずらす（月末の0〜4営業日前）: たまたま月末が良かっただけではないか
        rep["offsets_net_mmf"] = {}
        for n in NAMES[2:]:
            c = [metrics.cagr(lab.run(n, "net_mmf", end=R, offset=o).equity) for o in range(5)]
            rep["offsets_net_mmf"][n] = {"min": float(min(c)), "max": float(max(c)), "median": float(np.median(c))}
        # 4. 感度（条件を少し変える。選ぶためではなく、決めた値が「たまたまの当たり」でないかを見る）
        grid = {"stock_mom": [{"lookback_months": L, "top": t, "keep_rank": 2 * t} for L in (6, 9, 12) for t in (5, 10, 20)],
                "etf_rotation": [{"top": t} for t in (2, 3, 4)],
                "trend_spy": [{"sma_months": m} for m in (8, 10, 12)]}
        rep["sensitivity_net_mmf"] = {}
        for n, vs in grid.items():
            rows = []
            for v in vs:
                e = lab.run(n, "net_mmf", end=R, extra=v).equity
                rows.append(dict(v, cagr=metrics.cagr(e), max_dd=metrics.max_dd(e)))
            rep["sensitivity_net_mmf"][n] = rows
        log.info("ランダム・判定日・感度 完了")

    # 5. 金庫を開ける（最後の1回）: 全期間と金庫の期間だけ
    rep["final"] = {}
    for sc in ("net_bil", "net_mmf"):
        rep["final"][sc] = {}
        for n in NAMES:
            full = lab.run(n, sc)
            vault = lab.run(n, sc, start=str(lab.vault.date()))
            rep["final"][sc][n] = {"full": lab.stats(full), "vault": metrics.summary(vault.equity)}
            if sc == "net_mmf":
                rep["final"][sc][n].update(_tax_fx(lab, full.equity, n))
    rep["final"]["window_2016"] = {n: lab.stats(lab.run(n, "net_mmf", start="2016-01-01")) for n in NAMES}
    for n in NAMES:
        e = lab.run(n, "net_mmf", start="2016-01-01").equity
        rep["final"]["window_2016"][n]["after_tax_cagr"] = metrics.cagr(e if n.startswith("hold_") else metrics.after_tax(e, cfg["tax"]["rate"], cfg["tax"]["carry_years"]))
    root = cfg["data"]["dir"]
    (root / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float))
    (root / "report.md").write_text(markdown(rep))
    log.info("保存しました: %s", root / "report.md")
    return rep


def _tax_fx(lab: Lab, eq: pd.Series, name: str) -> dict:
    """税引き後（持ち続ける物差しは NISA＝非課税、売買する戦略は特定口座＝毎年 20.315%）と円建て。"""
    cfg = lab.cfg
    taxed = eq if name.startswith("hold_") else metrics.after_tax(eq, cfg["tax"]["rate"], cfg["tax"]["carry_years"])
    out = {"after_tax_cagr": metrics.cagr(taxed), "account": "NISA" if name.startswith("hold_") else "特定口座"}
    if lab.fx is not None:
        j = metrics.to_jpy(eq, lab.fx, cfg["costs"].get("fx_spread_jpy", 0))
        jt = metrics.to_jpy(taxed, lab.fx, cfg["costs"].get("fx_spread_jpy", 0))
        out.update(jpy_cagr=metrics.cagr(j), jpy_max_dd=metrics.max_dd(j), jpy_after_tax_cagr=metrics.cagr(jt))
    return out


def _p(v) -> str:
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:+.1%}"


def markdown(rep: dict) -> str:
    L = rep["labels"]
    out = [f"# USScout 過去の口座のシミュレーション（データ {rep['data_end']} まで）", "",
           f"研究期間 {rep['research_period'][0]}〜{rep['research_period'][1]}、金庫 {rep['vault_start']}〜。"
           "資金6,700ドル・1株単位・月末の引けで判定→翌営業日の寄付で売買。配当込み（再投資）。", ""]
    names = {"gross": "手数料なし", "net_bil": "手数料0.495%＋気配差0.05%（片道）・逃避先 BIL", "net_mmf": "同じ手数料・逃避先 米ドルMMF（手数料なし）"}
    for sc, tab in rep["research"].items():
        out += [f"## 研究期間: {names[sc]}", "", "| 戦略 | 年率 | 最大下落 | 最悪の年 | プラスの年 | SPYとの差(年の中央値) | SPYに勝った年 | コスト/年 | 投資比率 |",
                "|---|---|---|---|---|---|---|---|---|"]
        for n, m in tab.items():
            out.append(f"| {L.get(n, n)} | {_p(m['cagr'])} | {_p(m['max_dd'])} | {_p(m['worst_year'])} | {m['pos_years']:.0%} | "
                       f"{_p(m.get('excess_median'))} | {m.get('beat_years', 0):.0%} | {m.get('cost_per_year', 0):.2%} | {m.get('exposure', 1):.0%} |")
        out.append("")
    if "random_stock_mom_gross" in rep:
        r = rep["random_stock_mom_gross"]
        out += [f"- 個別株モメンタム（手数料なし）とランダムに選んだ場合（{r['n']}通り）: ランダムの年率 中央値 {_p(r['cagr_median'])}"
                f"（{_p(r['cagr_min'])}〜{_p(r['cagr_max'])}）。モメンタムが上回ったのは {r['momentum_beats']}/{r['n']}", ""]
        out += ["## 判定日を月末の0〜4営業日前にずらした場合（米ドルMMF・研究期間）", "", "| 戦略 | 年率の最小 | 中央値 | 最大 |", "|---|---|---|---|"]
        for n, v in rep["offsets_net_mmf"].items():
            out.append(f"| {L.get(n, n)} | {_p(v['min'])} | {_p(v['median'])} | {_p(v['max'])} |")
        out += ["", "## 感度（米ドルMMF・研究期間）", ""]
        for n, vs in rep["sensitivity_net_mmf"].items():
            out.append(f"- {L.get(n, n)}: " + "、".join(
                ",".join(f"{k}={v}" for k, v in x.items() if k not in ("cagr", "max_dd")) + f" → {_p(x['cagr'])}/{_p(x['max_dd'])}" for x in vs))
        out.append("")
    for sc in ("net_bil", "net_mmf"):
        out += [f"## 金庫を開けた後: 全期間（{rep['research_period'][0]}〜{rep['data_end']}）・{names[sc]}", "",
                "| 戦略 | 年率 | 最大下落 | 最悪の年 | 金庫の期間の騰落(年率) | 税引き後の年率 | 円建ての年率 | 円建ての最大下落 |", "|---|---|---|---|---|---|---|---|"]
        for n, v in rep["final"][sc].items():
            f, vt = v["full"], v["vault"]
            tax = f"{_p(v['after_tax_cagr'])}（{v['account']}）" if "after_tax_cagr" in v else "—"
            out.append(f"| {L.get(n, n)} | {_p(f['cagr'])} | {_p(f['max_dd'])} | {_p(f['worst_year'])} | {_p(vt['cagr'])} | {tax} | "
                       f"{_p(v.get('jpy_cagr'))} | {_p(v.get('jpy_max_dd'))} |")
        out.append("")
    out += ["## 2016年〜（日本株版の「今のAI」+12.1%・税引き後 約+9.7% と同じ期間の目安）・米ドルMMF", "",
            "| 戦略 | 年率 | 最大下落 | 税引き後（持ち続けはNISA） |", "|---|---|---|---|"]
    for n, m in rep["final"]["window_2016"].items():
        out.append(f"| {L.get(n, n)} | {_p(m['cagr'])} | {_p(m['max_dd'])} | {_p(m['after_tax_cagr'])} |")
    out += ["", "注意: 銘柄は上場廃止の会社の株価が取れない（生存者バイアス）。S&P500 の入れ替えの履歴は 2008〜2010年に抜けがある。"
            "税金は毎年末に利益の20.315%を引く簡略計算（含み益にも課税＝厳しめ）。米国の配当の源泉税10%は入れていない。"]
    return "\n".join(out)
