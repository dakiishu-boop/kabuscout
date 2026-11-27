"""上昇トレンドのときだけ2倍（docs/JOURNAL.md §91）の長期の検証。毎日の値動きから計算する速い版。

- 2倍の合成: 毎日 L × 指数の騰落 − (L−1) × (借りる金利 + spread) − 信託報酬（すべて年率を252で割る）
- 借りる金利・現金の利回り: ^IRX（1960年〜）。それより前は借りる金利3%・現金0%（厳しめ）。米国以外の指数は現金0%（厳しめ）
- 判定: 終値で判定 → 翌営業日の終値で売買（1日遅らせる＝厳しめ。本番は翌日の寄付）。切り替えのたびに cost（片道）
- 月1回: 月の最後の営業日に「その日から21営業日ずつさかのぼった10個の終値の平均」より上か（usscout の trend_spy と同じ判定）
- 毎日: 終値が200日平均より上か（Gayed & Bilello の規則）
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics

INDICES = {"^GSPC": "S&P500", "^N225": "日経平均", "^GDAXI": "DAX", "^FTSE": "FTSE100", "^HSI": "ハンセン"}
LONG = list(INDICES) + ["^IRX", "SPY", "SSO", "UPRO", "JPY=X", "^SP500TR"]
ER = 0.0089        # 2倍 ETF の信託報酬（SSO 0.87%、厳しめ）
SPREAD = 0.005     # 借りる金利の上乗せ（年）
PRE1960_FIN = 0.03


def fetch(root: Path, start: str = "1927-01-01") -> pd.DataFrame:
    import yfinance as yf
    d = yf.download(LONG, start=start, progress=False, auto_adjust=True, group_by="ticker", threads=True)
    c = pd.concat({k: d[k]["Close"] for k in LONG if k in d.columns.get_level_values(0)}, axis=1)
    c.index = pd.to_datetime(c.index).tz_localize(None).normalize()
    c = c.sort_index()
    root.mkdir(parents=True, exist_ok=True)
    c.to_parquet(root / "long.parquet")
    return c


def load(root: Path) -> pd.DataFrame:
    return pd.read_parquet(Path(root) / "long.parquet")


def rates(idx: pd.DatetimeIndex, irx: pd.Series, us: bool = True) -> tuple[pd.Series, pd.Series]:
    """(借りる金利, 現金の利回り) の年率。"""
    y = irx.reindex(idx).ffill() / 100
    fin = y.fillna(PRE1960_FIN) + SPREAD
    cash = (y - 0.005).clip(lower=0).fillna(0.0) if us else pd.Series(0.0, index=idx)
    return fin, cash


def synth(r: pd.Series, L: float, fin: pd.Series, er: float = ER) -> pd.Series:
    """毎日 L 倍に合わせ直す商品の騰落（L ≤ 1 なら現金を混ぜる扱いにしないで、そのまま L 倍の値動き）。"""
    if L <= 1:
        return L * r
    return L * r - (L - 1) * fin / 252 - er / 252


def signal_daily(px: pd.Series, n: int = 200) -> pd.Series:
    return (px > px.rolling(n, min_periods=n).mean()).astype(float).where(px.rolling(n).count() >= n)


def signal_monthly(px: pd.Series, months: int = 10, step: int = 21) -> pd.Series:
    sma = sum(px.shift(step * i) for i in range(months)) / months
    s = (px > sma).astype(float).where(sma.notna())
    me = px.groupby(px.index.to_period("M")).tail(1).index  # 月の最後の営業日
    out = pd.Series(np.nan, index=px.index)
    out[me] = s[me]
    return out.ffill()


def run(r: pd.Series, sig: pd.Series, L: float, fin: pd.Series, cash: pd.Series, cost: float, lag: int = 2) -> dict:
    """sig（終値での判定）→ lag 日後の騰落から L 倍（lag=2: 判定の翌日の終値で売買し、その次の日から効く）。"""
    e = sig.shift(lag).fillna(0.0)
    rl = synth(r, L, fin)
    sw = e.diff().abs().fillna(e.iloc[0])
    R = e * rl + (1 - e) * cash / 252 - sw.shift(-1).fillna(0) * cost  # 売買した日（効く前日）の終値でコスト
    return {"ret": R, "exposure": e * L, "switches": sw}


def curve(R: pd.Series) -> pd.Series:
    return (1 + R).cumprod()


def stats(R: pd.Series, expo: pd.Series | None = None, sw: pd.Series | None = None) -> dict:
    eq = curve(R)
    m = metrics.summary(eq)
    mon = eq.resample("ME").last().pct_change().dropna()
    m.update(worst_day=float(R.min()), worst_day_date=str(R.idxmin().date()), worst_month=float(mon.min()),
             vol=float(R.std() * np.sqrt(252)))
    if expo is not None:
        m["avg_exposure"] = float(expo.mean())
    if sw is not None:
        m["switches_per_year"] = float(sw.sum() / metrics.years(eq))
    return m


def evaluate(px: pd.Series, irx: pd.Series, start, end=None, us: bool = True, cost: float = 0.00545) -> dict:
    """1つの指数・1つの期間で、戦略4つと物差しを計算する。px は配当込みなら配当込み、なければ価格指数。"""
    px = px.dropna()
    r = px.pct_change().fillna(0.0)
    fin, cash = rates(px.index, irx, us)
    sigs = {"monthly": signal_monthly(px), "daily": signal_daily(px)}
    win = (r.index >= pd.Timestamp(start)) & (r.index <= pd.Timestamp(end or r.index[-1]))
    rw, fw, cw = r[win], fin[win], cash[win]
    out = {"start": str(rw.index[0].date()), "end": str(rw.index[-1].date())}
    out["buy_hold"] = stats(rw)
    for name, sg in sigs.items():
        for L in (1, 2, 3):
            res = run(r, sg, L, fin, cash, cost)
            R, e, sw = res["ret"][win], res["exposure"][win], res["switches"][win]
            out[f"{name}_{L}x"] = stats(R, e, sw)
            if L == 2:
                # 物差し: 平均の投資比率をそろえて持ち続ける（2倍の商品と現金を毎日一定の割合で）
                f = float(e.mean()) / 2
                Rm = f * synth(rw, 2, fw) + (1 - f) * cw / 252
                out[f"{name}_2x_exposure_matched"] = dict(stats(Rm), fraction_in_2x=f)
                # 物差し: 値動きの大きさ（年率の標準偏差）をそろえて持ち続ける
                k = float(R.std() / rw.std())
                Rv = synth(rw, k, fw) if k > 1 else k * rw + (1 - k) * cw / 252
                out[f"{name}_2x_vol_matched"] = dict(stats(Rv), leverage=k)
                out[f"{name}_2x_after_tax"] = metrics.cagr(metrics.after_tax(curve(R)))
    return out


def validate_sso(c: pd.DataFrame) -> dict:
    """合成の2倍（SPY の配当込みの騰落から）と、実際の SSO（配当込み）の年ごとの差。調整はしない。"""
    d = c[["SPY", "SSO", "^IRX"]].dropna(subset=["SPY", "SSO"])
    r = d["SPY"].pct_change().fillna(0)
    fin, _ = rates(d.index, c["^IRX"])
    syn = curve(synth(r, 2, fin))
    act = d["SSO"] / d["SSO"].iloc[0]
    ys, ya = metrics.yearly(syn), metrics.yearly(act)
    gap = (ys - ya)
    return {"start": str(d.index[0].date()), "cagr_synth": metrics.cagr(syn), "cagr_actual": metrics.cagr(act),
            "gap_per_year_mean": float(gap.mean()), "gap_per_year_abs_max": float(gap.abs().max()),
            "by_year": {int(k): round(float(v), 4) for k, v in gap.items()}}


def run_all(root: Path, refresh: bool = False) -> dict:
    root = Path(root)
    c = fetch(root) if refresh or not (root / "long.parquet").exists() else load(root)
    irx = c["^IRX"]
    rep = {"validate_sso": validate_sso(c)}
    gspc = c["^GSPC"].dropna()
    rep["sp500_long"] = evaluate(gspc, irx, "1929-01-01")                       # (b) 1929〜（200日分の助走の後）
    rep["sp500_post_paper"] = evaluate(gspc, irx, "2015-11-01")                  # (c) 研究の後
    if "^SP500TR" in c and c["^SP500TR"].notna().sum() > 1000:
        tr = c["^SP500TR"].dropna()
        rep["sp500tr_1989"] = evaluate(tr, irx, str((tr.index[0] + pd.DateOffset(years=1)).date()))  # 配当込み（参考）
    rep["sp500_2008"] = evaluate(c["SPY"].dropna(), irx, "2008-01-01")          # (a) の合成版（配当込み）
    rep["sp500_2008_jp_etf_cost"] = evaluate(c["SPY"].dropna(), irx, "2008-01-01", cost=0.001)
    rep["intl"] = {}
    for k in ("^N225", "^GDAXI", "^FTSE", "^HSI"):
        s = c[k].dropna()
        rep["intl"][k] = evaluate(s, irx, str((s.index[0] + pd.DateOffset(years=1)).date()), us=False)
    (root / "levtrend.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float))
    (root / "levtrend.md").write_text(markdown(rep))
    return rep


def _p(v):
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:+.1%}"


def table(ev: dict, title: str) -> list[str]:
    rows = [("持ち続ける（1倍）", "buy_hold"), ("本命: 月1回・2倍", "monthly_2x"), ("月1回・1倍（トレンドだけ）", "monthly_1x"),
            ("月1回・3倍（参考）", "monthly_3x"), ("研究の再現: 毎日200日・2倍", "daily_2x"), ("毎日200日・1倍", "daily_1x"),
            ("毎日200日・3倍（参考）", "daily_3x"), ("物差し: 投資比率をそろえて持ち続ける（月1回・2倍と同じ）", "monthly_2x_exposure_matched"),
            ("物差し: 値動きの大きさをそろえて持ち続ける（月1回・2倍と同じ）", "monthly_2x_vol_matched"),
            ("物差し: 投資比率をそろえて持ち続ける（毎日・2倍と同じ）", "daily_2x_exposure_matched")]
    out = [f"### {title}（{ev['start']}〜{ev['end']}）", "",
           "| | 年率 | 最大下落 | 最悪の年 | 最悪の月 | 最悪の日 | 年率の標準偏差 | 平均の投資比率 | 切り替え/年 |", "|---|---|---|---|---|---|---|---|---|"]
    for lab, k in rows:
        m = ev.get(k)
        if not m:
            continue
        out.append(f"| {lab} | {_p(m['cagr'])} | {_p(m['max_dd'])} | {_p(m['worst_year'])} | {_p(m['worst_month'])} | "
                   f"{_p(m['worst_day'])}（{m['worst_day_date']}） | {m['vol']:.0%} | {m.get('avg_exposure', 1):.2f} | {m.get('switches_per_year', 0):.1f} |")
    out += ["", f"- 税引き後（特定口座・毎年20.315%）: 月1回・2倍 {_p(ev.get('monthly_2x_after_tax'))}、毎日・2倍 {_p(ev.get('daily_2x_after_tax'))}"
            f"（持ち続ける側を NISA にすると税引き前と同じ {_p(ev['buy_hold']['cagr'])}）", ""]
    return out


def markdown(rep: dict) -> str:
    v = rep["validate_sso"]
    out = ["# 上昇トレンドのときだけ2倍（長期の検証・毎日の値動きから計算）", "",
           f"合成の2倍 と 実際の SSO（{v['start']}〜、配当込み）: 年率 {_p(v['cagr_synth'])} と {_p(v['cagr_actual'])}、"
           f"年ごとの差の平均 {_p(v['gap_per_year_mean'])}・最大 {_p(v['gap_per_year_abs_max'])}", ""]
    out += table(rep["sp500_long"], "S&P500（配当なし）1929〜")
    out += table(rep["sp500_post_paper"], "S&P500（配当なし）研究の後 2015-11〜")
    if "sp500tr_1989" in rep:
        out += table(rep["sp500tr_1989"], "S&P500 トータルリターン（配当込み）")
    out += table(rep["sp500_2008"], "SPY（配当込み）2008〜・米国の 2倍 ETF のコスト（片道0.545%）")
    out += table(rep["sp500_2008_jp_etf_cost"], "SPY（配当込み）2008〜・東証の 2倍 ETF を手数料無料で（気配差0.1%）")
    for k, ev in rep["intl"].items():
        out += table(ev, f"{INDICES[k]}（配当なし・現金0%・借りる金利は米国の短期金利＋0.5%）")
    return "\n".join(out)


# ---- §92: ボラティリティ管理（Moreira & Muir）。指数への投資比率 0〜2倍を、2倍の商品と現金の混ぜ方で持つ ----
def exposure_volmanaged(px: pd.Series, window: int = 21, cap: float = 2.0) -> pd.Series:
    """月の最後の営業日に、直近 window 日の分散の逆数 × その日までの全期間の分散 → 翌月の投資比率（先読みなし）。"""
    r = px.pct_change()
    rv = r.rolling(window, min_periods=window).var()
    target = r.expanding(min_periods=252).var()
    e = (target / rv).clip(upper=cap)
    me = px.groupby(px.index.to_period("M")).tail(1).index
    out = pd.Series(np.nan, index=px.index)
    out[me] = e[me]
    return out.ffill()


def run_mix(r: pd.Series, expo: pd.Series, fin: pd.Series, cash: pd.Series, cost: float, lag: int = 2) -> dict:
    """指数への投資比率 expo（0〜2）→ 2倍の商品に expo/2、残りを現金。売買の量 |Δ(expo/2)| にコスト。"""
    f = (expo.shift(lag) / 2).fillna(0.0)
    sw = f.diff().abs().fillna(f.iloc[0])
    R = f * synth(r, 2, fin) + (1 - f) * cash / 252 - sw.shift(-1).fillna(0) * cost
    return {"ret": R, "exposure": f * 2, "switches": sw}


def evaluate_vm(px: pd.Series, irx: pd.Series, start, end=None, us: bool = True, cost: float = 0.00545) -> dict:
    px = px.dropna()
    r = px.pct_change().fillna(0.0)
    fin, cash = rates(px.index, irx, us)
    res = run_mix(r, exposure_volmanaged(px), fin, cash, cost)
    win = (r.index >= pd.Timestamp(start)) & (r.index <= pd.Timestamp(end or r.index[-1]))
    rw, fw, cw = r[win], fin[win], cash[win]
    R, e, sw = res["ret"][win], res["exposure"][win], res["switches"][win]
    f = float(e.mean()) / 2
    Rm = f * synth(rw, 2, fw) + (1 - f) * cw / 252
    return {"start": str(rw.index[0].date()), "end": str(rw.index[-1].date()), "buy_hold": stats(rw),
            "volmanaged": stats(R, e, sw), "volmanaged_exposure_matched": dict(stats(Rm), fraction_in_2x=f),
            "volmanaged_after_tax": metrics.cagr(metrics.after_tax(curve(R))), "turnover_per_year": float(sw.sum() / metrics.years(curve(R)))}


def run_vm(root: Path) -> dict:
    c = load(root)
    irx = c["^IRX"]
    g = c["^GSPC"].dropna()
    rep = {"sp500_long": evaluate_vm(g, irx, "1929-01-01"), "sp500_post_paper": evaluate_vm(g, irx, "2016-05-01"),
           "sp500_2008": evaluate_vm(c["SPY"].dropna(), irx, "2008-01-01"),
           "sp500_2008_jp_etf_cost": evaluate_vm(c["SPY"].dropna(), irx, "2008-01-01", cost=0.001), "intl": {}}
    if "^SP500TR" in c and c["^SP500TR"].notna().sum() > 1000:
        tr = c["^SP500TR"].dropna()
        rep["sp500tr_1989"] = evaluate_vm(tr, irx, str((tr.index[0] + pd.DateOffset(years=1)).date()))
    for k in ("^N225", "^GDAXI", "^FTSE", "^HSI"):
        s = c[k].dropna()
        rep["intl"][k] = evaluate_vm(s, irx, str((s.index[0] + pd.DateOffset(years=1)).date()), us=False)
    (Path(root) / "volmanaged.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float))
    (Path(root) / "volmanaged.md").write_text(markdown_vm(rep))
    return rep


def markdown_vm(rep: dict) -> str:
    names = {"sp500_long": "S&P500（配当なし）1929〜", "sp500_post_paper": "S&P500（配当なし）研究の後 2016-05〜",
             "sp500tr_1989": "S&P500 トータルリターン（配当込み・参考）", "sp500_2008": "SPY（配当込み）2008〜・米国 ETF のコスト",
             "sp500_2008_jp_etf_cost": "SPY（配当込み）2008〜・東証 ETF 手数料無料（参考）"}
    out = ["# 値動きが落ち着いているときほど多く持つ（ボラティリティ管理・Moreira & Muir の規則・投資比率0〜2倍）", "",
           "| 期間 | 持ち続ける 年率・最大下落 | ボラティリティ管理 年率・最大下落 | 平均の投資比率 | 投資比率をそろえた持ち続け | 税引き後 | 最悪の日 |",
           "|---|---|---|---|---|---|---|"]
    items = [(names[k], rep[k]) for k in names if k in rep] + [(INDICES[k], v) for k, v in rep["intl"].items()]
    for lab, ev in items:
        b, v, m = ev["buy_hold"], ev["volmanaged"], ev["volmanaged_exposure_matched"]
        out.append(f"| {lab}（{ev['start'][:7]}〜） | {_p(b['cagr'])}・{_p(b['max_dd'])} | {_p(v['cagr'])}・{_p(v['max_dd'])} | {v['avg_exposure']:.2f} | "
                   f"{_p(m['cagr'])}・{_p(m['max_dd'])} | {_p(ev['volmanaged_after_tax'])} | {_p(v['worst_day'])}（{v['worst_day_date']}） |")
    return "\n".join(out)
