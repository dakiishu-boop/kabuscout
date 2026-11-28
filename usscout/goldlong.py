"""§95: S&P500＋金（NISA）を2000年より前で確かめる（docs/JOURNAL.md §95 の計画どおり）。月ごとの値で計算する。

- 金: 世界銀行 Pink Sheet（CMO Historical Data Monthly、ドル/トロイオンスの月平均、1960-01〜）
- S&P500 の配当込み: Robert Shiller の ie_data.xls（月平均の値 P・年率の配当 D）。前月比 (P + D/12) ÷ 前月の P
- ドル円: FRED EXJPUS（月平均）。短期金利: FRED TB3MS
ダウンロードしたファイルは data/us_scout/long_history/ に置く（git に入らない）。中身はデータとして読むだけ。
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

WB_URL = "https://thedocs.worldbank.org/en/doc/74e8be41ceb20fa0da750cda2f6b9e4e-0050012026/related/CMO-Historical-Data-Monthly.xlsx"
SHILLER_PAGE = "https://shillerdata.com/"
FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"
MIXES = {"S&P500 80%＋金 20%": 0.8, "S&P500 60%＋金 40%": 0.6}
FEE_SP, FEE_GOLD, MOVE_COST = 0.001, 0.0025, 0.001


def fetch(root: Path) -> Path:
    import re
    d = Path(root) / "long_history"
    d.mkdir(parents=True, exist_ok=True)
    (d / "wb_monthly.xlsx").write_bytes(requests.get(WB_URL, timeout=120).content)
    html = requests.get(SHILLER_PAGE, timeout=60).text
    m = re.search(r'(//img1\.wsimg\.com/[^"]+/ie_data\.xls[^"]*)', html)
    if not m:
        raise RuntimeError("Shiller の ie_data.xls の場所が見つかりません")
    (d / "ie_data.xls").write_bytes(requests.get("https:" + m.group(1), timeout=120).content)
    for sid in ("EXJPUS", "TB3MS"):
        (d / f"{sid}.csv").write_text(requests.get(FRED.format(sid), timeout=60).text)
    return d


def _month(idx) -> pd.DatetimeIndex:
    return pd.PeriodIndex(idx, freq="M").to_timestamp(how="end").normalize()


def load(root: Path) -> pd.DataFrame:
    d = Path(root) / "long_history"
    m = pd.read_excel(d / "wb_monthly.xlsx", "Monthly Prices", header=None)
    hdr = next(i for i in range(12) if any(str(v).strip() == "Gold" for v in m.iloc[i]))
    col = [j for j, v in enumerate(m.iloc[hdr].astype(str)) if v.strip() == "Gold"][0]
    g = m.iloc[hdr + 2:, [0, col]].dropna()
    g = g[g.iloc[:, 0].astype(str).str.match(r"^\d{4}M\d{2}$")]
    gold = pd.Series(pd.to_numeric(g.iloc[:, 1], errors="coerce").to_numpy(),
                     index=_month([pd.Period(f"{s[:4]}-{s[5:]}", "M") for s in g.iloc[:, 0].astype(str)]))
    z = pd.read_excel(d / "ie_data.xls", "Data", header=None).iloc[8:, :3]
    z.columns = ["date", "P", "D"]
    z = z[pd.to_numeric(z["date"], errors="coerce").notna()].astype(float)
    yr = np.floor(z["date"]).astype(int)
    mo = np.rint((z["date"] - yr) * 100).astype(int)
    z.index = _month([pd.Period(f"{y}-{m:02d}", "M") for y, m in zip(yr, mo)])
    z = z.dropna(subset=["P", "D"])
    tr = ((z["P"] + z["D"] / 12) / z["P"].shift(1)).fillna(1.0).cumprod()
    fred = {}
    for sid in ("EXJPUS", "TB3MS"):
        f = pd.read_csv(d / f"{sid}.csv")
        f.columns = ["date", sid]
        s = pd.to_numeric(f[sid], errors="coerce")
        fred[sid] = pd.Series(s.to_numpy(), index=_month(pd.to_datetime(f["date"]).dt.to_period("M")))
    return pd.DataFrame({"sp_tr": tr, "gold": gold, "usdjpy": fred["EXJPUS"], "tbill": fred["TB3MS"]}).sort_index()


def _fees(s: pd.Series, fee: float) -> pd.Series:
    r = s.pct_change().fillna(0.0) - fee / 12
    return (1 + r).cumprod()


def mix(sp: pd.Series, gold: pd.Series, w: float) -> pd.Series:
    """毎年1月に S&P500 w・金 1−w に戻す（前の年の12月末の値で）。動かす金額に MOVE_COST。"""
    rs, rg = sp.pct_change().fillna(0.0), gold.pct_change().fillna(0.0)
    vs, vg, out = w, 1 - w, []
    for i, d in enumerate(sp.index):
        if i > 0 and d.month == 1:
            tot = vs + vg
            move = abs(w * tot - vs)
            tot -= move * MOVE_COST * 2
            vs, vg = w * tot, (1 - w) * tot
        vs *= 1 + rs.iloc[i]
        vg *= 1 + rg.iloc[i]
        out.append(vs + vg)
    return pd.Series(out, index=sp.index)


def stats(v: pd.Series, rf_m: pd.Series | None = None) -> dict:
    v = v.dropna()
    r = v.pct_change().dropna()
    yrs = len(r) / 12
    cagr = float((v.iloc[-1] / v.iloc[0]) ** (1 / yrs) - 1)
    dd = float((v / v.cummax() - 1).min())
    ex = r - (rf_m.reindex(r.index).fillna(0.0) if rf_m is not None else 0.0)
    sharpe = float(ex.mean() / r.std() * np.sqrt(12)) if r.std() > 0 else np.nan
    y = v.groupby(v.index.year).last()
    y = pd.concat([pd.Series([v.iloc[0]], index=[v.index[0].year - 1]), y]).pct_change().dropna()
    return {"cagr": cagr, "max_dd": dd, "worst_year": float(y.min()), "sharpe": sharpe, "calmar": cagr / abs(dd) if dd < 0 else np.nan,
            "start": str(v.index[0].date()), "end": str(v.index[-1].date())}


def validate(df: pd.DataFrame, root: Path) -> dict:
    """世界銀行の金 と Yahoo の金先物（月平均）、Shiller の配当込み と ^SP500TR（月平均）の年ごとの差（合わせ込みはしない）。"""
    out = {}
    root = Path(root)
    try:
        g = pd.read_parquet(root / "gold.parquet")["GC=F"].dropna()
        gm = g.groupby(g.index.to_period("M")).mean()
        gm.index = gm.index.to_timestamp(how="end").normalize()
        a, b = df["gold"].reindex(gm.index).dropna(), gm
        b = b.reindex(a.index)
        ya, yb = a.groupby(a.index.year).last().pct_change(), b.groupby(b.index.year).last().pct_change()
        gap = (ya - yb).dropna()
        out["gold_vs_gcf"] = {"from": str(a.index[0].date()), "gap_mean": float(gap.mean()), "gap_abs_max": float(gap.abs().max())}
    except Exception as e:
        out["gold_vs_gcf"] = {"error": str(e)}
    try:
        t = pd.read_parquet(root / "long.parquet")["^SP500TR"].dropna()
        tm = t.groupby(t.index.to_period("M")).mean()
        tm.index = tm.index.to_timestamp(how="end").normalize()
        a = df["sp_tr"].reindex(tm.index).dropna()
        b = tm.reindex(a.index)
        ya, yb = a.groupby(a.index.year).last().pct_change(), b.groupby(b.index.year).last().pct_change()
        gap = (ya - yb).dropna()
        out["sp_vs_sp500tr"] = {"from": str(a.index[0].date()), "gap_mean": float(gap.mean()), "gap_abs_max": float(gap.abs().max())}
    except Exception as e:
        out["sp_vs_sp500tr"] = {"error": str(e)}
    return out


def evaluate(root: Path) -> dict:
    df = load(root)
    df = df.dropna(subset=["sp_tr", "gold"])
    windows = {"usd_pre2000": ("1972-01-31", "1999-12-31", "usd"), "jpy_pre2000": ("1973-03-31", "1999-12-31", "jpy"),
               "usd_full": ("1972-01-31", None, "usd"), "jpy_full": ("1973-03-31", None, "jpy"),
               "usd_gold_winter": ("1980-01-31", "1999-12-31", "usd")}
    rep = {"validate": validate(df, root), "data_end": str(df.index[-1].date()), "windows": {}}
    for name, (a, b, ccy) in windows.items():
        w = df.loc[a:b] if b else df.loc[a:]
        if ccy == "jpy":
            w = w.dropna(subset=["usdjpy"])
        fx = w["usdjpy"] if ccy == "jpy" else pd.Series(1.0, index=w.index)
        sp = _fees(w["sp_tr"] * fx, FEE_SP)
        gd = _fees(w["gold"] * fx, FEE_GOLD)
        rf = (w["tbill"] / 100 / 12) if ccy == "usd" else None
        rows = {"S&P500 だけ": stats(sp, rf), "金だけ": stats(gd, rf)}
        for k, wt in MIXES.items():
            rows[k] = stats(mix(sp, gd, wt), rf)
        base = rows["S&P500 だけ"]
        for k in MIXES:
            m = rows[k]
            m["pass"] = bool(m["sharpe"] > base["sharpe"] and m["calmar"] > base["calmar"] and base["cagr"] - m["cagr"] <= 0.02)
        rep["windows"][name] = rows
    decisive = ["usd_pre2000", "jpy_pre2000", "usd_full", "jpy_full"]
    rep["verdict"] = {k: all(rep["windows"][w][k]["pass"] for w in decisive) for k in MIXES}
    root = Path(root)
    (root / "goldlong.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float))
    (root / "goldlong.md").write_text(markdown(rep))
    return rep


def _p(v):
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:+.1%}"


def markdown(rep: dict) -> str:
    names = {"usd_pre2000": "ドル建て 1972〜1999（決め手）", "jpy_pre2000": "円建て 1973-03〜1999（決め手）", "usd_full": "ドル建て 1972〜（全期間）",
             "jpy_full": "円建て 1973-03〜（全期間）", "usd_gold_winter": "（参考）ドル建て 1980〜1999（金の冬）"}
    v = rep["validate"]
    out = ["# S&P500＋金を2000年より前で確かめる（月ごとの値・毎年1月に戻す・NISA の想定で税金なし）", "",
           f"- 確かめ: 世界銀行の金と金先物（{v['gold_vs_gcf'].get('from', '')}〜）の年ごとの差 平均 {_p(v['gold_vs_gcf'].get('gap_mean'))}・最大 {_p(v['gold_vs_gcf'].get('gap_abs_max'))}。"
           f"Shiller の配当込みと ^SP500TR（{v['sp_vs_sp500tr'].get('from', '')}〜）の差 平均 {_p(v['sp_vs_sp500tr'].get('gap_mean'))}・最大 {_p(v['sp_vs_sp500tr'].get('gap_abs_max'))}", ""]
    for k, rows in rep["windows"].items():
        r0 = next(iter(rows.values()))
        out += [f"## {names[k]}（{r0['start'][:7]}〜{r0['end'][:7]}）", "", "| | 年率 | 最大下落（月） | 最悪の年 | シャープ | 年率÷最大下落 | 基準 |", "|---|---|---|---|---|---|---|"]
        for n, m in rows.items():
            mark = "" if "pass" not in m else ("○" if m["pass"] else "×")
            out.append(f"| {n} | {_p(m['cagr'])} | {_p(m['max_dd'])} | {_p(m['worst_year'])} | {m['sharpe']:.2f} | {m['calmar']:.2f} | {mark} |")
        out.append("")
    out.append("判定（決め手の4つの期間すべてで基準を満たすか）: " + "、".join(f"{k} {'合格' if ok else '不合格'}" for k, ok in rep["verdict"].items()))
    return "\n".join(out)
