"""企業の来期の利益を予想する「予想の練習」と、その予想で銘柄を選べるかの検証。

- 予想する値: 来期の純利益の増減 ÷ 今期の売上高（1株あたりの値は分割で基準が変わるので使わない）
- 予想に使うもの: 有報の過去の推移（売上・利益・利益率・ROE・自己資本比率）と、予想する時点までの株価の動き・業種
- 予想する時点: 決算期末の95日後（有報が出そろった後）。年ごとに「その年より前に答えが分かっていたデータ」だけで学習する
- 答え合わせ: 「来期も今期と同じ（増減0）」という単純な予想より、会社ごとの順位がどれだけ当たるか
- AIの特徴量（FC_FEATURES）: 予想した利益の増減・来期の利益を、その日の時価総額で割った値（自前の予想に基づく益回り）

ジャーナル ラウンド10: 予想の順位相関 0.30（「今期の増減が続く」は −0.10）。流動性のある全銘柄で毎月、
割安さ（実績の益回り）・モメンタム・規模をそろえても、予想した増益÷時価総額の上位−下位は +0.8%/月（t=3.6）。
"""
from __future__ import annotations

import functools
import json
import logging

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor

from . import data, edinet

log = logging.getLogger(__name__)
FC_FEATURES = ["fc_dey", "fc_fey"]
LAG = pd.Timedelta(days=95)
FEATS = ["margin", "margin_chg", "margin_vs_avg", "sales_g1", "sales_g2", "dprof1", "dprof2", "roe", "equity_ratio", "loss",
         "log_sales", "mom1", "mom3", "mom6", "mom12", "vol60", "off_high", "sector"]


def panel(cfg: dict) -> pd.DataFrame:
    """会社×決算期の表（特徴量と答え）。"""
    f = edinet.load(cfg)
    f = f.dropna(subset=["sales", "profit"]).sort_values(["code", "period_end"]).copy()
    f = f[f["sales"] > 0]
    g = f.groupby("code")
    gap = (f["period_end"] - g["period_end"].shift(1)).dt.days
    annual = gap.between(330, 400)  # 前の期とちょうど1年（決算期の変更は除く）
    prev = lambda c, k=1: g[c].shift(k)
    f["margin"] = (f["profit"] / f["sales"]).clip(-1, 1)
    f["margin_chg"] = (f["margin"] - g["margin"].shift(1)).where(annual)
    f["margin_vs_avg"] = f["margin"] - g["margin"].transform(lambda s: s.rolling(4, min_periods=2).mean())
    f["sales_g1"] = (f["sales"] / prev("sales") - 1).where(annual).clip(-0.9, 3)
    f["sales_g2"] = (prev("sales") / prev("sales", 2) - 1).clip(-0.9, 3)
    f["dprof1"] = ((f["profit"] - prev("profit")) / f["sales"]).where(annual).clip(-0.5, 0.5)
    f["dprof2"] = ((prev("profit") - prev("profit", 2)) / prev("sales")).clip(-0.5, 0.5)
    f["loss"] = (f["profit"] < 0).astype(float)
    f["log_sales"] = np.log10(f["sales"])
    nxt_gap = (g["period_end"].shift(-1) - f["period_end"]).dt.days
    f["y"] = ((g["profit"].shift(-1) - f["profit"]) / f["sales"]).where(nxt_gap.between(330, 400)).clip(-0.3, 0.3)
    f["known"] = f["period_end"] + LAG
    f["y_known"] = g["period_end"].shift(-1) + LAG
    return f


def add_market(cfg: dict, f: pd.DataFrame, close: pd.DataFrame, uni: pd.DataFrame) -> pd.DataFrame:
    """予想する時点（known）までの株価の動きと業種。"""
    idx = close.index
    lr = np.log(close).diff()
    feats = {"mom1": close / close.shift(21) - 1, "mom3": close / close.shift(63) - 1, "mom6": close / close.shift(126) - 1,
             "mom12": close / close.shift(250) - 1, "vol60": lr.rolling(60, min_periods=40).std() * np.sqrt(250),
             "off_high": close / close.rolling(250, min_periods=120).max() - 1}
    f = f[f["code"].isin(set(close.columns))].copy()
    pos = idx.searchsorted(f["known"].to_numpy(), side="right") - 1  # known の日以前で最後の営業日
    ok = pos >= 0
    f = f[ok].copy()
    pos = pos[ok]
    ci = close.columns.get_indexer(f["code"])
    for k, w in feats.items():
        f[k] = w.to_numpy()[pos, ci]
    f["sector"] = f["code"].map(dict(zip(uni["code"], uni["sector33"]))).astype("category")
    return f


def walk_forward(f: pd.DataFrame, start_year: int = 2016, params: dict | None = None) -> pd.DataFrame:
    """年ごとに学習し直して予想する（その年の1月1日より前に答えが分かっていた期だけで学習）。"""
    p = dict(max_iter=400, learning_rate=0.03, max_leaf_nodes=15, min_samples_leaf=100, l2_regularization=1.0,
             categorical_features="from_dtype", random_state=42, **(params or {}))
    out = []
    for y in range(start_year, int(f["known"].dt.year.max()) + 1):
        cut = pd.Timestamp(f"{y}-01-01")
        tr = f[(f["y_known"] < cut) & f["y"].notna()]
        te = f[(f["known"] >= cut) & (f["known"] < pd.Timestamp(f"{y + 1}-01-01"))]
        if len(tr) < 2000 or te.empty:
            continue
        m = HistGradientBoostingRegressor(**p).fit(tr[FEATS], tr["y"])
        out.append(te.assign(pred=m.predict(te[FEATS])))
        log.info("  %d年: %d社・期で学習 → %d件を予想", y, len(tr), len(te))
    return pd.concat(out, ignore_index=True)


def score(pr: pd.DataFrame) -> pd.DataFrame:
    """年ごとの答え合わせ。順位相関（予想と実際）、増益/減益の当たり、単純な予想との比較。"""
    rows = []
    for y, g in pr.dropna(subset=["y"]).groupby(pr["known"].dt.year):
        rows.append(dict(year=int(y), n=len(g),
                         rank_ic=g["pred"].corr(g["y"], method="spearman"),
                         naive_ic=g["dprof1"].corr(g["y"], method="spearman"),  # 「今期の増減がそのまま続く」
                         hit=float(((g["pred"] > 0) == (g["y"] > 0)).mean()), up_rate=float((g["y"] > 0).mean()),
                         mae=float((g["pred"] - g["y"]).abs().mean()), mae_zero=float(g["y"].abs().mean())))
    return pd.DataFrame(rows)


def monthly_spread(pr: pd.DataFrame, close: pd.DataFrame, turnover: pd.DataFrame, col: str = "pred",
                   min_turnover: float = 1e8, start: str = "2016-06-01", n_q: int = 5) -> dict:
    """毎月末、各社の最新の予想（1年以内）で5つに分け、翌月の株価の上がり方を比べる（流動性の下限つき）。"""
    me = close.resample("ME").last().index
    me = close.index[close.index.searchsorted(me, side="right") - 1]
    me = me[me >= pd.Timestamp(start)]
    tv = turnover.rolling(20, min_periods=10).mean()
    pr = pr.sort_values("known")
    res = []
    for a, b in zip(me[:-1], me[1:]):
        cur = pr[(pr["known"] <= a) & (pr["known"] > a - pd.Timedelta(days=380))].drop_duplicates("code", keep="last")
        cur = cur[cur["code"].isin(close.columns)]
        r = close.loc[b, cur["code"]].to_numpy() / close.loc[a, cur["code"]].to_numpy() - 1
        liq = tv.loc[a, cur["code"]].to_numpy() >= min_turnover
        d = pd.DataFrame({"s": cur[col].to_numpy(), "r": r})[liq].dropna()
        if len(d) < 200:
            continue
        d["q"] = pd.qcut(d["s"].rank(method="first"), n_q, labels=False)
        q = d.groupby("q")["r"].mean()
        res.append(dict(month=b, spread=q.iloc[-1] - q.iloc[0], **{f"q{i + 1}": v for i, v in q.items()}, n=len(d)))
    m = pd.DataFrame(res).set_index("month")
    s = m["spread"]
    return dict(months=len(m), spread=float(s.mean()), t=float(s.mean() / (s.std(ddof=1) / np.sqrt(len(s)))),
                positive=float((s > 0).mean()), quintiles=[float(m[f"q{i + 1}"].mean()) for i in range(n_q)],
                yearly={int(k): float(v) for k, v in s.groupby(s.index.year).sum().items()}, table=m)


def load_prices(cfg: dict):
    st = data.PriceStore(cfg["data"]["price_dir"])
    p = st.load(fields=["close", "volume"], start="2012-01-01")
    close = p["close"].astype("float64")
    return close, close * p["volume"]


def _stamp(cfg: dict) -> str:
    f = edinet.load(cfg)
    return "none" if f is None else f"{len(f)}-{f['filed'].max()}-{pd.Timestamp.today().year}"  # 年が変わったらその年の分を学習し直す


@functools.lru_cache(maxsize=2)
def _cached(path: str, stamp: str):
    return pd.read_parquet(path)


def predictions(cfg: dict, store=None, uni: pd.DataFrame | None = None) -> pd.DataFrame | None:
    """各社・各期の予想（年ごとの学習し直しで先読みなし）。EDINET のデータが増えたときだけ計算し直す。
    n_adj = 今の株価の基準（分割調整後）での株数。時価総額 = 調整後の株価 × n_adj。"""
    path = cfg["data"]["db_path"].parent / "forecast.parquet"
    meta = path.with_suffix(".json")
    stamp = _stamp(cfg)
    if stamp == "none":
        return None
    if path.exists() and meta.exists() and json.loads(meta.read_text()).get("stamp") == stamp:
        return _cached(str(path), stamp)
    store = store or data.PriceStore(cfg["data"]["price_dir"])
    if uni is None:
        from .data import refresh_universe
        from .db import connect
        uni = refresh_universe(connect(cfg["data"]["db_path"]), cfg)
    p = store.load(fields=["close"], start="2012-01-01")
    close = p["close"].astype("float64")
    log.info("企業の利益の予想を作り直しています")
    pr = walk_forward(add_market(cfg, panel(cfg), close, uni))
    f = edinet.load(cfg)
    f = f[f["eps"].notna() & (f["eps"].abs() > 0)].assign(shares=lambda x: x["profit"] / x["eps"])
    pr = pr.merge(f.loc[f["shares"] > 0, ["code", "period_end", "shares"]], on=["code", "period_end"], how="left")
    fac = store.split_factor(close.index, close.columns) if hasattr(store, "split_factor") else None
    fd = pd.to_datetime(pr["filed"].str[:10], errors="coerce").fillna(pr["known"])
    if fac is not None:
        pos = np.clip(close.index.searchsorted(fd.to_numpy(), side="right") - 1, 0, None)
        pr["n_adj"] = pr["shares"].to_numpy() * fac.to_numpy()[pos, close.columns.get_indexer(pr["code"])]
    else:
        pr["n_adj"] = pr["shares"]
    pr["fc_chg"] = pr["pred"] * pr["sales"]  # 予想した利益の増減（円）
    pr["fc_profit"] = pr["profit"] + pr["fc_chg"]  # 予想した来期の利益（円）
    out = pr[["code", "period_end", "known", "pred", "fc_chg", "fc_profit", "n_adj"]].dropna(subset=["n_adj"])
    out.to_parquet(path)
    meta.write_text(json.dumps({"stamp": stamp, "rows": len(out)}))
    return out


def wide_features(pr: pd.DataFrame | None, close: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """日付×銘柄の特徴量。予想は次の期の予想が出るまで（最長300営業日）使い、その日の株価で時価総額を計算し直す。"""
    idx, cols = close.index, close.columns
    empty = lambda: pd.DataFrame(np.nan, index=idx, columns=cols, dtype="float32")
    if pr is None or pr.empty:
        return {k: empty() for k in FC_FEATURES}
    ev = pr[pr["code"].isin(set(cols)) & (pr["known"] <= idx[-1])]
    ev = ev.assign(pos=idx.searchsorted(ev["known"].to_numpy())).sort_values("known").drop_duplicates(["code", "pos"], keep="last")
    w = {}
    for k in ("fc_chg", "fc_profit", "n_adj"):
        x = ev.pivot(index="pos", columns="code", values=k).reindex(index=range(len(idx)), columns=cols).ffill(limit=300)
        x.index = idx
        w[k] = x
    mcap = close * w["n_adj"]
    mcap = mcap.where(mcap > 0)
    return {"fc_dey": (w["fc_chg"] / mcap).clip(-0.5, 0.5).astype("float32"),
            "fc_fey": (w["fc_profit"] / mcap).clip(-0.5, 0.5).astype("float32")}
