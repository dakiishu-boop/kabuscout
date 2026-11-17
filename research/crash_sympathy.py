"""検証: 急落日の「つられ下げ」を、引け前（15時ごろ）に判定してその日の引けで買う別枠（ユーザーの案、ジャーナル §86）。

§67〜70 の続き。今回の問い:
  1) 荒れた相場（急落が何日も続く・米国の選挙・大きな出来事の最中）でも成り立つか、どこで大きく負けるか
  2) リスクを減らす方法が効くか: 日経平均の売りで値動きを打ち消す（ヘッジ）・損切・自社の開示がある銘柄や下げ最大の銘柄を外す・続落の日は買わない

事前に決めた定義（§67 と同じ。今回は調整しない）:
  急落日 D = 日経平均の終値が前日比 −2%以下（2008年〜）
  対象 = D の前日までの20日平均の売買代金が1億円以上
  つられ下げ = 下げが一番大きい3業種（震源）でなく、業種の中央値 ≥ 市場の中央値（巻き込まれた側）で、本人 ≤ 市場の中央値
  除く: その日に自社の開示が終値に入る銘柄（全題名 data/tdnet_all.parquet、15時以降は翌日扱い）・その日の下げが最大2%の銘柄
  買い = D の終値（15時に判定して引成で買う。判定に終値を使うので、実際より少し楽観的）。売り = 1・3・5営業日後の終値
  コスト = 往復 0.2%（研究の他の検証と同じ）。ヘッジ = 同じ期間の日経平均のリターンを引く（先物・ベア型ETFの売り。ヘッジのコスト +0.05%）
読み方: 急落日ごとの平均を1件として t を計算（同じ日の銘柄はまとめて動くので、銘柄ごとに数えると t が大きく出すぎる）

使い方: .venv/bin/python research/crash_sympathy.py   結果: data/research_cache/result_crash_sympathy.md
"""
from __future__ import annotations

import _lab as lab
import numpy as np
import pandas as pd

from kabuscout import events
from kabuscout.config import load_config
from kabuscout.pipeline import store_of
from kabuscout.regimes import EVENTS

COST, HEDGE_COST = 0.002, 0.0005
HOLDS = [1, 3, 5]
US_ELECTIONS = ["2008-11-04", "2010-11-02", "2012-11-06", "2014-11-04", "2016-11-08", "2018-11-06", "2020-11-03", "2022-11-08",
                "2024-11-05", "2026-11-03"]


def load(cfg):
    store, uni = store_of(cfg), lab.universe(cfg)
    proxy = cfg["universe"]["market_proxy"]
    p = store.load(fields=["open", "high", "low", "close", "volume"], codes=uni["code"].tolist() + [proxy], start="2007-01-01")
    mkt = p["close"].pop(proxy).ffill()
    for k in ("open", "high", "low", "volume"):
        p[k] = p[k].drop(columns=[proxy], errors="ignore")
    sec = uni.set_index("code")["sector33"]
    return p, mkt, sec


def disclosure_days(cfg, idx, cols) -> set:
    t = pd.read_parquet(cfg["data"]["db_path"].parent / "tdnet_all.parquet", columns=["pubdate", "code"])
    t["pubdate"] = pd.to_datetime(t["pubdate"])
    t["code"] = t["code"].astype(str).str[:4]
    t = t[t["code"].isin(set(cols))]
    k = events.known_dates(t["pubdate"], idx)
    ok = k < len(idx)
    return set(zip(t["code"].to_numpy()[ok], idx[k[ok]]))


def candidates(p, mkt, sec, disc, min_turnover=1e8):
    c = p["close"]
    r = c.pct_change(fill_method=None)
    mr = mkt.pct_change()
    tv = (c * p["volume"]).rolling(20, min_periods=10).mean().shift(1)
    crash = mr[(mr <= -0.02) & (mr.index >= "2008-01-01")].index
    idx = c.index
    rows = []
    for d in crash:
        x = pd.DataFrame({"r": r.loc[d], "tv": tv.loc[d]}).dropna()
        x = x[(x["tv"] >= min_turnover) & (x["r"].abs() < 0.3)]
        if len(x) < 100:
            continue
        x["sector"] = x.index.map(sec).fillna("?")
        med = x["r"].median()
        sm = x.groupby("sector")["r"].median()
        worst3 = set(sm.nsmallest(3).index)
        x["secmed"] = x["sector"].map(sm)
        cand = x[~x["sector"].isin(worst3) & (x["secmed"] >= med) & (x["r"] <= med)].copy()
        if cand.empty:
            continue
        cut = x["r"].quantile(0.02)
        cand["worst2"] = cand["r"] <= cut
        cand["own"] = [(code, d) in disc for code in cand.index]
        i = idx.get_loc(d)
        e = c.iloc[i][cand.index].to_numpy()
        for h in HOLDS:
            j = min(i + h, len(idx) - 1)
            cand[f"r{h}"] = c.iloc[j][cand.index].to_numpy() / e - 1 - COST if i + h < len(idx) else np.nan
            cand[f"m{h}"] = mkt.iloc[j] / mkt.iloc[i] - 1 if i + h < len(idx) else np.nan
        # 損切 −5%（5日の間に安値が entry×0.95 に触れたら、その値段か、それより下で寄り付いたら寄付で売る）
        stop = e * 0.95
        out = cand["r5"].to_numpy().copy()
        hit = np.zeros(len(cand), dtype=bool)
        for k in range(1, 6):
            if i + k >= len(idx):
                break
            lo, op = p["low"].iloc[i + k][cand.index].to_numpy(), p["open"].iloc[i + k][cand.index].to_numpy()
            now = ~hit & (lo <= stop)
            px = np.where(op <= stop, op, stop)
            out = np.where(now, px / e - 1 - COST, out)
            hit |= now
        cand["r5_stop"] = out
        cand["date"], cand["mkt_d"], cand["med_d"] = d, mr.loc[d], med
        cand["mkt_next"] = mr.iloc[i + 1] if i + 1 < len(idx) else np.nan
        cand["mkt_prev"] = mr.iloc[i - 1]
        rows.append(cand.reset_index().rename(columns={"index": "code"}))
    return pd.concat(rows, ignore_index=True)


def day_stats(v: pd.Series) -> dict:
    v = v.dropna()
    if len(v) < 3:
        return dict(n=len(v))
    return dict(n=int(len(v)), mean=float(v.mean()), t=float(v.mean() / v.std(ddof=1) * np.sqrt(len(v))), pos=float((v > 0).mean()),
                worst=float(v.min()), p10=float(v.quantile(0.1)))


def fmt(label, s):
    if s.get("n", 0) < 3:
        return f"| {label} | {s.get('n', 0)} | | | | |"
    return f"| {label} | {s['n']} | {s['mean']:+.2%}（t={s['t']:+.1f}） | {s['pos']:.0%} | {s['p10']:+.1%} | {s['worst']:+.1%} |"


HEAD = "| 区分 | 急落日の数 | 1回の手取り（日ごとの平均） | プラスの日 | 悪い方から10% | 最悪の日 |\n|---|---|---|---|---|---|"


def by_day(c: pd.DataFrame, col: str, hedge: bool = False) -> pd.Series:
    h = col.lstrip("r").split("_")[0]
    v = c[col] - (c[f"m{h}"] + HEDGE_COST if hedge else 0)
    return v.groupby(c["date"]).mean()


def sleeve(c: pd.DataFrame, n: int, col: str, hedge: bool, rule: str, seed: int = 0) -> pd.Series:
    """別枠の口座: 急落日ごとに n 銘柄を同じ金額ずつ買う（rule = 売買代金の大きい順 / くじ）。日ごとのバスケットのリターン。"""
    rng = np.random.default_rng(seed)
    out = {}
    for d, g in c.groupby("date"):
        g = g.dropna(subset=[col])
        if g.empty:
            continue
        g = g.nlargest(n, "tv") if rule == "liquid" else g.iloc[rng.permutation(len(g))[:n]]
        h = col.lstrip("r").split("_")[0]
        out[d] = float((g[col] - (g[f"m{h}"] + HEDGE_COST if hedge else 0)).mean())
    return pd.Series(out)


def equity(v: pd.Series, frac: float) -> dict:
    """資金の frac を急落日ごとに別枠に入れたときの、別枠の分だけの資金の増減（重なりは無視した近似）。"""
    eq = (1 + frac * v.sort_index()).cumprod()
    dd = float((eq / eq.cummax() - 1).min())
    yrs = (v.index.max() - v.index.min()).days / 365.25
    return dict(cagr=float(eq.iloc[-1] ** (1 / max(yrs, 1)) - 1), dd=dd, final=float(eq.iloc[-1]))


def main():
    lab.setup_logging()
    cfg = load_config()
    p, mkt, sec = load(cfg)
    disc = disclosure_days(cfg, p["close"].index, p["close"].columns)
    c = candidates(p, mkt, sec, disc)
    base = c[~c["own"] & ~c["worst2"]]
    last_title = pd.read_parquet(cfg["data"]["db_path"].parent / "tdnet_all.parquet", columns=["pubdate"])["pubdate"].max()
    L = [f"急落日 {c['date'].nunique()}日（{c['date'].min().date()}〜{c['date'].max().date()}）、つられ下げの候補 {len(c):,}行（1日平均 {len(c) / c['date'].nunique():.0f}）。"
         f"自社の開示の判定は題名のある {pd.Timestamp(last_title).date()} まで", ""]

    L += ["### 1) 何日持つか・ヘッジ・損切（自社の開示あり・下げ最大2%を除いた候補）", "", HEAD]
    for h in HOLDS:
        L.append(fmt(f"{h}日後の引けで売る", day_stats(by_day(base, f"r{h}"))))
        L.append(fmt(f"　　日経平均の売りでヘッジ", day_stats(by_day(base, f"r{h}", hedge=True))))
    L.append(fmt("5日・損切 −5%", day_stats(by_day(base, "r5_stop"))))
    L.append(fmt("　　ヘッジあり", day_stats(by_day(base, "r5_stop", hedge=True))))
    L.append(fmt("（参考）除く前の全部・5日", day_stats(by_day(c, "r5"))))
    L.append(fmt("（参考）自社の開示あり・5日", day_stats(by_day(c[c["own"]], "r5"))))
    L.append(fmt("（参考）下げ最大2%・5日", day_stats(by_day(c[c["worst2"]], "r5"))))
    L.append(fmt("（参考）同じ5日の日経平均（コスト0.2%引き）", day_stats(base.groupby("date")["m5"].first() - COST)))

    L += ["", "### 2) 荒れ方ごと（5日・ヘッジなし／ありの順）", "", HEAD]
    d5, d5h = by_day(base, "r5"), by_day(base, "r5", hedge=True)
    info = base.groupby("date")[["mkt_d", "mkt_next", "mkt_prev"]].first()
    ev = pd.Series([next((e.name for e in EVENTS if e.kind == "crisis" and pd.Timestamp(e.start) <= d <= pd.Timestamp(e.end)), "") for d in info.index], index=info.index)
    near_elec = pd.Series([any(abs((d - pd.Timestamp(x)).days) <= 10 for x in US_ELECTIONS) for d in info.index], index=info.index)
    groups = [
        ("下げ −2〜−3%", info["mkt_d"] > -0.03), ("下げ −3〜−5%", (info["mkt_d"] <= -0.03) & (info["mkt_d"] > -0.05)), ("下げ −5%以下", info["mkt_d"] <= -0.05),
        ("前日も −1%以下（続落の途中）", info["mkt_prev"] <= -0.01), ("前日は −1%より上（下げ始め）", info["mkt_prev"] > -0.01),
        ("翌日も −1%以下（結果的に続落）※事前には分からない", info["mkt_next"] <= -0.01), ("翌日は −1%より上", info["mkt_next"] > -0.01),
        ("大きな出来事の最中（regimes.EVENTS）", ev != ""), ("それ以外", ev == ""),
        ("米国の選挙の前後10日", near_elec),
        ("2008〜2012年", info.index.year <= 2012), ("2013〜2019年", (info.index.year >= 2013) & (info.index.year <= 2019)),
        ("2020〜2026年", info.index.year >= 2020), ("2026年9月〜（最近）", info.index >= "2026-09-01"),
    ]
    for label, m in groups:
        m = pd.Series(np.asarray(m, dtype=bool), index=info.index)
        L.append(fmt(label, day_stats(d5[m.reindex(d5.index).fillna(False)])))
        L.append(fmt("　　ヘッジあり", day_stats(d5h[m.reindex(d5h.index).fillna(False)])))
    rec = info[info.index >= "2026-09-01"]
    if len(rec):
        L += ["", "- 最近の急落日（日経平均の下げ → 候補の5日後・ヘッジなし／あり）: " + "、".join(
            f"{d:%m/%d} {info.loc[d, 'mkt_d']:+.1%} → {d5.get(d, np.nan):+.1%}／{d5h.get(d, np.nan):+.1%}" for d in rec.index)]
    elec = info[near_elec]
    L += ["- 米国の選挙の前後10日の急落日: " + ("、".join(f"{d:%Y-%m-%d} {info.loc[d, 'mkt_d']:+.1%} → {d5.get(d, np.nan):+.1%}" for d in elec.index) or "なし")]

    L += ["", "### 3) 別枠の口座として（急落日ごとに5銘柄を同じ金額ずつ、5日後の引けで売る。資金の20%を使う近似。2008年〜）", "",
          "| 選び方 | ヘッジ | 1回の手取り（日ごと） | 別枠の分の年率への寄与 | 別枠の分だけの最大下落 | 最悪の日 |", "|---|---|---|---|---|---|"]
    for rule, name in (("liquid", "売買代金の大きい順"), ("random", "くじ（10回の平均）")):
        for hedge in (False, True):
            if rule == "liquid":
                vs = [sleeve(base, 5, "r5", hedge, rule)]
            else:
                vs = [sleeve(base, 5, "r5", hedge, rule, s) for s in range(10)]
            st = [day_stats(v) for v in vs]
            eqs = [equity(v, 0.2) for v in vs]
            L.append(f"| {name} | {'あり' if hedge else 'なし'} | {np.mean([s['mean'] for s in st]):+.2%}（t={np.mean([s['t'] for s in st]):+.1f}） | "
                     f"{np.mean([e['cagr'] for e in eqs]):+.2%} | {np.mean([e['dd'] for e in eqs]):+.1%} | {np.mean([s['worst'] for s in st]):+.1%} |")
    v = sleeve(base, 5, "r5", False, "liquid")
    worst = v.nsmallest(5)
    L += ["", "- 別枠（売買代金の大きい順・ヘッジなし）の悪かった急落日: " + "、".join(f"{d:%Y-%m-%d}（日経 {info.loc[d, 'mkt_d']:+.1%}）{x:+.1%}" for d, x in worst.items())]
    yr = v.groupby(v.index.year).agg(["mean", "size"])
    L += ["- 年ごと（1回の手取り・回数）: " + "、".join(f"{y}: {m:+.1%}（{int(n)}）" for y, (m, n) in yr.iterrows())]
    body = "\n".join(L)
    pth = lab.save_result(cfg, "crash_sympathy", dict(n_days=int(c["date"].nunique())), body)
    base.to_parquet(lab.cache_dir(cfg) / "crash_sympathy_rows.parquet")
    print("\n" + body + f"\n\n保存: {pth}")


if __name__ == "__main__":
    main()
