"""検証: 急落日の「つられ下げ」を場中のどの時刻に買うのがよいか（5分足、ジャーナル §86）。

data/intraday/5m（Yahoo の5分足、2026年8月〜。手元だけ）で、急落日 D に時刻 T の値段だけで候補を決め（crash_sympathy.py と同じ定義を
時刻 T の騰落率で計算）、T の値段で買う版と、T に決めて引けで買う版を比べる。売りは D の引け・1日後・5日後の引け（日足）。
5分足の時刻は足の始まりなので、T の値段 = T−5分に始まる足の終値。日経平均の場中の値は日経225連動ETF（1321）で代用。
自社の開示は、D−1 の15時以降〜D の T より前の題名（data/tdnet_all.parquet）。
急落日が数日しかないので統計的な結論は出せない。日ごとの値を並べて、時刻による違いの向きと大きさを見るだけ。
"""
from __future__ import annotations

from pathlib import Path

import _lab as lab
import numpy as np
import pandas as pd

from kabuscout.config import load_config
from kabuscout.pipeline import store_of

COST = 0.002
TIMES = ["09:30", "10:30", "11:30", "13:00", "14:00", "15:00"]
ETF = "1321"


def main():
    lab.setup_logging()
    cfg = load_config()
    root = cfg["data"]["db_path"].parent
    bars = pd.concat([pd.read_parquet(f) for f in sorted((root / "intraday" / "5m").glob("*.parquet"))], ignore_index=True)
    bars["code"] = bars["code"].astype(str)
    bars["day"] = bars["datetime"].dt.normalize()
    store, uni = store_of(cfg), lab.universe(cfg)
    proxy = cfg["universe"]["market_proxy"]
    p = store.load(fields=["close", "volume"], codes=uni["code"].tolist() + [proxy], start="2026-05-01")
    close = p["close"]
    mkt = close.pop(proxy).ffill()
    tv = (close * p["volume"].drop(columns=[proxy], errors="ignore")).rolling(20, min_periods=10).mean().shift(1)
    sec = uni.set_index("code")["sector33"]
    idx = close.index
    titles = pd.read_parquet(root / "tdnet_all.parquet", columns=["pubdate", "code"])
    titles["code"] = titles["code"].astype(str).str[:4]
    mr = mkt.pct_change()
    days = sorted(set(bars["day"]) & set(idx))
    crash = [d for d in days if mr.get(d, 0) <= -0.02]
    L = [f"5分足の期間 {days[0].date()}〜{days[-1].date()}（{len(days)}営業日）、うち急落日（日経 −2%以下）: "
         + "、".join(f"{d:%m/%d} {mr[d]:+.1%}" for d in crash), ""]
    rows = []
    for d in crash:
        i = idx.get_loc(d)
        prev = close.iloc[i - 1]
        b = bars[bars["day"] == d]
        e5 = close.iloc[min(i + 5, len(idx) - 1)] if i + 5 < len(idx) else None
        e1 = close.iloc[i + 1] if i + 1 < len(idx) else None
        for T in TIMES + ["引け"]:
            if T == "引け":
                px, etf = close.iloc[i], np.nan
                t_cut = d + pd.Timedelta(hours=15)
            else:
                t_bar = d + pd.Timedelta(T + ":00") - pd.Timedelta(minutes=5)
                snap = b[b["datetime"] <= t_bar].sort_values("datetime").groupby("code")["close"].last()
                px = snap
                etf = snap.get(ETF, np.nan) / close.iloc[i - 1].get(ETF, np.nan) - 1 if ETF in close.columns else np.nan
                t_cut = d + pd.Timedelta(T + ":00")
            r = (px / prev - 1).dropna()
            x = pd.DataFrame({"r": r, "tv": tv.iloc[i].reindex(r.index)}).dropna()
            x = x[(x["tv"] >= 1e8) & (x["r"].abs() < 0.3)]
            if len(x) < 100:
                continue
            x["sector"] = x.index.map(sec).fillna("?")
            med = x["r"].median()
            sm = x.groupby("sector")["r"].median()
            x["secmed"] = x["sector"].map(sm)
            cand = x[~x["sector"].isin(set(sm.nsmallest(3).index)) & (x["secmed"] >= med) & (x["r"] <= med)]
            own = set(titles.loc[(titles["pubdate"] >= idx[i - 1] + pd.Timedelta(hours=15)) & (titles["pubdate"] < t_cut), "code"])
            cand = cand[~cand.index.isin(own) & (cand["r"] > x["r"].quantile(0.02))]
            codes = cand.index
            entry = px.reindex(codes) if T != "引け" else close.iloc[i][codes]
            res = dict(day=d, T=T, n=len(codes), mkt_at_T=med, etf_at_T=etf,
                       to_close=float((close.iloc[i][codes] / entry - 1).mean()),
                       buyT_1d=float((e1[codes] / entry - 1 - COST).mean()) if e1 is not None else np.nan,
                       buyT_5d=float((e5[codes] / entry - 1 - COST).mean()) if e5 is not None else np.nan,
                       decideT_buyclose_5d=float((e5[codes] / close.iloc[i][codes] - 1 - COST).mean()) if e5 is not None else np.nan,
                       nikkei_5d=float(mkt.iloc[min(i + 5, len(idx) - 1)] / mkt.iloc[i] - 1) if i + 5 < len(idx) else np.nan)
            rows.append(res)
    df = pd.DataFrame(rows)
    L += ["| 急落日 | 判定の時刻 | 候補 | その時刻の市場の中央値 | 買ってから引けまで | T で買い→翌日 | T で買い→5日後 | T で決めて引けで買い→5日後 | 日経（引け→5日後） |",
          "|---|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        L.append(f"| {r['day']:%m/%d} | {r['T']} | {r['n']} | {r['mkt_at_T']:+.1%} | {r['to_close']:+.2%} | {r['buyT_1d']:+.2%} | {r['buyT_5d']:+.2%} | "
                 f"{r['decideT_buyclose_5d']:+.2%} | {r['nikkei_5d']:+.2%} |")
    avg = df.groupby("T", sort=False)[["n", "to_close", "buyT_1d", "buyT_5d", "decideT_buyclose_5d"]].mean()
    L += ["", "**急落日の平均（時刻ごと）**", "", "| 判定の時刻 | 候補（平均） | 買ってから引けまで | T で買い→翌日 | T で買い→5日後 | T で決めて引けで買い→5日後 |", "|---|---|---|---|---|---|"]
    L += [f"| {t} | {r['n']:.0f} | {r['to_close']:+.2%} | {r['buyT_1d']:+.2%} | {r['buyT_5d']:+.2%} | {r['decideT_buyclose_5d']:+.2%} |" for t, r in avg.iterrows()]
    body = "\n".join(L)
    pth = lab.save_result(cfg, "crash_intraday", dict(rows=df.to_dict("records")), body)
    print("\n" + body + f"\n\n保存: {pth}")


if __name__ == "__main__":
    main()
