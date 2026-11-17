"""検証: 場中にリアルタイムで動くシステムなら何ができるか（ユーザーの依頼、ジャーナル §87）。5分足（2026年8月〜）だけで試す。

事前に決めた条件（結果を見て変えない。試した数 = 下の4つ × 遅れ2通り）:
  A1 場中の好材料の開示: 9:00〜14:30 に出た題名のうち、自社株買い・上方修正・増配（kabuscout/tdnet.py と同じ判定）。
     出た時刻の5分後以降に始まる最初の足の始値で買う
  A2 場中の開示への反応に乗る: 9:00〜14:30 に出た開示（題名を問わない）で、出る直前の足の終値 → 出てから10分後の足の終値が +3%以上。
     その次の足の始値で買う
  B1 朝の勢い: 9:00〜10:00 の値上がり（始値→9:55 の足の終値）が大きい上位10銘柄で、この1時間の売買代金が直近20日の1日平均の50%以上。10:00 の足の始値で買う
  B2 朝の売られすぎ（B1 の逆）: 同じ条件で値下がりの大きい上位10銘柄
  売り: その日の引け（日足の終値）と、翌日の引け。コスト 0.2%（参考 0.3%）
  共通: 直近20日（前日まで）の1日平均の売買代金 1億円以上。買う足が実際にある銘柄だけ
  「20分遅れ」版: 判断に使う値段は同じで、買う時刻だけ20分後（無料の株価データは約20分遅れのため）
  日経225連動ETF（1321）の同じ時刻からの値動きを引いた値も出す（ETF の引けは5分足の最後の足の終値で代用）。日を単位に平均して t 値。前半・後半に分ける
"""
from __future__ import annotations

import re

import _lab as lab
import numpy as np
import pandas as pd

from kabuscout.config import load_config
from kabuscout.pipeline import store_of
from kabuscout.tdnet import EVENT_RULES, EXCLUDE

ETF = "1321"
GOOD = re.compile("|".join(p for k, p in EVENT_RULES if k in ("buyback", "forecast_up", "dividend_up")))


def first_open(bd: pd.DataFrame, code: str, t: pd.Timestamp, within: int = 15):
    w = bd[(bd["code"] == code) & (bd["datetime"] >= t) & (bd["datetime"] <= t + pd.Timedelta(minutes=within))]
    return (float(w["open"].iloc[0]), w["datetime"].iloc[0]) if len(w) else (np.nan, None)


def etf_at(bd: pd.DataFrame, t) -> float:
    w = bd[(bd["code"] == ETF) & (bd["datetime"] >= t)]
    return float(w["open"].iloc[0]) if len(w) else np.nan


def tstat(x: pd.Series) -> str:
    x = x.dropna()
    return f"{x.mean():+.2%}（t={x.mean() / x.std(ddof=1) * np.sqrt(len(x)):+.1f}、{len(x)}日）" if len(x) > 2 else f"—（{len(x)}日）"


def main():
    lab.setup_logging()
    cfg = load_config()
    root = cfg["data"]["db_path"].parent
    bars = pd.concat([pd.read_parquet(f) for f in sorted((root / "intraday" / "5m").glob("*.parquet"))], ignore_index=True)
    bars["code"] = bars["code"].astype(str)
    bars["day"] = bars["datetime"].dt.normalize()
    bars = bars[bars["datetime"].dt.time < pd.Timestamp("15:25").time()]
    store, uni = store_of(cfg), lab.universe(cfg)
    p = store.load(fields=["open", "close", "volume"], codes=uni["code"].tolist() + [ETF], start="2026-06-01")
    close = p["close"]
    tv = (close * p["volume"]).rolling(20, min_periods=10).mean().shift(1)
    idx = close.index
    titles = pd.read_parquet(root / "tdnet_all.parquet", columns=["pubdate", "code", "title"])
    titles["code"] = titles["code"].astype(str).str[:4]
    days = [d for d in sorted(set(bars["day"]) & set(idx)) if idx.get_loc(d) + 1 < len(idx)]
    # 1321 は日足の保存先に無いので、5分足の最後の足（15:20 の足）の終値をその日の引けの代わりにする
    etf_close = bars[bars["code"] == ETF].sort_values("datetime").groupby("day")["close"].last()
    trades = []
    for d in days:
        i = idx.get_loc(d)
        bd = bars[bars["day"] == d]
        liquid = set(tv.iloc[i][tv.iloc[i] >= 1e8].index) - {ETF}
        c0, c1 = close.iloc[i], close.iloc[i + 1]
        sig = []   # (戦略, 銘柄, 判断の時刻)
        tt = titles[(titles["pubdate"] >= d + pd.Timedelta("9h")) & (titles["pubdate"] <= d + pd.Timedelta("14h30min"))
                    & titles["code"].isin(liquid) & ~titles["title"].str.contains(EXCLUDE)]
        for code, g in tt.groupby("code"):
            t = g["pubdate"].min()
            if g["title"].str.contains(GOOD).any():
                sig.append(("A1 場中の好材料の開示", code, g.loc[g["title"].str.contains(GOOD), "pubdate"].min() + pd.Timedelta(minutes=5)))
            b = bd[bd["code"] == code].set_index("datetime")["close"]
            before = b[b.index < t - pd.Timedelta(minutes=4)]       # 出る直前に終わった足
            after = b[b.index <= t + pd.Timedelta(minutes=5)]       # 出てから10分後に終わる足まで
            if len(before) and len(after) and after.index[-1] >= t:
                if after.iloc[-1] / before.iloc[-1] - 1 >= 0.03:
                    sig.append(("A2 開示への反応（+3%以上）に乗る", code, after.index[-1] + pd.Timedelta(minutes=5)))
        m = bd[(bd["datetime"] < d + pd.Timedelta("10h")) & bd["code"].isin(liquid)].sort_values("datetime")
        g = m.groupby("code")
        x = pd.DataFrame({"o": g["open"].first(), "c": g["close"].last(), "t0": g["datetime"].first(), "t1": g["datetime"].last(),
                          "val": (m["close"] * m["volume"]).groupby(m["code"]).sum()})
        x = x[(x["t0"] <= d + pd.Timedelta("9h05min")) & (x["t1"] >= d + pd.Timedelta("9h50min"))]
        x["r"] = x["c"] / x["o"] - 1
        x = x[x["val"] >= 0.5 * tv.iloc[i].reindex(x.index)]
        t10 = d + pd.Timedelta("10h")
        sig += [("B1 朝の勢い（上位10）", c, t10) for c in x.nlargest(10, "r").index]
        sig += [("B2 朝の売られすぎ（下位10）", c, t10) for c in x.nsmallest(10, "r").index]
        for strat, code, t in sig:
            for delay in (0, 20):
                te = t + pd.Timedelta(minutes=delay)
                if te >= d + pd.Timedelta("15h20min"):
                    continue
                e, et = first_open(bd, code, te)
                if not np.isfinite(e) or code not in c0.index:
                    continue
                ee = etf_at(bd, et)
                trades.append(dict(day=d, strat=strat, delay=delay, code=code,
                                   r0=c0[code] / e - 1, r1=c1[code] / e - 1,
                                   x0=c0[code] / e - etf_close.get(d, np.nan) / ee, x1=c1[code] / e - etf_close.get(idx[i + 1], np.nan) / ee))
    tr = pd.DataFrame(trades)
    half = days[len(days) // 2]
    L = [f"5分足 {days[0]:%Y-%m-%d}〜{days[-1]:%Y-%m-%d}（{len(days)}営業日）。1日ごとの平均を日を単位に平均（t値・日数）", "",
         "| 戦略 | 遅れ | 回数 | 引けで売る（コスト0.2%） | 翌日の引けで売る（0.2%） | 引け・ETFを引く | 翌日引け・ETFを引く | 翌日引け（0.3%） | 前半／後半（翌日引け・0.2%） |",
         "|---|---|---|---|---|---|---|---|---|"]
    for (s, dl), g in tr.groupby(["strat", "delay"]):
        dd = g.groupby("day")[["r0", "r1", "x0", "x1"]].mean()
        L.append(f"| {s} | {'20分遅れ' if dl else 'すぐ'} | {len(g)} | {tstat(dd['r0'] - 0.002)} | {tstat(dd['r1'] - 0.002)} | {tstat(dd['x0'])} | "
                 f"{tstat(dd['x1'])} | {tstat(dd['r1'] - 0.003)} | {tstat(dd.loc[dd.index < half, 'r1'] - 0.002)}／{tstat(dd.loc[dd.index >= half, 'r1'] - 0.002)} |")
    body = "\n".join(L)
    pth = lab.save_result(cfg, "intraday_live", {}, body)
    print("\n" + body + f"\n\n保存: {pth}")


if __name__ == "__main__":
    main()
