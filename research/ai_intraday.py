"""検証: 本番のAIを場中（10:30）の値段で計算し直して、その場で買ったらどうなるか（ユーザーの依頼「場中にリアルタイムで考えるシステム」、ジャーナル §87）。

事前に決めた条件（結果を見て変えない）:
  - 場中版: 日 D の 10:30 時点の値段（終値 = 10:25 に始まる5分足の終値、高値・安値 = それまでの5分足の範囲、
    出来高 = 日足の出来高 × 10:30 までの5分足の出来高の割合）でシグナル表を作り、手元の本番のAIで点を付ける。
    上位2を 10:30 に始まる5分足の始値で買う（無ければ 10:45 までの最初の足の始値。それも無ければ買わない）
  - 今の方式: D−1 の引けの値段で点を付け、上位2を D の寄付で買う
  - どちらも D+4 の引けで売る（売りの日をそろえる）。コスト 0.2%（参考 0.3%）
  - 相場全体の向きを外すため、同じ期間の日経225連動ETF（1321）の値動きを引いた値も出す（ETF の寄付・引けは5分足の最初・最後の足で代用）
  - 日を単位に平均し t 値を出す。前半・後半に分けて見る
5分足は 2026年8月〜（約60日）しかないので、統計的な結論は出せない前提。
"""
from __future__ import annotations

import _lab as lab
import numpy as np
import pandas as pd

from kabuscout import data, learner, pool
from kabuscout.config import load_config
from kabuscout.db import connect
from kabuscout.pipeline import dataset, store_of

SNAP = "10:30"
ETF = "1321"


def snapshot(b: pd.DataFrame, d: pd.Timestamp) -> pd.DataFrame:
    cut = d + pd.Timedelta(SNAP + ":00") - pd.Timedelta(minutes=5)
    upto = b[b["datetime"] <= cut].sort_values("datetime")
    g = upto.groupby("code")
    tot = b.groupby("code")["volume"].sum()
    return pd.DataFrame({"close": g["close"].last(), "high": g["high"].max(), "low": g["low"].min(),
                         "vol_frac": (g["volume"].sum() / tot.replace(0, np.nan)).clip(0, 1)})


def entry_price(b: pd.DataFrame, d: pd.Timestamp) -> pd.Series:
    t0 = d + pd.Timedelta(SNAP + ":00")
    w = b[(b["datetime"] >= t0) & (b["datetime"] <= t0 + pd.Timedelta(minutes=15))].sort_values("datetime")
    return w.groupby("code")["open"].first()


def tstat(x: pd.Series) -> str:
    x = x.dropna()
    return f"{x.mean():+.2%}（t={x.mean() / x.std(ddof=1) * np.sqrt(len(x)):+.1f}）" if len(x) > 2 else "—"


def main():
    lab.setup_logging()
    cfg = load_config()
    root = cfg["data"]["db_path"].parent
    bars = pd.concat([pd.read_parquet(f) for f in sorted((root / "intraday" / "5m").glob("*.parquet"))], ignore_index=True)
    bars["code"] = bars["code"].astype(str)
    bars["day"] = bars["datetime"].dt.normalize()
    store, uni = store_of(cfg), lab.universe(cfg)
    px = store.load(fields=["open", "close"], codes=uni["code"].tolist(), start="2026-07-01")
    idx = px["close"].index
    # 1321 は日足の保存先に無いので、5分足の最初の足の始値＝寄付、最後の足の終値＝引けの代わりにする
    eb = bars[bars["code"] == ETF].sort_values("datetime").groupby("day")
    etf_open, etf_close = eb["open"].first(), eb["close"].last()
    days = [d for d in sorted(set(bars["day"]) & set(idx)) if 1 <= idx.get_loc(d) and idx.get_loc(d) + 4 < len(idx)]
    con = connect(cfg["data"]["db_path"])
    bundle = learner.load_active(con, cfg)
    weights = learner.load_strategy_weights(con)
    # 今の方式: 引けの値段のシグナル表を一度だけ作る
    first, last = idx[idx.get_loc(days[0]) - 1], days[-1]
    _, base = pool.build_window(store, cfg, uni, (first - pd.Timedelta(days=pool.WARMUP_DAYS)).strftime("%Y-%m-%d"),
                                (last + pd.Timedelta(days=1)).strftime("%Y-%m-%d"), keep_from=first.strftime("%Y-%m-%d"))
    orig = data.load_panel
    rows = []
    for d in days:
        i = idx.get_loc(d)
        b = bars[bars["day"] == d]
        prev = idx[i - 1]
        now_top = learner.score(dataset(cfg, base[base["date"] == prev]), bundle, weights).head(2)["code"].tolist()
        snap = snapshot(b, d)

        def patched(store_, start=None, end=None, codes=None, _d=d, _s=snap):
            p = orig(store_, start=start, end=end, codes=codes)
            if _d in p["close"].index:
                s = _s.reindex(p["close"].columns)
                for k in ("close", "high", "low"):
                    p[k].loc[_d] = s[k].where(s[k].notna(), p[k].loc[_d])
                p["volume"].loc[_d] = p["volume"].loc[_d] * s["vol_frac"].fillna(1.0)
            return p
        pool.data.load_panel = patched
        try:
            _, pl = pool.build_window(store, cfg, uni, (d - pd.Timedelta(days=pool.WARMUP_DAYS)).strftime("%Y-%m-%d"),
                                      (d + pd.Timedelta(days=1)).strftime("%Y-%m-%d"), keep_from=d.strftime("%Y-%m-%d"))
        finally:
            pool.data.load_panel = orig
        live = learner.score(dataset(cfg, pl[pl["date"] == d]), bundle, weights)
        ent = entry_price(b, d)
        live_top = [c for c in live["code"] if c in ent.index][:2]
        exit_ = px["close"].iloc[i + 4]
        op = px["open"].iloc[i]
        etf_now = etf_close.get(idx[i + 4], np.nan) / etf_open.get(d, np.nan) - 1
        etf_live = etf_close.get(idx[i + 4], np.nan) / ent.get(ETF, np.nan) - 1
        r_now = float(np.nanmean([exit_[c] / op[c] - 1 for c in now_top])) if now_top else np.nan
        r_live = float(np.nanmean([exit_[c] / ent[c] - 1 for c in live_top])) if live_top else np.nan
        rows.append(dict(day=d, now=r_now, live=r_live, now_x=r_now - etf_now, live_x=r_live - etf_live,
                         same=len(set(now_top) & set(live_top)) / 2, now_top=",".join(now_top), live_top=",".join(live_top)))
        lab.log.info("%s 今 %+.2f%% 場中 %+.2f%% 重なり %.0f%%", d.date(), r_now * 100, r_live * 100, rows[-1]["same"] * 100)
    df = pd.DataFrame(rows)
    half = len(df) // 2
    L = [f"手元の本番のAI（{bundle['version']}）、{len(df)}日（{df['day'].min():%m/%d}〜{df['day'].max():%m/%d}）。上位2を D+4 の引けで売る", "",
         "| 区分 | 今（前日の引けで計算→寄付で買い） | 場中（10:30 で計算→10:30 に買い） | 差（場中−今） |", "|---|---|---|---|"]
    for name, m in (("全部", slice(None)), ("前半", slice(0, half)), ("後半", slice(half, None))):
        x = df.iloc[m]
        for cost in (0.002, 0.003):
            L.append(f"| {name}・コスト{cost:.1%} | {tstat(x['now'] - cost)} | {tstat(x['live'] - cost)} | {tstat(x['live'] - x['now'])} |")
        L.append(f"| {name}・日経225ETFを引く | {tstat(x['now_x'])} | {tstat(x['live_x'])} | {tstat(x['live_x'] - x['now_x'])} |")
    L += ["", f"- 上位2の重なり（今と場中）平均 {df['same'].mean():.0%}"]
    body = "\n".join(L)
    pth = lab.save_result(cfg, "ai_intraday", dict(rows=df.to_dict("records")), body)
    print("\n" + body + f"\n\n保存: {pth}")


if __name__ == "__main__":
    main()
