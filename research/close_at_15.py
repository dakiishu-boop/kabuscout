"""検証: 本番のAIを引け前（15:00）の値段で計算したら、引けの値段で計算した場合とどれだけ候補が変わるか（ジャーナル §86）。

「15時に計算して、その日の引け（15:30のクロージング・オークション）で買う」の実現性の確認。
その日の値段を 15:00 時点の値に差し替えてシグナル表を作り（終値 = 14:55 に始まる5分足の終値、高値・安値 = 15:00 までの5分足の範囲、
出来高 = その日の出来高 × 15:00 までの5分足の出来高の割合）、手元の本番のAIで点を付ける。比べるのは:
  - 上位2・上位15の重なり（15時の候補のうち、引けで計算しても入っていた割合）
  - 上位2の結果: 今の買い方（引けで計算 → 翌日の寄付で買う）と、15時に計算 → その日の引けで買う、の5日後の引けまでの値動き（コスト 0.2% 引き）
5分足は 2026年8月〜なので、日数は少ない（急落日 ＋ ふつうの日を何日か）。
"""
from __future__ import annotations

import _lab as lab
import numpy as np
import pandas as pd

from kabuscout import data, learner, pool
from kabuscout.config import load_config
from kabuscout.db import connect
from kabuscout.pipeline import dataset, store_of

COST = 0.002
SNAP = "15:00"


def snapshot(bars: pd.DataFrame, d: pd.Timestamp) -> pd.DataFrame:
    b = bars[bars["day"] == d]
    cut = d + pd.Timedelta(SNAP + ":00") - pd.Timedelta(minutes=5)
    upto = b[b["datetime"] <= cut].sort_values("datetime")
    g = upto.groupby("code")
    tot = b.groupby("code")["volume"].sum()
    return pd.DataFrame({"close": g["close"].last(), "high": g["high"].max(), "low": g["low"].min(),
                         "vol_frac": (g["volume"].sum() / tot.replace(0, np.nan)).clip(0, 1)})


def main():
    lab.setup_logging()
    cfg = load_config()
    root = cfg["data"]["db_path"].parent
    bars = pd.concat([pd.read_parquet(f) for f in sorted((root / "intraday" / "5m").glob("*.parquet"))], ignore_index=True)
    bars["code"] = bars["code"].astype(str)
    bars["day"] = bars["datetime"].dt.normalize()
    store, uni = store_of(cfg), lab.universe(cfg)
    proxy = cfg["universe"]["market_proxy"]
    px = store.load(fields=["open", "close"], codes=uni["code"].tolist() + [proxy], start="2026-07-01")
    mr = px["close"][proxy].pct_change()
    idx = px["close"].index
    days = [d for d in sorted(set(bars["day"]) & set(idx)) if idx.get_loc(d) + 5 < len(idx)]
    crash = [d for d in days if mr[d] <= -0.02]
    normal = [d for d in days if d not in crash]
    sample = crash + list(np.array(normal)[np.linspace(0, len(normal) - 1, 10).round().astype(int)])
    con = connect(cfg["data"]["db_path"])
    bundle = learner.load_active(con, cfg)
    weights = learner.load_strategy_weights(con)
    orig = data.load_panel
    rows = []
    for d in sorted(sample):
        snap = snapshot(bars, d)
        out = {}
        for mode in ("close", "15:00"):
            def patched(store_, start=None, end=None, codes=None, _d=d, _m=mode):
                p = orig(store_, start=start, end=end, codes=codes)
                if _m == "15:00" and _d in p["close"].index:
                    s = snap.reindex(p["close"].columns)
                    for k in ("close", "high", "low"):
                        p[k].loc[_d] = s[k].where(s[k].notna(), p[k].loc[_d])
                    p["volume"].loc[_d] = p["volume"].loc[_d] * s["vol_frac"].fillna(1.0)
                return p
            pool.data.load_panel = patched
            _, pl = pool.build_window(store, cfg, uni, (d - pd.Timedelta(days=pool.WARMUP_DAYS)).strftime("%Y-%m-%d"),
                                      (d + pd.Timedelta(days=1)).strftime("%Y-%m-%d"), keep_from=d.strftime("%Y-%m-%d"))
            pool.data.load_panel = orig
            today = dataset(cfg, pl[pl["date"] == d])
            out[mode] = learner.score(today, bundle, weights).reset_index(drop=True)
        a, b = out["close"], out["15:00"]
        i = idx.get_loc(d)
        c5, op1, c0 = px["close"].iloc[i + 5], px["open"].iloc[i + 1], px["close"].iloc[i]
        t2a, t2b = a.head(2)["code"].tolist(), b.head(2)["code"].tolist()
        rows.append(dict(day=d, crash=d in crash, mkt=mr[d], n_close=len(a), n_15=len(b),
                         top2_same=len(set(t2a) & set(t2b)) / 2, top15_same=len(set(a.head(15)["code"]) & set(b.head(15)["code"])) / 15,
                         now=float(np.nanmean([c5[c] / op1[c] - 1 - COST for c in t2a])),       # 引けで計算 → 翌日の寄付で買う（今）
                         at15=float(np.nanmean([c5[c] / c0[c] - 1 - COST for c in t2b])),       # 15時に計算 → その日の引けで買う
                         top2_close=",".join(t2a), top2_15=",".join(t2b)))
        lab.log.info("%s 上位2の重なり %.0f%%・上位15 %.0f%%", d.date(), rows[-1]["top2_same"] * 100, rows[-1]["top15_same"] * 100)
    df = pd.DataFrame(rows)
    L = [f"手元の本番のAI（{bundle['version']}）、{len(df)}日（急落日 {int(df['crash'].sum())}日）", "",
         "| 日 | 日経 | 候補の数（引け／15時） | 上位2の重なり | 上位15の重なり | 今（翌寄付で買い→5日後） | 15時に計算→引けで買い→5日後 |", "|---|---|---|---|---|---|---|"]
    L += [f"| {r['day']:%m/%d}{'（急落）' if r['crash'] else ''} | {r['mkt']:+.1%} | {r['n_close']}／{r['n_15']} | {r['top2_same']:.0%} | {r['top15_same']:.0%} | "
          f"{r['now']:+.2%} | {r['at15']:+.2%} |" for _, r in df.iterrows()]
    for name, m in (("全部", df.index == df.index), ("急落日", df["crash"]), ("ふつうの日", ~df["crash"])):
        x = df[m]
        if len(x):
            L.append(f"\n- {name}（{len(x)}日）: 上位2の重なり {x['top2_same'].mean():.0%}・上位15 {x['top15_same'].mean():.0%}、"
                     f"5日後の手取り 今 {x['now'].mean():+.2%} ／ 15時→引け {x['at15'].mean():+.2%}")
    body = "\n".join(L)
    pth = lab.save_result(cfg, "close_at_15", dict(rows=df.to_dict("records")), body)
    print("\n" + body + f"\n\n保存: {pth}")


if __name__ == "__main__":
    main()
