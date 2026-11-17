"""検証: 本番のAIの口座は荒れた相場でどうなるか、リスクを減らす方法は効くか（ユーザーの依頼、ジャーナル §86）。

予測は本番と同じAIの年ごとの学び直し（research/earn_timing.py の base、2008年〜、種0・1）。口座は本番と同じ（backtest.simulate）。
比べる方法（事前に決めた。調整しない）:
  休む: 急落日（日経 −2%以下）とその後2営業日は新規の買いをしない（保有はそのまま）
  ヘッジ（日経平均の売りで値動きを打ち消す。先物・ベア型ETFの想定。持っている株の金額 × 割合だけ売る。費用は売った金額の年0.5%）
    いつも半分 / 弱い相場だけ全部（日経が75日線の下）/ 急落日の後5営業日だけ全部
見るところ: 年率・最大下落・最悪の年、大きな出来事の最中（regimes.EVENTS）と最近（2026年9月〜）の口座の増減
"""
from __future__ import annotations

import _lab as lab
import numpy as np
import pandas as pd

from kabuscout import backtest
from kabuscout.config import load_config
from kabuscout.pipeline import store_of
from kabuscout.regimes import EVENTS

HEDGE_CARRY = 0.005


def stats(eq: pd.Series) -> dict:
    r = eq.pct_change().dropna()
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    ends = eq.groupby(eq.index.year).last()
    yearly = pd.Series(ends.to_numpy() / np.r_[eq.iloc[0], ends.to_numpy()[:-1]] - 1, index=ends.index)
    return dict(cagr=float((eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1), dd=float((eq / eq.cummax() - 1).min()),
                worst=float(yearly.min()), sharpe=float(r.mean() / r.std() * np.sqrt(250)), yearly=yearly)


def window_ret(eq: pd.Series, a, b) -> float:
    x = eq[(eq.index >= pd.Timestamp(a)) & (eq.index <= pd.Timestamp(b))]
    return float(x.iloc[-1] / eq[eq.index < x.index[0]].iloc[-1] - 1) if len(x) and (eq.index < x.index[0]).any() else np.nan


def overlay(eq: pd.Series, expo: pd.Series, mret: pd.Series, h: pd.Series) -> pd.Series:
    """前日の終わりの保有比率 × 割合 h だけ日経平均を売っていたときの資産。"""
    r = eq.pct_change().fillna(0)
    hh = (expo.shift(1) * h.shift(1)).fillna(0)
    m = mret.reindex(eq.index).fillna(0)
    return eq.iloc[0] * (1 + r - hh * m - hh * HEDGE_CARRY / 250).cumprod()


def main():
    lab.setup_logging()
    cfg = load_config()
    store = store_of(cfg)
    cal = lab.calendar(cfg, store)
    proxy = cfg["universe"]["market_proxy"]
    mk = store.load(fields=["close"], codes=[proxy])["close"][proxy].dropna()
    mret = mk.pct_change()
    crash = mret[mret <= -0.02].index
    after = pd.Series(0.0, index=mk.index)
    pos = {d: i for i, d in enumerate(mk.index)}
    for d in crash:
        after.iloc[pos[d]: pos[d] + 5] = 1.0          # 急落日の引けから5営業日
    weak = (mk < mk.rolling(75).mean()).astype(float)
    pause_days = set()
    for d in crash:
        pause_days |= set(mk.index[pos[d]: pos[d] + 3])
    arms = {}
    for seed in (0, 1):
        pr = pd.read_parquet(lab.cache_dir(cfg) / f"preds_earn_base_s{seed}_2008.parquet")
        sim = backtest.simulate(pr, cal[cal >= pr["date"].min()], cfg, "ai")
        eq, ex = sim["equity"], sim["exposure"]
        beta = float(np.polyfit(mret.reindex(eq.index).fillna(0), eq.pct_change().fillna(0), 1)[0])
        lab.log.info("種%d: 口座の日経平均への感応度 β=%.2f・平均の保有比率 %.0f%%", seed, beta, ex.mean() * 100)
        arms.setdefault("今の本番", []).append(eq)
        sp = backtest.simulate(pr[~pr["date"].isin(pause_days)], cal[cal >= pr["date"].min()], cfg, "ai")
        arms.setdefault("急落日と後2日は新規の買いを休む", []).append(sp["equity"])
        one = pd.Series(1.0, index=eq.index)
        arms.setdefault("ヘッジ: いつも保有の半分", []).append(overlay(eq, ex, mret, one * 0.5))
        arms.setdefault("ヘッジ: 弱い相場（75日線の下）だけ全部", []).append(overlay(eq, ex, mret, weak.reindex(eq.index).fillna(0)))
        arms.setdefault("ヘッジ: 急落日の後5日だけ全部", []).append(overlay(eq, ex, mret, after.reindex(eq.index).fillna(0)))
    base = arms["今の本番"]
    crises = [e for e in EVENTS if e.kind == "crisis" and pd.Timestamp(e.end) >= base[0].index[0]]
    L = ["### 口座全体（2008年〜、種0／種1）", "", "| 方法 | 年率 | 最大下落 | 最悪の年 | シャープ | 年ごとの差（平均, t） |", "|---|---|---|---|---|---|"]
    for k, eqs in arms.items():
        st = [stats(e) for e in eqs]
        if k == "今の本番":
            diff = ""
        else:
            d = pd.concat([s["yearly"] - b["yearly"] for s, b in zip(st, [stats(e) for e in base])], axis=1).mean(axis=1)
            diff = f"{d.mean():+.1%}（t={d.mean() / d.std(ddof=1) * np.sqrt(len(d)):+.1f}）"
        L.append(f"| {k} | {st[0]['cagr']:+.1%}／{st[1]['cagr']:+.1%} | {st[0]['dd']:.1%}／{st[1]['dd']:.1%} | {st[0]['worst']:+.1%}／{st[1]['worst']:+.1%} | "
                 f"{st[0]['sharpe']:.2f}／{st[1]['sharpe']:.2f} | {diff} |")
    L += ["", "### 大きな出来事の最中の口座の増減（種0・種1の平均）", "",
          "| 出来事 | 日経平均 | " + " | ".join(arms) + " |", "|---|---|" + "---|" * len(arms)]
    windows = [(e.name, e.start, e.end) for e in crises] + [("2026年9月〜（最近）", "2026-09-01", "2026-12-31"), ("2026年10月15日〜（最近の荒れ）", "2026-10-15", "2026-12-31")]
    for name, a, b in windows:
        m = window_ret(mk, a, b)
        vals = [np.nanmean([window_ret(e, a, b) for e in eqs]) for eqs in arms.values()]
        L.append(f"| {name} | {m:+.1%} | " + " | ".join(f"{v:+.1%}" for v in vals) + " |")
    body = "\n".join(L)
    pth = lab.save_result(cfg, "turbulence_hedge", {}, body)
    print("\n" + body + f"\n\n保存: {pth}")


if __name__ == "__main__":
    main()
