"""検証: 決算好反応を「情報が終値に入った最初の日」に拾う（場中・寄り前・休日の発表は1日早く）。

今の本番（kabuscout/earnings.py の wide_features）は、発表の時刻に関係なく
  反応 = 発表日の前日の終値 → 発表日の「翌営業日」の終値、使えるのは翌営業日の引け後から
としている。15時以降の発表ならこれで最短だが、取引時間中（決算好反応の約19%、§58）・寄り前・休日の発表は、
発表日（休日なら次の営業日）の終値で反応がもう分かるので、買いが1日遅い。
§58 は発表時刻を「特徴量」に足しただけで、シグナルの日付は動かしていない。§69 では買いを遅らせると大きく悪化した。

比べる2つ（事前に決めた。条件・売買ルールは決算好反応の今のまま・調整しない）:
  base  = 今と同じ（k = 発表日+1）
  early = TDnet の決算短信の時刻が分かり、15時より前か休日の発表なら k = 発表日（休日なら次の営業日）。
          反応 = 前日の終値 → k の終値。時刻が分からない（Yahoo だけ・2010年より前）発表は今と同じ
          15:00〜15:29 の発表は 2024-11-05 以降は取引時間中だが、events.known_dates と同じく翌営業日扱い（控えめ側）

使い方（リポジトリの直下で）:
  .venv/bin/python research/earn_timing.py --no-ai          # 数分: シグナル表を2つ作り、網そのものの成績だけ見る
  .venv/bin/python research/earn_timing.py --seeds 0,1      # ＋AI・口座（2つ × 種2つで約40分）
結果: data/research_cache/result_earn_timing.md（ジャーナルに貼れる形）

読むところ:
  1) 「同じ発表を両方が初日に拾った組」の差（early − base）。ここがプラスで t が立たなければ、口座を見ても意味がない
  2) early で初日に拾った候補の手取りが、base より落ちていないか（1日の反応だけで判定するので、質が落ちる可能性がある）
  3) AI・口座は最後。上位2の1回平均と順位相関が動かないなら、年率の差は経路のぶれ（§80）
"""
from __future__ import annotations

import argparse

import _lab as lab
import numpy as np
import pandas as pd

from kabuscout import earnings, pool, tdnet
from kabuscout.config import load_config
from kabuscout.pipeline import store_of

_ORIG = earnings.wide_features
STATE = dict(mode="base", times={})  # times: (code, 発表日) → その日の一番早い決算短信の時刻


def announce_times(td: pd.DataFrame | None) -> dict:
    if td is None or td.empty:
        return {}
    e = td[td["events"].str.contains("earnings")]
    first = e.groupby([e["code"], e["pubdate"].dt.normalize()])["pubdate"].min()
    return {(c, d): t for (c, d), t in first.items()}


def is_early(pub, d: pd.Timestamp, idx: pd.DatetimeIndex, i0: int) -> bool:
    """発表日（休日なら次の営業日）の終値に反応が入っているか。時刻が分からなければ False（今と同じ扱い）。"""
    if pub is None or i0 >= len(idx):
        return False
    return bool(idx[i0] != d or pub.hour < 15)


def timed_wide(ev, close, volume, mkt, max_age: int = 60):
    """earnings.wide_features の差し替え。mode="base" なら元と同じ値（＋ earn_early の印）、"early" なら早い発表を1日早く使う。"""
    out = _ORIG(ev, close, volume, mkt, max_age)
    idx, cols = close.index, close.columns
    out["earn_early"] = pd.DataFrame(np.nan, index=idx, columns=cols, dtype="float32")
    if ev is None or ev.empty:
        return out
    ev = ev[ev["code"].isin(set(cols)) & (ev["date"] >= idx[0] - pd.Timedelta(days=120))]
    c, v, m = close.to_numpy(), volume.to_numpy(), mkt.to_numpy()
    cix = {k: i for i, k in enumerate(cols)}
    T = len(idx)
    react = np.full((T, len(cols)), np.nan, dtype="float32")
    surp, vol, known, early = react.copy(), react.copy(), react.copy(), react.copy()
    vavg = pd.DataFrame(v).rolling(20, min_periods=10).mean().shift(1).to_numpy()
    times, shift = STATE["times"], STATE["mode"] == "early"
    for code, d, s in ev[["code", "date", "surprise"]].itertuples(index=False):
        j = cix[code]
        i0 = idx.searchsorted(d)
        e = is_early(times.get((code, pd.Timestamp(d))), pd.Timestamp(d), idx, i0)
        k = i0 if (e and shift) else i0 + 1
        if i0 < 1 or k >= T:
            continue
        react[k, j] = c[k, j] / c[i0 - 1, j] - 1 - (m[k] / m[i0 - 1] - 1)
        surp[k, j] = s if s == s else np.nan
        vol[k, j] = np.nanmax([v[i0, j], v[k, j]]) / vavg[i0, j] if vavg[i0, j] and vavg[i0, j] == vavg[i0, j] else np.nan
        known[k, j] = k
        early[k, j] = 1.0 if e else 0.0
    f = lambda a: pd.DataFrame(a, index=idx, columns=cols).ffill(limit=max_age)
    kn = f(known)
    out["earn_react"], out["earn_surprise"], out["earn_vol"] = (f(a).astype("float32") for a in (react, surp, vol))
    out["earn_age"] = (pd.DataFrame(np.arange(T)[:, None].repeat(len(cols), 1), index=idx, columns=cols) - kn).where(kn.notna()).astype("float32")
    out["earn_early"] = f(early).astype("float32")
    return out


def install(cfg: dict) -> None:
    earnings.wide_features = timed_wide
    if "earn_early" not in pool.FIELDS:
        pool.FIELDS.append("earn_early")
    STATE["times"] = announce_times(tdnet.load(cfg))


def diagnostics(cfg: dict, dB: pd.DataFrame, dE: pd.DataFrame, cal: pd.DatetimeIndex) -> tuple[dict, str]:
    cost = cfg["research"]["cost"]
    pead = lambda d: d["s_earnings_drift"] == 1
    first = lambda d: pead(d) & (d["earn_age"] == 0)
    res, L = {}, ["### 網そのものの成績（決算好反応・AI なし・実際の売買ルール・コスト込み）", "", lab.STATS_HEAD]
    rows = [
        ("全部・base", dB, pead(dB)), ("全部・early", dE, pead(dE)),
        ("早い発表の初日・base（買いは発表の2営業日後の寄付）", dB, first(dB) & (dB["earn_early"] == 1)),
        ("早い発表の初日・early（買いは発表の翌営業日の寄付）", dE, first(dE) & (dE["earn_early"] == 1)),
        ("15時以降の発表の初日（両方同じはず）・base", dB, first(dB) & (dB["earn_early"] == 0)),
        ("15時以降の発表の初日・early", dE, first(dE) & (dE["earn_early"] == 0)),
    ]
    for label, d, m in rows:
        res[label] = s = lab.signal_stats(d, m, cost)
        L.append(lab.fmt_stats(label, s))
    share = float((dB.loc[pead(dB), "earn_early"] == 1).mean()) if pead(dB).any() else float("nan")
    L += ["", f"- 決算好反応の候補のうち「早い発表」（15時より前・休日）の割合: {share:.0%}（§58 の取引時間中 19% より広い定義）"]

    # 同じ発表を両方が初日に拾った組: early の初日 = base の初日の1営業日前
    nxt = pd.Series(cal[1:], index=cal[:-1])
    a = dE[first(dE) & (dE["earn_early"] == 1)][["date", "code", "ret_plan"]].assign(date=lambda x: x["date"].map(nxt))
    b = dB[first(dB) & (dB["earn_early"] == 1)][["date", "code", "ret_plan"]]
    mt = a.merge(b, on=["date", "code"], suffixes=("_early", "_base")).dropna()
    res["matched_n"] = int(len(mt))
    if len(mt):
        diff = mt["ret_plan_early"] - mt["ret_plan_base"]
        by = diff.groupby(mt["date"].dt.year).mean()
        res["matched"] = dict(mean=float(diff.mean()), t_years=lab._t(by), pos_years=float((by > 0).mean()), years=int(len(by)),
                              early=float(mt["ret_plan_early"].mean() - cost), base=float(mt["ret_plan_base"].mean() - cost),
                              yearly={int(k): float(x) for k, x in by.items()})
        r = res["matched"]
        L.append(f"- **同じ発表を両方が初日に拾った組 {len(mt):,}件**: early {r['early']:+.2%} / base {r['base']:+.2%}、"
                 f"差 {r['mean']:+.2%}/回（年ごと t={r['t_years']:+.2f}、プラスの年 {r['pos_years']:.0%}・{r['years']}年）")
        L.append("  - 年ごとの差: " + " ".join(f"{y}:{x:+.1%}" for y, x in r["yearly"].items()))
    only_e, only_b = len(a) - len(mt), len(b) - len(mt)
    L.append(f"- 片方だけが初日に拾った発表: early だけ {only_e:,}件・base だけ {only_b:,}件"
             "（1日の反応で +5% に届いたが2日では届かない／その逆。early だけの分の質は上の表の差に出る）")
    return res, "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--no-ai", action="store_true", help="シグナル表と網の成績だけ（数分）")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--start-year", type=int)
    ap.add_argument("--rebuild", action="store_true", help="保存済みのシグナル表・予測を使わずに作り直す")
    a = ap.parse_args()
    lab.setup_logging()
    cfg = load_config()
    store, uni = store_of(cfg), lab.universe(cfg)
    cal = lab.calendar(cfg, store)
    install(cfg)
    lab.log.info("決算短信の時刻が分かる発表: %d件", len(STATE["times"]))
    pools = {}
    for mode in ("base", "early"):
        STATE["mode"] = mode
        pools[mode] = lab.dataset(cfg, lab.build_pool(cfg, store, uni, f"earn_{mode}", a.rebuild))
    res, text = diagnostics(cfg, lab.outcomes(cfg, pools["base"]), lab.outcomes(cfg, pools["early"]), cal)
    out = dict(diagnostics=res)
    if not a.no_ai:
        seeds = [int(s) for s in a.seeds.split(",")]
        sy = a.start_year or cfg["backtest"]["start_year"]
        arms = {k: lab.run_arm(cfg, pools[k], cal, f"earn_{k}", seeds, sy, a.rebuild) for k in ("base", "early")}
        out["ai"] = {k: [dict(seed=r["seed"], stats=r["stats"], since=r.get("since"), cand=r["cand"], yearly=r["yearly"]) for r in v]
                     for k, v in arms.items()}
        out["paired"] = lab.paired(arms["base"], arms["early"])
        text += "\n\n" + lab.report_ai(f"AI・口座（{sy}年〜、本番と同じ設定）", arms, "base")
    p = lab.save_result(cfg, "earn_timing", out, text)
    print("\n" + text + f"\n\n保存: {p}")


if __name__ == "__main__":
    main()
