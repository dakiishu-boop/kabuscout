"""検証: 適時開示の「網」の取りこぼし（自社株買いの拾い漏れ・業績修正への好反応）。

背景:
- §75 の監査で、本番の題名の分類は自社株買いの本物の一部しか拾えていなかった（改良版で 拾えた割合 8→60%）。
  §75 の検証は「発表から何日か」の特徴量の作り直しで、候補（網）が増えたかどうかは記録に無い。
  自社株買いは AI の上位2で 1回 +1.38%（186件、§41）と、数は少ないが成績の良い網
- 業績予想の修正は、§71 で「特徴量」としては効かなかった。ただしラウンド1の教訓は
  「決算の情報は特徴量より網として使う方が有効」で、修正の日に AI の上位2に入った銘柄は +3.12%（152件、§70）。
  §72 の「やらないこと」に「修正をきっかけにした新戦略」があるので、理由（synthesis.md）を先に確認してから走らせること

比べる3つ（事前に決めた。調整しない）:
  base    = 全題名（data/tdnet_all.parquet、§70）を本番の tdnet.classify で分類（本番の tdnet.parquet と同じはず）
  buyback = 自社株買いの判定だけを下の BUYBACK_POS / BUYBACK_NEG に差し替え。戦略の条件・売買ルールは今の「自社株買い発表」のまま
            ※ この正規表現は Claude が題名の型から書いたもので、§75 の監査済みの改良版ではない。
              scratchpad に改良版が残っていれば、下の2つの定数をそれに置き換えてから走らせる
  rev     = 新しい網「修正好反応」: 業績予想の修正・予想と実績の差異（§71 と同じ拾い方、訂正を除く）が終値に入った最初の日から5営業日以内・
            その日の騰落率が日経平均比 +5%以上・出来高が直前20日平均の1.5倍以上（決算好反応の既定値と同じ数字）。
            売買ルールは決算好反応に採用済みのものをそのまま使う（無ければ共通ルール）。AI には印（s_rev_drift）だけ渡す

使い方（リポジトリの直下で。data/tdnet_all.parquet が要る）:
  .venv/bin/python research/event_nets.py --no-ai                 # 数分×3: 網そのものの成績・件数・題名の見本
  .venv/bin/python research/event_nets.py --seeds 0,1 --start-year 2011   # ＋AI・口座（3つ × 種2つで約1時間）
  .venv/bin/python research/event_nets.py --arms base,buyback     # 片方だけ
結果: data/research_cache/result_event_nets.md

読むところ:
  1) 新しく拾った分（「新規だけ」の行）の件数と1回の手取り。年20件に満たない・手取りが +0.3% 未満なら、口座では見えない
  2) 題名の見本（new_only_titles）を目で見て、自社株買いの決定でないものが多ければ正規表現を直す
  3) AI・口座は最後。上位2の内訳（mix）で、その網の件数と手取りが base からどう変わったか
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import _lab as lab
import numpy as np
import pandas as pd

from kabuscout import earnings, events, learner, pool, strategies, tdnet
from kabuscout.config import load_config
from kabuscout.pipeline import store_of

# ---- 自社株買い（決定・枠の設定・ToSTNeT-3 での買付けの予告）。取得状況・結果・終了・処分・他社の買付けへの応募は除く
BUYBACK_POS = re.compile(r"自己株式(の)?(取得|立会外買付|公開買付け|市場買付|買付け)|自己の株式の取得|ToSTNeT|ＴｏＳＴＮｅＴ|自社株買い")
BUYBACK_NEG = re.compile(r"取得状況|状況に関する|取得結果|結果|取得終了|終了|完了|処分|応募|訂正|中止|延期|お詫び|経過|株式報酬|譲渡制限")
# ---- 業績予想の修正（向きを問わない。§71 と同じ拾い方）
REV = re.compile(r"業績予想.*修正|上方修正|下方修正|予想.{0,8}実績.{0,8}差異")
REV_NEG = re.compile(r"訂正")
REV_FEATURES = ["rev_age", "rev_react", "rev_vol"]

STATE = dict(td=None, rev=None)
_ORIG_EARN = earnings.wide_features


def is_buyback(title: str) -> bool:
    t = title or ""
    return bool(BUYBACK_POS.search(t)) and not BUYBACK_NEG.search(t)


def is_revision(title: str) -> bool:
    t = title or ""
    return bool(REV.search(t)) and not REV_NEG.search(t)


def classify_new(title: str) -> list[str]:
    """本番の分類のうち、自社株買いだけを差し替える。"""
    ev = [e for e in tdnet.classify(title) if e != "buyback"]
    return ev + ["buyback"] if is_buyback(title) else ev


def load_titles(cfg: dict, path: str | None) -> pd.DataFrame:
    p = Path(path) if path else cfg["data"]["db_path"].parent / "tdnet_all.parquet"
    if not p.exists():
        sys.exit(f"全題名のファイルがありません: {p}\n§70 で作った data/tdnet_all.parquet（pubdate・code・title）を置くか、--titles で場所を指定してください。")
    t = pd.read_parquet(p)
    pick = lambda names: next((c for c in names if c in t.columns), None)
    cp, cc, ct = pick(["pubdate", "pub", "datetime", "time", "date"]), pick(["code", "company_code"]), pick(["title"])
    if not (cp and cc and ct):
        sys.exit(f"{p} の列が分かりません（必要: pubdate・code・title）。今の列: {list(t.columns)}")
    pub = pd.to_datetime(t[cp])
    if getattr(pub.dt, "tz", None) is not None:
        pub = pub.dt.tz_convert("Asia/Tokyo").dt.tz_localize(None)
    out = pd.DataFrame({"pubdate": pub, "code": t[cc].astype(str).str[:4], "title": t[ct].astype(str)})
    return out.dropna(subset=["pubdate"]).drop_duplicates(["pubdate", "code", "title"]).reset_index(drop=True)


def classified(titles: pd.DataFrame, fn) -> pd.DataFrame:
    ev = titles["title"].map(lambda t: ",".join(fn(t)))
    return titles.assign(events=ev)[ev != ""].reset_index(drop=True)


# ---------------------------------------------------------------- 修正好反応
def rev_wide(rev: pd.DataFrame | None, close: pd.DataFrame, volume: pd.DataFrame, mkt: pd.Series, max_age: int = 60) -> dict:
    """修正の開示が終値に入った最初の日 k（events.known_dates: 15時以降は翌営業日）での反応。k より先の値は使わない。
    rev_react = k の日の騰落率 − 日経平均、rev_vol = k の出来高 ÷ それ以前20日の平均、rev_age = k からの営業日数。"""
    idx, cols = close.index, close.columns
    T, N = len(idx), len(cols)
    out = {f: pd.DataFrame(np.nan, index=idx, columns=cols, dtype="float32") for f in REV_FEATURES}
    if rev is None or rev.empty:
        return out
    rev = rev[rev["code"].isin(set(cols)) & (rev["pubdate"] >= idx[0] - pd.Timedelta(days=100)) & (rev["pubdate"] <= idx[-1] + pd.Timedelta(days=1))]
    if rev.empty:
        return out
    cix = {k: i for i, k in enumerate(cols)}
    k = events.known_dates(rev["pubdate"], idx)
    j = rev["code"].map(cix).to_numpy()
    ok = (k >= 1) & (k < T)
    k, j = k[ok], j[ok]
    c, v, m = close.to_numpy("float64"), volume.to_numpy("float64"), mkt.to_numpy("float64")
    vavg = pd.DataFrame(v).rolling(20, min_periods=10).mean().shift(1).to_numpy()
    react = np.full((T, N), np.nan, dtype="float32")
    vol, known = react.copy(), react.copy()
    react[k, j] = c[k, j] / c[k - 1, j] - 1 - (m[k] / m[k - 1] - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        vol[k, j] = np.where(vavg[k, j] > 0, v[k, j] / vavg[k, j], np.nan)
    known[k, j] = k
    f = lambda a: pd.DataFrame(a, index=idx, columns=cols).ffill(limit=max_age)
    kn = f(known)
    out["rev_react"], out["rev_vol"] = f(react).astype("float32"), f(vol).astype("float32")
    out["rev_age"] = (pd.DataFrame(np.arange(T)[:, None].repeat(N, 1), index=idx, columns=cols) - kn).where(kn.notna()).astype("float32")
    return out


def rev_drift(i, p):
    return (i["rev_age"] >= 0) & (i["rev_age"] <= p["age"]) & (i["rev_react"] >= p["react"]) & (i["rev_vol"] >= p["vol"])


REV_STRATEGY = strategies.Strategy(
    "rev_drift", "修正好反応", "業績予想の修正に市場が好反応（市場比+5%以上・大商い）した銘柄の、発表後数日以内", rev_drift,
    dict(age=(1, 10, 5, 10), react=(0.02, 0.15, 0.05, 0.02), vol=(1.0, 3.0, 1.5, 1.0)))


def _earn_with_rev(ev, close, volume, mkt, max_age: int = 60):
    out = _ORIG_EARN(ev, close, volume, mkt, max_age)
    out.update(rev_wide(STATE["rev"], close, volume, mkt, max_age))
    return out


def register_rev(cfg: dict, titles: pd.DataFrame) -> dict:
    """網「修正好反応」をこのプロセスの中だけで足す（base・buyback を先に済ませてから呼ぶ。元には戻さない）。"""
    if "rev_drift" in strategies.BY_KEY:
        return cfg
    STATE["rev"] = titles[titles["title"].map(is_revision)][["pubdate", "code"]].reset_index(drop=True)
    earnings.wide_features = _earn_with_rev            # pool.build_window が close・volume・日経平均を渡してくれる場所
    pool.FIELDS.extend(f for f in REV_FEATURES if f not in pool.FIELDS)
    strategies.STRATEGIES.append(REV_STRATEGY)
    strategies.BY_KEY["rev_drift"] = REV_STRATEGY
    strategies.STRATEGY_NAMES["rev_drift"] = REV_STRATEGY.name
    strategies.FLAG_COLS.append("s_rev_drift")          # learner が持っているのも同じリスト
    if "s_rev_drift" not in learner.MODEL_FEATURES:
        learner.MODEL_FEATURES.append("s_rev_drift")    # AI には印だけ渡す（反応の大きさなどは §71 で効かなかった）
    per = dict(cfg["labeling"].get("per_strategy") or {})
    if "earnings_drift" in per:                         # 売買ルールは決算好反応と同じ
        per["rev_drift"] = per["earnings_drift"]
    return {**cfg, "labeling": {**cfg["labeling"], "per_strategy": per}}


# ---------------------------------------------------------------- 表
def title_report(titles: pd.DataFrame, n: int = 40) -> tuple[dict, str]:
    old = titles["title"].map(lambda t: "buyback" in tdnet.classify(t))
    new = titles["title"].map(is_buyback)
    rev = titles["title"].map(is_revision)
    yr = titles["pubdate"].dt.year
    tab = pd.DataFrame({"自社株買い・今": old.groupby(yr).sum(), "自社株買い・差し替え": new.groupby(yr).sum(),
                        "新しく拾う": (new & ~old).groupby(yr).sum(), "外す": (old & ~new).groupby(yr).sum(),
                        "業績予想の修正": rev.groupby(yr).sum()}).astype(int)
    # 題名の型ごとの件数（会社名・期・回数などの数字を落として、同じ型をまとめる）
    shape = lambda m: titles.loc[m, "title"].str.replace(r"[0-9０-９]+", "0", regex=True).value_counts().head(n)
    sample = lambda m: [f"{t}（{c:,}件）" for t, c in shape(m).items()]
    res = dict(counts={int(y): {k: int(v) for k, v in r.items()} for y, r in tab.iterrows()},
               new_only_titles=sample(new & ~old), dropped_titles=sample(old & ~new))
    L = ["### 題名の件数（年ごと）", "", "| 年 | " + " | ".join(tab.columns) + " |", "|---|" + "---|" * len(tab.columns)]
    L += [f"| {y} | " + " | ".join(f"{int(v):,}" for v in r) + " |" for y, r in tab.iterrows()]
    L += ["", f"- 合計: 今 {int(old.sum()):,}件 → 差し替え {int(new.sum()):,}件（新しく拾う {int((new & ~old).sum()):,}・外す {int((old & ~new).sum()):,}）、修正 {int(rev.sum()):,}件",
          "- 新しく拾う題名（多い順。自社株買いの決定でないものが多ければ、正規表現を直してからやり直す）:"]
    L += [f"  - {t}" for t in res["new_only_titles"][:30]]
    L += ["- 外す題名（多い順）:"] + [f"  - {t}" for t in res["dropped_titles"][:10]]
    return res, "\n".join(L)


def net_report(cfg: dict, d: dict[str, pd.DataFrame], since: int) -> tuple[dict, str]:
    cost = cfg["research"]["cost"]
    res, L = {}, [f"### 網そのものの成績（AI なし・実際の売買ルール・コスト込み、{since}年〜）", "", lab.STATS_HEAD]

    def add(label, df, m):
        res[label] = s = lab.signal_stats(df, m, cost, since)
        L.append(lab.fmt_stats(label, s))

    b = d["base"]
    add("自社株買い発表・base", b, b["s_buyback"] == 1)
    add("（参考）決算好反応・base", b, b["s_earnings_drift"] == 1)
    if "buyback" in d:
        x = d["buyback"]
        old = set(zip(b.loc[b["s_buyback"] == 1, "date"], b.loc[b["s_buyback"] == 1, "code"]))
        is_old = pd.Series([k in old for k in zip(x["date"], x["code"])], index=x.index)
        add("自社株買い発表・buyback（全部）", x, x["s_buyback"] == 1)
        add("　うち新規だけ（base では自社株買いの候補でなかった）", x, (x["s_buyback"] == 1) & ~is_old)
        add("　うち新規で、他のどの網にも掛かっていない", x, (x["s_buyback"] == 1) & ~is_old & (x["n_strat"] == 1))
    if "rev" in d:
        x = d["rev"]
        r = x["s_rev_drift"] == 1
        add("修正好反応・rev（全部）", x, r)
        add("　うち決算好反応と重ならない", x, r & (x["s_earnings_drift"] == 0))
        add("　うち他のどの網にも掛かっていない（新しい候補）", x, r & (x["n_strat"] == 1))
        add("　初日（rev_age=0）だけ", x, r & (x["rev_age"] == 0))
    return res, "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arms", default="base,buyback,rev")
    ap.add_argument("--titles", help="全題名の Parquet（既定: data/tdnet_all.parquet）")
    ap.add_argument("--no-ai", action="store_true")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--start-year", type=int, default=2011, help="AI・口座を始める年（題名は2010年からなので既定は2011）")
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    lab.setup_logging()
    cfg = load_config()
    arms = ["base"] + [k for k in ("buyback", "rev") if k in a.arms.split(",")]  # rev は関数を差し替えるので必ず最後
    titles = load_titles(cfg, a.titles)
    lab.log.info("全題名 %d件（%s〜%s）", len(titles), titles["pubdate"].min().date(), titles["pubdate"].max().date())
    prod = tdnet.load(cfg)
    td_base = classified(titles, tdnet.classify)
    if prod is not None:
        lab.log.info("本番の tdnet.parquet %d行 / 全題名を本番の分類にかけたもの %d行（大きく違えば全題名の側が欠けている）", len(prod), len(td_base))
    out, text = {}, []
    out["titles"], t = title_report(titles)
    text.append(t)

    store, uni = store_of(cfg), lab.universe(cfg)
    cal = lab.calendar(cfg, store)
    tdnet.load = lambda _cfg: STATE["td"]               # pool.build_window が読む適時開示を差し替える
    seeds = [int(s) for s in a.seeds.split(",")]
    cfgs, outs, runs = {}, {}, {}
    for k in arms:  # 1つずつ最後まで済ませる（rev で網を足すと AI の特徴量が増えるので、base・buyback の学習はその前に終える）
        STATE["td"] = classified(titles, classify_new) if k == "buyback" else td_base
        cfgs[k] = register_rev(cfg, titles) if k == "rev" else cfg
        ds = lab.dataset(cfgs[k], lab.build_pool(cfgs[k], store, uni, f"nets_{k}", a.rebuild))
        o = lab.outcomes(cfgs[k], ds)  # 表に使う列だけ残す（3つ分を持つのでメモリを節約）
        outs[k] = o[[c for c in o.columns if c in ("date", "code", "ret_plan", "n_strat", "rev_age") or c.startswith("s_")]]
        if not a.no_ai:
            runs[k] = lab.run_arm(cfgs[k], ds, cal, f"nets_{k}", seeds, a.start_year, a.rebuild)
    out["nets"], t = net_report(cfg, outs, a.start_year)
    text.append(t)
    if not a.no_ai:
        out["ai"] = {k: [dict(seed=r["seed"], stats=r["stats"], since=r.get("since"), cand=r["cand"], yearly=r["yearly"]) for r in v]
                     for k, v in runs.items()}
        out["paired"] = {k: lab.paired(runs["base"], runs[k]) for k in arms if k != "base"}
        text.append(lab.report_ai(f"AI・口座（{a.start_year}年〜、本番と同じ設定）", runs, "base"))
        mix = ["", "- AI の上位2の内訳（種0、件数・1回の手取り）:"]
        for k in arms:
            m = runs[k][0]["cand"]["mix"]
            mix.append(f"  - {k}: " + "、".join(f"{strategies.STRATEGY_NAMES.get(n, n)} {v['n']}件 {v['net']:+.2%}" for n, v in
                                             sorted(m.items(), key=lambda kv: -kv[1]["n"])))
        text.append("\n".join(mix))
    body = "\n\n".join(text)
    p = lab.save_result(cfg, "event_nets", out, body)
    print("\n" + body + f"\n\n保存: {p}")


if __name__ == "__main__":
    main()
