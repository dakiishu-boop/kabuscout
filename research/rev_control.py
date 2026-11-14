"""検証: 「修正好反応」の対照実験（AI なし）。

問い: 修正好反応の網の手取り（2011年〜 +1.12%、§84）は「業績修正」のおかげか、それとも
「大きく上がって大商い」という形と決算好反応の売買ルールのおかげか。

対照 ctl（事前に決めた。調整しない）:
  適時開示の題名が1件も無い日（その銘柄の題名が終値に入る日＝events.known_dates が一致する日が無い）に、
  修正好反応と同じ条件（その日の騰落率が日経平均比 +5%以上・出来高が直前20日平均の1.5倍以上）を満たした銘柄を
  「架空の開示」とみなし、修正好反応と全く同じ仕組み（その日から5営業日以内・同じ売買ルール）で候補にする。
  題名は2010年からなので、架空の開示も2010年以降だけ作る。

読み方:
  対照が +0.2% 前後なら、修正の中身に意味がある。+1% 近くなら、急騰・大商いの形と売買ルールの効果にすぎない。
  修正好反応の側は「決算短信と同じ日の修正を除いた行」と比べる（対照は開示が無い日なので、決算の日は最初から入らない）。
  期間は 2011〜2018年 と 2019年〜 に分け、両方プラスかを見る。

使い方: .venv/bin/python research/rev_control.py   （シグナル表を1つ作るので約7分。rev の表は event_nets.py で作った保存済みを使う）
結果: data/research_cache/result_rev_control.md
"""
from __future__ import annotations

import argparse

import _lab as lab
import event_nets as en
import numpy as np
import pandas as pd

from kabuscout import earnings, events, tdnet
from kabuscout.config import load_config
from kabuscout.pipeline import store_of

CTL = dict(titles=None, start=pd.Timestamp("2010-01-01"))


def disclosure_mask(titles: pd.DataFrame, idx: pd.DatetimeIndex, cols: pd.Index) -> np.ndarray:
    """(日, 銘柄) で、その銘柄の何かの題名が終値に入る日なら True。"""
    T, N = len(idx), len(cols)
    m = np.zeros((T, N), dtype=bool)
    t = titles[titles["code"].isin(set(cols)) & (titles["pubdate"] >= idx[0] - pd.Timedelta(days=5))
               & (titles["pubdate"] <= idx[-1] + pd.Timedelta(days=1))]
    if t.empty:
        return m
    k = events.known_dates(t["pubdate"], idx)
    cix = {c: i for i, c in enumerate(cols)}
    j = t["code"].map(cix).to_numpy()
    ok = k < T
    m[k[ok], j[ok]] = True
    return m


def pseudo_events(close: pd.DataFrame, volume: pd.DataFrame, mkt: pd.Series, titles: pd.DataFrame,
                  start: pd.Timestamp, react_min: float = 0.05, vol_min: float = 1.5) -> pd.DataFrame:
    """開示が無い日に、修正好反応と同じ形（市場比 +5%以上・出来高1.5倍以上）になった (日, 銘柄) を「架空の開示」にする。
    rev_wide と同じ式。架空の開示は 10:00 発表扱い（その日の終値に入る）。"""
    idx, cols = close.index, close.columns
    c, v, mk = close.to_numpy("float64"), volume.to_numpy("float64"), mkt.to_numpy("float64")
    vavg = pd.DataFrame(v).rolling(20, min_periods=10).mean().shift(1).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        react = np.full_like(c, np.nan)
        react[1:] = c[1:] / c[:-1] - 1 - (mk[1:] / mk[:-1] - 1)[:, None]
        vol = np.where(vavg > 0, v / vavg, np.nan)
    spike = (react >= react_min) & (vol >= vol_min) & ~disclosure_mask(titles, idx, cols)
    spike &= (idx >= start)[:, None]
    k, j = np.nonzero(spike)
    return pd.DataFrame({"pubdate": idx[k] + pd.Timedelta(hours=10), "code": cols[j]})


def _earn_with_ctl(ev, close, volume, mkt, max_age: int = 60):
    out = en._ORIG_EARN(ev, close, volume, mkt, max_age)
    out.update(en.rev_wide(pseudo_events(close, volume, mkt, CTL["titles"], CTL["start"]), close, volume, mkt, max_age))
    return out


def event_day_pos(d: pd.DataFrame, cal: pd.DatetimeIndex) -> np.ndarray:
    """候補の行から、元の開示（架空の開示）が終値に入った日の位置（= 今日の位置 − rev_age）。"""
    return cal.searchsorted(d["date"].to_numpy()) - d["rev_age"].fillna(-1e9).to_numpy().astype(int)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--titles", help="全題名の Parquet（既定: data/tdnet_all.parquet）")
    ap.add_argument("--since", type=int, default=2011)
    ap.add_argument("--split", type=int, default=2019, help="前半・後半の区切りの年（後半の最初の年）")
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    lab.setup_logging()
    cfg = load_config()
    titles = en.load_titles(cfg, a.titles)
    CTL["titles"] = titles[["pubdate", "code"]]
    td_base = en.classified(titles, tdnet.classify)
    en.STATE["td"] = td_base
    tdnet.load = lambda _cfg: en.STATE["td"]
    cfg_rev = en.register_rev(cfg, titles)              # rev_drift の網・売買ルール・特徴量の列を足す
    store, uni = store_of(cfg), lab.universe(cfg)
    cal = lab.calendar(cfg, store)
    cost = cfg["research"]["cost"]

    # 修正好反応（保存済みの表。event_nets.py --arms base,rev で作ったもの）
    earnings.wide_features = en._earn_with_rev
    rev = lab.outcomes(cfg_rev, lab.dataset(cfg_rev, lab.build_pool(cfg_rev, store, uni, "nets_rev", False)))
    rev = rev[[c for c in rev.columns if c in ("date", "code", "ret_plan", "n_strat", "rev_age") or c.startswith("s_")]]
    # 対照（開示の無い日の急騰・大商い）
    earnings.wide_features = _earn_with_ctl
    ctl = lab.outcomes(cfg_rev, lab.dataset(cfg_rev, lab.build_pool(cfg_rev, store, uni, "nets_ctl", a.rebuild)))
    ctl = ctl[[c for c in ctl.columns if c in ("date", "code", "ret_plan", "n_strat", "rev_age") or c.startswith("s_")]]

    # 決算短信と同じ日に終値に入った修正（決算の題名がその銘柄・その日にある）
    e = td_base[td_base["events"].str.contains("earnings")]
    ek = events.known_dates(e["pubdate"], cal)
    earn_days = set(zip(e["code"], ek))
    rpos = event_day_pos(rev, cal)
    same_day = np.array([(c, int(k)) in earn_days for c, k in zip(rev["code"], rpos)])

    r, c_ = rev["s_rev_drift"] == 1, ctl["s_rev_drift"] == 1
    rows = [
        ("修正好反応（全部）", rev, r),
        ("修正好反応・決算短信と同じ日の修正を除く", rev, r & ~same_day),
        ("修正好反応・決算好反応と重ならない", rev, r & (rev["s_earnings_drift"] == 0)),
        ("修正好反応・決算短信と同じ日を除き、決算好反応とも重ならない", rev, r & ~same_day & (rev["s_earnings_drift"] == 0)),
        ("修正好反応・他のどの網にも掛かっていない", rev, r & (rev["n_strat"] == 1)),
        ("対照：開示の無い日の急騰・大商い（全部）", ctl, c_),
        ("対照・決算好反応と重ならない", ctl, c_ & (ctl["s_earnings_drift"] == 0)),
        ("対照・他のどの網にも掛かっていない", ctl, c_ & (ctl["n_strat"] == 1)),
        ("対照・初日（rev_age=0）だけ", ctl, c_ & (ctl["rev_age"] == 0)),
        ("（参考）修正好反応・初日だけ", rev, r & (rev["rev_age"] == 0)),
    ]
    out, L = {}, [f"### 修正好反応 と 対照（開示の無い日の同じ形）。AI なし・同じ売買ルール・コスト込み", ""]
    periods = [(f"{a.since}年〜", a.since, 9999), (f"{a.since}〜{a.split - 1}年", a.since, a.split - 1), (f"{a.split}年〜", a.split, 9999)]
    for name, y0, y1 in periods:
        L += [f"#### {name}", "", lab.STATS_HEAD]
        for label, d, m in rows:
            yr = d["date"].dt.year
            s = lab.signal_stats(d, m & (yr >= y0) & (yr <= y1), cost, y0)
            out.setdefault(label, {})[name] = s
            L.append(lab.fmt_stats(label, s))
        L.append("")
    n_ev = int((ctl.loc[c_, "rev_age"] == 0).sum())
    L += [f"- 決算短信と同じ日の修正は、修正好反応の候補 {int(r.sum()):,}行のうち {int((r & same_day).sum()):,}行",
          f"- 対照の候補のうち初日の行は {n_ev:,}行（架空の開示の数の目安。網の他の条件を通ったものだけ）"]
    body = "\n".join(L)
    p = lab.save_result(cfg, "rev_control", out, body)
    print("\n" + body + f"\n\n保存: {p}")


if __name__ == "__main__":
    main()
