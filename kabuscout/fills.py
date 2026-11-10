"""実際の約定の記録と、想定した値段とのずれ（約定のずれ）の集計（ジャーナル 78）。

過去の検証は「往復0.2%のコスト＋損切は1ティック不利」を仮定している。往復コストが0.1%違うと年率が約4ポイント動く（ジャーナル 77）ので、
実際に売買するようになったら、注文ごとに約定した値段を記録して、仮定が合っているかを確かめる。

- 記録は data/real_fills.csv（手元だけ。個人の売買の記録なので公開のリポジトリには入れない）
- 想定の値段（比べる相手）は注文の種類で決まる:
    open  = 寄付の成行（想定はその日の始値）／ close = 引けの成行（想定はその日の終値）
    limit = 指値（想定は指値。利確など）／ stop = 逆指値（想定は逆指値の値段。損切など）
  open・close はこの仕組みの株価（Yahoo）から、limit・stop は記録するときに入れた値段（--ref）を使う
- ずれは「不利な方向をプラス」で、% とティック数の両方で出す（ティックは標準の呼値の表。大型株は実際の呼値がもっと細かいので、% の方を見る）

使い方:
  python -m kabuscout fill --side buy --code 7203 --shares 100 --price 2510 --kind open --date 2026-11-10
  python -m kabuscout fill --side sell --code 7203 --shares 100 --price 2440 --kind stop --ref 2442
  python -m kabuscout fills          # 集計
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

from .risk import tick_size

COLS = ["date", "code", "side", "kind", "shares", "price", "ref", "note"]
KINDS = ("open", "close", "limit", "stop")


def path(cfg: dict) -> Path:
    return cfg["data"]["db_path"].parent / "real_fills.csv"


def add(cfg: dict, date: str, code: str, side: str, kind: str, shares: int, price: float,
        ref: float | None = None, note: str = "") -> dict:
    if side not in ("buy", "sell"):
        raise ValueError("side は buy か sell")
    if kind not in KINDS:
        raise ValueError(f"kind は {', '.join(KINDS)} のどれか")
    if kind in ("limit", "stop") and not ref:
        raise ValueError("指値・逆指値は、注文した値段を --ref で入れてください")
    if not (shares > 0 and price > 0):
        raise ValueError("株数と値段は正の数")
    row = dict(date=str(pd.Timestamp(date).date()), code=str(code), side=side, kind=kind, shares=int(shares),
               price=float(price), ref=float(ref) if ref else "", note=note)
    p = path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    new = not p.exists()
    with p.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        if new:
            w.writeheader()
        w.writerow(row)
    return row


def load(cfg: dict) -> pd.DataFrame:
    p = path(cfg)
    if not p.exists():
        return pd.DataFrame(columns=COLS)
    d = pd.read_csv(p, dtype={"code": str})
    d["date"] = pd.to_datetime(d["date"])
    return d


def with_reference(fills: pd.DataFrame, px: dict[str, pd.DataFrame] | None) -> pd.DataFrame:
    """各約定に想定の値段（ref_px）・不利な方向のずれ（slip、割合）・ティック数（ticks）・その日までの20日平均の売買代金を付ける。
    px: {"open"|"close"|"volume": 日付×銘柄}（分割の調整は約定の日の値段に戻してある前提）"""
    d = fills.copy()
    ref_px, turn = [], []
    for r in d.itertuples():
        v = np.nan
        if r.kind in ("limit", "stop"):
            v = float(r.ref) if pd.notna(r.ref) and r.ref != "" else np.nan
        elif px is not None and r.kind in px and r.code in px[r.kind].columns and r.date in px[r.kind].index:
            v = float(px[r.kind].at[r.date, r.code])
        ref_px.append(v)
        t = np.nan
        if px is not None and "volume" in px and r.code in px["volume"].columns:
            c, vol = px["close"][r.code], px["volume"][r.code]
            t = float((c * vol).loc[:r.date].tail(20).mean())
        turn.append(t)
    d["ref_px"] = ref_px
    d["turnover20"] = turn
    sign = np.where(d["side"] == "buy", 1.0, -1.0)
    d["slip"] = sign * (d["price"] - d["ref_px"]) / d["ref_px"]
    d["ticks"] = sign * (d["price"] - d["ref_px"]) / tick_size(d["ref_px"].fillna(1000.0))
    return d


def summary(d: pd.DataFrame, assumed_cost: float) -> dict:
    """種類ごとの平均のずれと、往復コストの実測の目安（買いのずれの平均＋売りのずれの平均。損切は仮定の1ティックとも比べる）。"""
    ok = d.dropna(subset=["slip"])
    out = dict(n=int(len(d)), measured=int(len(ok)), by_kind={})
    for (side, kind), g in ok.groupby(["side", "kind"]):
        out["by_kind"][f"{side}/{kind}"] = dict(n=int(len(g)), slip=float(g["slip"].mean()) + 0.0, ticks=float(g["ticks"].mean()) + 0.0,
                                               worst_ticks=float(g["ticks"].max()) + 0.0)  # + 0.0 は「-0」と表示されないように
    b, s = ok[ok["side"] == "buy"], ok[ok["side"] == "sell"]
    if len(b) and len(s):
        # 損切は検証で1ティック不利を別に入れているので、その分を除いた残りを往復コストと比べる
        s_extra = s["slip"] - np.where(s["kind"] == "stop", tick_size(s["ref_px"]) / s["ref_px"], 0.0)
        out["round_trip"] = float(b["slip"].mean() + s_extra.mean())
        out["assumed"] = assumed_cost
    return out


def report(cfg: dict) -> str:
    from .data import PriceStore
    f = load(cfg)
    if f.empty:
        return f"記録がありません（{path(cfg)}）。python -m kabuscout fill ... で約定を記録してください"
    st = PriceStore(cfg["data"]["price_dir"])
    codes = sorted(f["code"].unique())
    raw = st.load(start=str((f["date"].min() - pd.Timedelta(days=45)).date()), fields=["open", "close", "volume"], codes=codes)
    try:  # その後の株式分割で調整された値段を、約定の日の実際の値段に戻す
        fac = st.split_factor(raw["close"].index, raw["close"].columns)
        raw = {k: (v * fac if k in ("open", "close") else v / fac) for k, v in raw.items()}
    except Exception:
        pass
    d = with_reference(f, raw)
    s = summary(d, cfg["research"]["cost"])
    lines = [f"約定の記録 {s['n']}件（想定の値段と比べられたもの {s['measured']}件）。ずれは不利な方向がプラス"]
    for k, v in s["by_kind"].items():
        lines.append(f"  {k:12s} {v['n']:3d}件  平均 {v['slip']:+.3%}（{v['ticks']:+.1f}ティック、最悪 {v['worst_ticks']:+.0f}）")
    miss = d[d["slip"].isna() & d["kind"].isin(["open", "close"])]
    if len(miss):  # 寄付・引けの約定は手元の株価と比べるので、株価がその日まで無いと比べられない
        last = raw["close"].index.max() if len(raw["close"]) else None
        lines.append(f"  ※ {len(miss)}件は手元の株価に約定の日が無く、比べられません（手元の株価は "
                     f"{last:%Y/%m/%d} まで）。python -m kabuscout prices で株価を更新してから、もう一度実行してください" if last is not None else
                     f"  ※ {len(miss)}件は手元に株価が無く、比べられません。python -m kabuscout prices で株価を取得してください")
    if "round_trip" in s:
        lines.append(f"往復コストの実測の目安 {s['round_trip']:+.3%}（検証の仮定 {s['assumed']:.2%}・損切は別に1ティック）")
        lines.append("  → 仮定より大きければ、過去の検証のコスト別の成績（Webページの「比べる基準の期間とコストの見積もり」）の近い行を目安にする")
    return "\n".join(lines)
