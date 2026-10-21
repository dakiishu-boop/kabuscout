"""シグナル検出ルール。

ここは「候補を拾う網」。各ルールの閾値（押し目の深さ・出来高の倍率など）はパラメータになっていて、
research.py の研究ループが過去データで調整する。どの網が今の相場で効いているかは learner.py が学習する。

判定式は要素ごとの比較だけで書く（シフトやローリングは indicators.py で事前計算）。
こうすると「日付×銘柄」の行列にも、シグナル表（1行=1銘柄日）にも同じ式がそのまま使える。
新しいルールは STRATEGIES に足すだけで、次の再学習・研究から自動的に評価対象になる。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd


@dataclass
class Strategy:
    key: str
    name: str
    desc: str
    detect: Callable[[dict, dict], pd.DataFrame]
    # パラメータ名 → (探索の下限, 上限, 初期値, 最も条件が緩くなる値)
    space: dict = field(default_factory=dict)

    def defaults(self) -> dict:
        return {k: v[2] for k, v in self.space.items()}

    def loose(self) -> dict:
        return {k: v[3] for k, v in self.space.items()}


def _between(x, lo, hi):
    return (x >= lo) & (x <= hi)


def _reversal(i):
    """反転の兆し: 陽線で前日比プラス、または長い下ヒゲ。"""
    return ((i["close"] > i["open"]) & (i["ret1"] > 0)) | (i["lower_shadow"] >= 0.5)


def pullback_25(i, p):
    uptrend = (i["slope75"] > 0) & (i["close"] > i["sma75"]) & (i["sma25"] > i["sma75"])
    had_run = i["high20"] >= i["sma25"] * (1 + p["run"])
    near_ma = _between(i["dev25"], p["dev_lo"], p["dev_hi"])
    depth = _between(i["pullback20"], p["depth_lo"], p["depth_hi"])
    quiet = i["vol5_20"] < p["vol_dry"]
    return uptrend & had_run & near_ma & depth & quiet & _reversal(i) & _between(i["rsi14"], p["rsi_lo"], p["rsi_hi"])


def pullback_75(i, p):
    long_up = (i["slope200"] > 0) & (i["close"] > i["sma200"])
    touch = (i["low"] <= i["sma75"] * (1 + p["touch"])) & (i["close"] >= i["sma75"] * (1 - p["below"]))
    had_run = i["high60"] >= i["sma75"] * (1 + p["run"])
    return long_up & touch & had_run & _reversal(i)


def breakout_52w(i, p):
    return (i["close"] > i["prior_high52"]) & (i["vol_ratio"] >= p["vol"]) & (i["close"] > i["open"])


def box_breakout(i, p):
    return ((i["box_hi60"] < i["box_lo60"] * (1 + p["width"])) & (i["close"] > i["box_hi60"])
            & (i["vol_ratio"] >= p["vol"]))


def volume_ignition(i, p):
    # 長く放置された銘柄に突然大商い＝初動。高値圏の銘柄は除外（見落とされがちな方を拾う）
    return ((i["vol_ratio"] >= p["vol"]) & (i["ret1"] >= p["ret"]) & (i["close_pos"] >= p["cpos"])
            & (i["dist_high52"] <= p["far"]) & (i["dev25"] < p["dev_max"]))


def golden_cross(i, p):
    return (i["sma5"] > i["sma25"]) & (i["sma5_prev"] <= i["sma25_prev"]) & (i["slope75"] > p["slope"])


def oversold_rebound(i, p):
    washed = (i["rsi14_min3"] < p["rsi"]) | (i["low"] < i["bb_lower"])
    confirm = (i["close"] > i["open"]) & (i["close"] > i["prev_high"])
    return washed & confirm & (i["dev200"] > p["floor"])


def squeeze(i, p):
    return (i["bbw_rank"] <= p["rank"]) & (i["slope25"] >= 0) & (i["close"] > i["sma25"]) & (i["slope75"] > 0)


def rs_dip(i, p):
    # 市場全体が下げる中で踏みとどまっている強い銘柄の押し目
    return ((i["mkt_ret5"] <= p["mkt"]) & (i["rs60"] >= p["rs"]) & (i["close"] > i["sma75"])
            & _between(i["dev25"], -p["band"], p["band"]))


STRATEGIES: list[Strategy] = [
    Strategy("pullback_25", "押し目(25日線)", "上昇トレンド中、出来高を細らせながら25日線まで調整し反転の兆し", pullback_25, dict(
        run=(0.02, 0.10, 0.05, 0.02), dev_lo=(-0.06, -0.01, -0.03, -0.06), dev_hi=(0.0, 0.04, 0.015, 0.04),
        depth_lo=(-0.25, -0.08, -0.15, -0.25), depth_hi=(-0.08, -0.02, -0.04, -0.02), vol_dry=(0.7, 1.3, 1.0, 1.3),
        rsi_lo=(25, 45, 35, 25), rsi_hi=(50, 65, 58, 65))),
    Strategy("pullback_75", "深押し(75日線)", "長期上昇トレンドの銘柄が75日線まで深く押して反発", pullback_75, dict(
        touch=(0.0, 0.03, 0.01, 0.03), below=(0.0, 0.05, 0.02, 0.05), run=(0.05, 0.20, 0.10, 0.05))),
    Strategy("breakout_52w", "52週高値ブレイク", "出来高を伴って1年来高値を更新", breakout_52w, dict(
        vol=(1.2, 3.0, 1.8, 1.2))),
    Strategy("box_breakout", "ボックス上抜け", "3ヶ月の保ち合いを出来高増で上放れ", box_breakout, dict(
        width=(0.10, 0.30, 0.20, 0.30), vol=(1.2, 2.5, 1.5, 1.2))),
    Strategy("volume_ignition", "出来高急増(初動)", "高値から離れた放置銘柄に大商いで陽線", volume_ignition, dict(
        vol=(2.0, 5.0, 3.0, 2.0), ret=(0.01, 0.05, 0.02, 0.01), cpos=(0.4, 0.8, 0.6, 0.4),
        far=(-0.30, -0.05, -0.15, -0.05), dev_max=(0.05, 0.20, 0.10, 0.20))),
    Strategy("golden_cross", "ゴールデンクロス", "5日線が25日線を上抜け（75日線は上向き）", golden_cross, dict(
        slope=(-0.01, 0.02, 0.0, -0.01))),
    Strategy("oversold_rebound", "売られすぎ反発", "RSI30割れ/BB-2σ割れの後、前日高値を超える陽線", oversold_rebound, dict(
        rsi=(20, 40, 30, 40), floor=(-0.40, -0.10, -0.25, -0.40))),
    Strategy("squeeze", "スクイーズ", "ボラティリティが半年で最小近くまで収縮、上放れ待ち", squeeze, dict(
        rank=(0.02, 0.15, 0.05, 0.15))),
    Strategy("rs_dip", "地合い逆行の押し目", "市場が下げる中でも強い銘柄が25日線付近で踏みとどまる", rs_dip, dict(
        mkt=(-0.04, -0.01, -0.02, -0.01), rs=(0.05, 0.20, 0.10, 0.05), band=(0.01, 0.05, 0.03, 0.05))),
]
BY_KEY = {s.key: s for s in STRATEGIES}
STRATEGY_NAMES = {s.key: s.name for s in STRATEGIES}
FLAG_COLS = [f"s_{s.key}" for s in STRATEGIES]


# ---------------------------------------------------------------- パラメータ（研究ループが更新）
def load_params(path: Path) -> dict[str, dict]:
    saved = json.loads(path.read_text()).get("params", {}) if path.exists() else {}
    return {s.key: {**s.defaults(), **saved.get(s.key, {})} for s in STRATEGIES}


def loose_params() -> dict[str, dict]:
    return {s.key: s.loose() for s in STRATEGIES}


def detect_all(ind, mask, params: dict[str, dict]) -> dict[str, pd.DataFrame]:
    """戦略ごとのブール判定（行列でもシグナル表でも可）。mask は流動性フィルタ。"""
    out = {}
    for s in STRATEGIES:
        r = s.detect(ind, params[s.key]).fillna(False).astype(bool)
        out[s.key] = r & mask if mask is not None else r
    return out


def apply_flags(df: pd.DataFrame, params: dict[str, dict]) -> pd.DataFrame:
    """シグナル表に、現在のパラメータでの戦略フラグ s_* を付け、どれかに該当する行だけ返す。"""
    flags = detect_all(df, None, params)
    df = df.copy()
    for k, v in flags.items():
        df[f"s_{k}"] = v.astype("int8")
    df["n_strat"] = df[FLAG_COLS].sum(axis=1)
    df = df[df["n_strat"] > 0].copy()
    keys = [s.key for s in STRATEGIES]
    arr = df[FLAG_COLS].to_numpy().astype(bool)
    df["strategies"] = [",".join(k for k, f in zip(keys, row) if f) for row in arr]
    return df


def reasons(row: pd.Series, keys: list[str]) -> str:
    """人が読むための根拠テキスト。"""
    parts = []
    if any(k.startswith("pullback") or k == "rs_dip" for k in keys):
        parts.append(f"25日線乖離{row['dev25']:+.1%}")
        parts.append(f"20日高値から{row['pullback20']:+.1%}")
    if "pullback_75" in keys:
        parts.append(f"75日線乖離{row['dev75']:+.1%}")
    if row.get("vol_ratio", 1) >= 1.5:
        parts.append(f"出来高{row['vol_ratio']:.1f}倍")
    elif row.get("vol5_20", 1) < 0.85:
        parts.append(f"出来高細り{row['vol5_20']:.2f}倍")
    if row.get("lower_shadow", 0) >= 0.5:
        parts.append("下ヒゲ")
    elif row.get("body", 0) > 0.5:
        parts.append("陽線")
    parts.append(f"RSI{row['rsi14']:.0f}")
    if "rs_dip" in keys or row.get("rs60", 0) >= 0.15:
        parts.append(f"対日経平均 60日{row['rs60']:+.0%}")
    if "squeeze" in keys:
        parts.append("BB幅が半年で最小圏")
    if row.get("dist_high52", -1) > -0.03 and "breakout_52w" not in keys:
        parts.append("52週高値圏")
    return "・".join(parts)
