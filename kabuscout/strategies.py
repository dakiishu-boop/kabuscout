"""シグナル検出ルール。

ここは「候補を拾う網」。どの網が今の相場で本当に効いているかは learner.py が
過去の結果から学習し、スコアと重みを自動で調整する。新しいアイデアは STRATEGIES に
関数を1つ足すだけで、次の再学習から自動的に評価対象になる。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd


@dataclass
class Strategy:
    key: str
    name: str
    desc: str
    detect: Callable[[dict], pd.DataFrame]


def _between(df, lo, hi):
    return (df >= lo) & (df <= hi)


def _reversal(i):
    """反転の兆し: 陽線で前日比プラス、または長い下ヒゲ。"""
    return ((i["close"] > i["open"]) & (i["ret1"] > 0)) | (i["lower_shadow"] >= 0.5)


def pullback_25(i):
    uptrend = (i["slope75"] > 0) & (i["close"] > i["sma75"]) & (i["sma25"] > i["sma75"])
    had_run = i["high20"] >= i["sma25"] * 1.05
    near_ma = _between(i["dev25"], -0.03, 0.015)
    depth = _between(i["pullback20"], -0.15, -0.04)
    quiet = i["vol5_20"] < 1.0
    return uptrend & had_run & near_ma & depth & quiet & _reversal(i) & _between(i["rsi14"], 35, 58)


def pullback_75(i):
    long_up = (i["slope200"] > 0) & (i["close"] > i["sma200"])
    touch = (i["low"] <= i["sma75"] * 1.01) & (i["close"] >= i["sma75"] * 0.98)
    had_run = i["high60"] >= i["sma75"] * 1.10
    return long_up & touch & had_run & _reversal(i)


def breakout_52w(i):
    return (i["close"] > i["prior_high52"]) & (i["vol_ratio"] >= 1.8) & (i["close"] > i["open"])


def box_breakout(i):
    c = i["close"]
    hi, lo = c.shift(1).rolling(60).max(), c.shift(1).rolling(60).min()
    return (hi / lo < 1.20) & (c > hi) & (i["vol_ratio"] >= 1.5)


def volume_ignition(i):
    # 長く放置された銘柄に突然大商い＝初動。高値圏の銘柄は除外（見落とされがちな方を拾う）
    return ((i["vol_ratio"] >= 3) & (i["ret1"] >= 0.02) & (i["close_pos"] >= 0.6)
            & (i["dist_high52"] <= -0.15) & (i["dev25"] < 0.10))


def golden_cross(i):
    s5, s25 = i["sma5"], i["sma25"]
    return (s5 > s25) & (s5.shift(1) <= s25.shift(1)) & (i["slope75"] > 0)


def oversold_rebound(i):
    washed = (i["rsi14"].rolling(3).min() < 30) | (i["low"] < i["bb_lower"])
    confirm = (i["close"] > i["open"]) & (i["close"] > i["high"].shift(1))
    return washed & confirm & (i["dev200"] > -0.25)


def squeeze(i):
    tight = i["bbw_rank"] <= 0.05
    return tight & (i["slope25"] >= 0) & (i["close"] > i["sma25"]) & (i["slope75"] > 0)


def rs_dip(i):
    # 市場全体が下げる中で踏みとどまっている強い銘柄の押し目
    return ((i["mkt_ret5"] <= -0.02) & (i["rs60"] >= 0.10) & (i["close"] > i["sma75"])
            & _between(i["dev25"], -0.03, 0.03))


STRATEGIES: list[Strategy] = [
    Strategy("pullback_25", "押し目(25日線)", "上昇トレンド中、出来高を細らせながら25日線まで調整し反転の兆し", pullback_25),
    Strategy("pullback_75", "深押し(75日線)", "長期上昇トレンドの銘柄が75日線まで深く押して反発", pullback_75),
    Strategy("breakout_52w", "52週高値ブレイク", "出来高を伴って1年来高値を更新", breakout_52w),
    Strategy("box_breakout", "ボックス上抜け", "3ヶ月の保ち合いを出来高増で上放れ", box_breakout),
    Strategy("volume_ignition", "出来高急増(初動)", "高値から離れた放置銘柄に3倍超の出来高で陽線", volume_ignition),
    Strategy("golden_cross", "ゴールデンクロス", "5日線が25日線を上抜け（75日線は上向き）", golden_cross),
    Strategy("oversold_rebound", "売られすぎ反発", "RSI30割れ/BB-2σ割れの後、前日高値を超える陽線", oversold_rebound),
    Strategy("squeeze", "スクイーズ", "ボラティリティが半年で最小まで収縮、上放れ待ち", squeeze),
    Strategy("rs_dip", "地合い逆行の押し目", "市場が下げる中でも強い銘柄が25日線付近で踏みとどまる", rs_dip),
]
STRATEGY_NAMES = {s.key: s.name for s in STRATEGIES}
FLAG_COLS = [f"s_{s.key}" for s in STRATEGIES]


def detect_all(ind: dict, universe_mask: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """戦略ごとのブール行列を返す（流動性フィルタ適用済み）。"""
    return {s.key: (s.detect(ind).fillna(False).astype(bool) & universe_mask) for s in STRATEGIES}


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
        parts.append(f"対TOPIX 60日{row['rs60']:+.0%}")
    if "squeeze" in keys:
        parts.append("BB幅が半年最小")
    if row.get("dist_high52", -1) > -0.03 and "breakout_52w" not in keys:
        parts.append("52週高値圏")
    return "・".join(parts)
