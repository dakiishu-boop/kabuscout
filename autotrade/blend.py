"""AIと指数を組み合わせた口座（指数の部分 = 「スリーブ」）。過去の検証（sim.run）と毎日の運用（live）で同じ計算を使う。

- 口座の資金のうち share を指数（S&P500 か 全世界株式）、残りを AI で運用する。指数は「口数 × 指数の値段（円）」で持つ
  （データの取得が遅れても、届いた日に正しい値に追いつく）
- 毎年の最初の営業日（寄付の前・前日の終値の時点）に、次の順で処理する:
  1. 前の年の AI の部分の利益に税（tax、例 0.20315）をかける。損失は3年まで繰り越す。現金で払い、足りなければ指数を売って払う
  2. 指数の比率を share に戻す。**現金だけで動かす**（指数を増やすときに AI の保有は売らない。足りない分はその年は戻しきらない）
- 指数の売買のコストは0とみなす（ノーロードの投資信託を想定）。信託報酬は値段の側で年0.1%を引く
- 指数は NISA（非課税）、AI は特定口座（課税）を想定。税をかけない場合は tax=0
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

COLS = {"S&P500": "spx_tr", "全世界株式（オルカン）": "acwi"}
FEE_PER_YEAR = 0.001


def index_levels(ix: pd.DataFrame, cal: pd.DatetimeIndex, name: str) -> tuple[np.ndarray, np.ndarray]:
    """日本の各営業日の朝の時点で分かっている指数の値段（円、配当込み、信託報酬を引いたもの）と、その元になった米国の日付。
    kabuscout.core.index_returns_jpy と同じく、米国の終値は日本の翌営業日に反映する。"""
    us = ix.ffill()
    v = (us[COLS[name]] * us["usdjpy"]).dropna()
    pos = v.index.searchsorted(cal, side="left") - 1
    ok = pos >= 0
    lvl = np.where(ok, v.to_numpy()[np.clip(pos, 0, None)], np.nan)
    src = np.where(ok, v.index.to_numpy()[np.clip(pos, 0, None)], np.datetime64("NaT"))
    fee = (1 - FEE_PER_YEAR / 250) ** np.arange(len(cal))
    return lvl * fee, src


@dataclass
class Sleeve:
    share: float
    tax: float = 0.0
    units: float = 0.0
    year: int | None = None          # 税と比率の戻しを済ませた年（同じ年に2回しない）
    ai_start: float = 0.0            # その年の初め（戻した後）の AI の部分の資産
    carry: list = field(default_factory=list)  # [[年, 繰り越した損失], ...]
    taxes_paid: float = 0.0
    events: list = field(default_factory=list)

    def value(self, level: float) -> float:
        return self.units * level if self.units and level == level else 0.0

    def _tax_due(self, gain: float, year: int, commit: bool) -> float:
        carry = [[y, a] for y, a in self.carry if year - y <= 3]
        if gain < 0:
            if commit:
                self.carry = carry + [[year, -gain]]
            return 0.0
        g = gain
        for c in carry:
            use = min(g, c[1])
            g -= use
            c[1] -= use
        if commit:
            self.carry = [c for c in carry if c[1] > 1e-9]
        return g * self.tax

    def pending_tax(self, ai_equity: float, year: int) -> float:
        """まだ払っていない今年の分の税（過去の検証の最後の日の評価に使う）。"""
        if not self.tax or self.year is None:
            return 0.0
        return self._tax_due(ai_equity - self.ai_start, year, commit=False)

    def new_year(self, year: int, broker, ai_equity: float, level: float, date: str = "") -> bool:
        """その年の最初の営業日の寄付の前に呼ぶ（ai_equity = 前日の終値での AI の部分の資産）。済んでいれば何もしない。"""
        if self.year == year or not (level == level and level > 0):
            return False
        if self.year is not None and self.tax:
            t = self._tax_due(ai_equity - self.ai_start, self.year, commit=True)
            if t > 0:
                pay = min(t, broker.cash())
                broker._cash -= pay
                rest = t - pay
                if rest > 0:  # 現金が足りなければ指数を売って払う
                    self.units -= min(self.units, rest / level)
                self.taxes_paid += t
                ai_equity -= pay
                self.events.append(f"{date} {self.year}年の AI の利益に税 {t:,.0f}円")
        total = ai_equity + self.value(level)
        diff = self.share * total - self.value(level)
        if diff > 0:
            move = min(diff, broker.cash())
            broker._cash -= move
            self.units += move / level
        elif diff < 0:
            broker._cash += -diff
            self.units -= -diff / level
            move = diff
        else:
            move = 0.0
        if abs(diff) > 1:
            self.events.append(f"{date} 指数の比率を戻す: {'+' if diff > 0 else ''}{(move if diff > 0 else diff):,.0f}円")
        self.ai_start = ai_equity - (move if diff > 0 else diff)
        self.year = year
        return True

    def to_dict(self) -> dict:
        d = asdict(self)
        d["events"] = d["events"][-50:]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Sleeve":
        return cls(**d)
