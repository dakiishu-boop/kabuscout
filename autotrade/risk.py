"""注文を出す直前の安全装置（毎日の運用・将来の実口座で使う。バックテストでは既定で使わない）。

人が見ていなくても事故にならないように、次の注文は出さない:
- データが古い（最新の株価が data_max_age_days 営業日より前）→ 全部止める（古い値段で判断しないため）
- 1注文の金額が max_order_value を超える、資金に対する割合が max_order_frac を超える
- 判断に使った値段が前日の終値から max_price_move 以上離れている（データの誤り・分割の調整漏れの疑い）
- 1日の注文数が max_orders_per_day を超える
- 口座が最高値から kill_drawdown 以上下がった → 新規の買いを止める（売りは通す）。止めたら人が解除するまで止まったまま
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .broker import Order


@dataclass
class RiskLimits:
    max_order_value: float = 400_000
    max_order_frac: float = 0.35
    max_price_move: float = 0.3
    max_orders_per_day: int = 20
    kill_drawdown: float = 0.35
    data_max_age_days: int = 3


@dataclass
class RiskGuard:
    limits: RiskLimits = field(default_factory=RiskLimits)
    halted: bool = False
    halt_reason: str = ""

    def check(self, orders: list[Order], equity: float, peak: float, last_close: dict[str, float],
              data_age_days: int) -> tuple[list[Order], dict[str, str]]:
        """通してよい注文と、止めた注文の理由 {銘柄: 理由} を返す。"""
        L = self.limits
        why: dict[str, str] = {}
        if data_age_days > L.data_max_age_days:
            return [], {o.code: f"データが古い（{data_age_days}営業日前）" for o in orders} or {"*": "データが古い"}
        if peak > 0 and equity < peak * (1 - L.kill_drawdown) and not self.halted:
            self.halted = True
            self.halt_reason = f"最高値から {equity / peak - 1:.1%}（上限 −{L.kill_drawdown:.0%}）"
        ok: list[Order] = []
        for o in orders:
            if o.side == "buy" and self.halted:
                why[o.code] = f"停止中: {self.halt_reason}"
                continue
            ref = last_close.get(o.code)
            if not (o.ref_px and o.ref_px > 0 and math.isfinite(o.ref_px)):
                why[o.code] = "値段が不明"
                continue
            if ref and ref > 0 and abs(o.ref_px / ref - 1) > L.max_price_move:
                why[o.code] = f"値段が前日の終値から離れすぎ（{o.ref_px:.0f} vs {ref:.0f}）"
                continue
            value = o.shares * o.ref_px
            if o.side == "buy" and (value > L.max_order_value or value > equity * L.max_order_frac):
                why[o.code] = f"1注文の金額が上限超え（{value:,.0f}円）"
                continue
            if len(ok) >= L.max_orders_per_day:
                why[o.code] = "1日の注文数の上限"
                continue
            ok.append(o)
        return ok, why

    def to_dict(self) -> dict:
        return dict(halted=self.halted, halt_reason=self.halt_reason)
