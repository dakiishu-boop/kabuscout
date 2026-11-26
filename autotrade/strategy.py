"""売買の判断（バックテストと毎日の運用で同じ関数を使う）。

毎日の引け後に呼ばれ、翌営業日の寄付で出す注文を返す。
- 入れ替え日（既定は5営業日ごと）に、AIの点数の上位 n_hold 銘柄を持つ
- すでに持っている銘柄は、順位が keep_rank 位以内なら持ち続ける（入れ替えすぎてコストで負けないため）
- 1銘柄に入れるのは資金 ÷ n_hold まで。100株で買えない銘柄は飛ばして次の候補へ（ただし depth 位まで）
- 任意: 損切（買値から stop_loss 下落した終値で翌朝売る）、地合いが弱いときに持つ数を減らす（regime）
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from .broker import Order, PaperBroker


@dataclass
class Params:
    n_hold: int = 8
    rebalance_every: int = 5
    keep_rank: int = 40          # 持っている銘柄はこの順位以内なら持ち続ける
    depth: int = 40              # 新規に買う候補を何位まで見るか
    stop_loss: float = 0.0       # 0 なら使わない（例 0.1 = 買値から −10% の終値で翌朝売る）
    regime_ma: int = 0           # 0 なら使わない。日経平均が regime_ma 日線を下回ったら持つ数を regime_scale 倍に
    regime_scale: float = 1.0
    lot: int = 100
    cost: float = 0.002          # 往復の取引コスト（kabuscout の research.cost と同じ既定値）
    max_adv_frac: float = 0.0    # 1注文を「20日平均売買代金 × この割合」までにする（0 なら上限なし。自分の注文で値段を動かさないため）

    def to_dict(self) -> dict:
        return asdict(self)


def target_count(p: Params, weak: bool) -> int:
    return max(0, int(round(p.n_hold * (p.regime_scale if weak else 1.0))))


def decide(broker: PaperBroker, scores: dict[str, float], eligible: set[str], close_px: dict[str, float],
           rebalance: bool, weak: bool, p: Params, turnover=None) -> list[Order]:
    """scores: 点数（高いほど良い。保有中の銘柄は対象外になっていても点数があれば入れる）。
    eligible: 新規に買ってよい銘柄。close_px: 今日の終値（実際の株価）。"""
    pos = broker.positions()
    half = p.cost / 2
    ranked = sorted((c for c in scores if c in eligible or c in pos), key=lambda c: -scores[c])
    rank = {c: i for i, c in enumerate(ranked)}
    orders: list[Order] = []
    selling: set[str] = set()
    for code, ps in pos.items():
        c = close_px.get(code)
        if not (c and math.isfinite(c)):
            continue  # 値段が無い日は判断しない
        why = None
        if p.stop_loss and c <= ps.entry_px * (1 - p.stop_loss):
            why = "損切"
        elif rebalance and rank.get(code, 10 ** 9) >= p.keep_rank:
            why = "順位低下"
        if why:
            orders.append(Order(code, "sell", ps.shares, c, why))
            selling.add(code)
    if not rebalance:
        return orders
    n_target = target_count(p, weak)
    if weak and p.regime_scale < 1:  # 弱い相場で持つ数を減らすときは、順位の低い保有から売る
        keep = sorted((c for c in pos if c not in selling), key=lambda c: rank.get(c, 10 ** 9))
        for code in keep[n_target:]:
            orders.append(Order(code, "sell", pos[code].shares, close_px.get(code) or pos[code].entry_px, "地合い"))
            selling.add(code)
    held = [c for c in pos if c not in selling]
    slots = n_target - len(held)
    if slots <= 0:
        return orders
    equity = broker.equity(close_px)
    avail = broker.cash() + sum(pos[c].shares * (close_px.get(c) or pos[c].entry_px) * (1 - half) for c in selling)
    slot_value = equity / p.n_hold
    for code in ranked[:p.depth]:
        if slots <= 0:
            break
        if code in pos or code not in eligible:
            continue
        c = close_px.get(code)
        if not (c and c > 0 and math.isfinite(c)):
            continue
        cap = slot_value
        if p.max_adv_frac and turnover is not None:
            t = turnover.get(code)
            cap = min(cap, (t or 0.0) * p.max_adv_frac)
        n = int(min(cap, avail) / (c * (1 + half)) // p.lot) * p.lot
        if n <= 0:
            continue  # 100株で買えない値段・現金不足
        orders.append(Order(code, "buy", n, c, f"順位{rank[code] + 1}"))
        avail -= n * c * (1 + half)
        slots -= 1
    return orders
