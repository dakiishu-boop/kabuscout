"""資金100万円で実際に売買することを見越したリスク管理（ペーパートレードと過去ペーパートレードで共通）。

- 株数は100株単位（単元）。買えない値段の銘柄は飛ばして次の候補へ
- 1銘柄に入れる金額は「資金 ÷ 同時保有数」まで
- さらに、損切になったときの損失が資金の2%を超えないように株数を減らす（値動きの荒い銘柄・損切幅の広い戦略で効く）。
  窓を開けて損切ラインを飛び越えると、これより大きく負けることはある
- 信用取引は使わない（現物のみ）。手数料は主要ネット証券の国内株現物の無料プランを想定し、気配差・滑りを research.cost で差し引く

検証（ジャーナル ラウンド8）: 「損切時の損失を資金の1%に揃える」方式は、損切幅の広い決算好反応などを小さく買うことになり年率が下がった。
資金が減ったらリスクを段階的に減らす方式も年率を大きく下げ、最大下落は縮まなかったので採用していない。
"""
from __future__ import annotations


def order_shares(acc: dict, equity: float, cash: float, price: float, stop_dist: float, slot: float) -> int:
    """買う株数（単元の倍数）。0 なら見送り。slot = 1銘柄に入れる上限額、stop_dist = 損切までの値幅（円）。"""
    lot = acc["lot_size"]
    if not (price > 0):
        return 0
    n = min(slot, cash) / price
    if acc.get("risk_per_trade") and stop_dist > 0:
        n = min(n, equity * acc["risk_per_trade"] / stop_dist)
    return int(n // lot) * lot


def tick_size(price):
    """東証の呼値の単位（標準の表。TOPIX100 の銘柄はもっと細かいが、ここでは保守的に標準の表を使う）。配列でも可。"""
    import numpy as np
    p = np.asarray(price, dtype="float64")
    return np.select([p <= 3000, p <= 5000, p <= 30000, p <= 50000, p <= 300000],
                     [1.0, 5.0, 10.0, 50.0, 100.0], default=500.0)
