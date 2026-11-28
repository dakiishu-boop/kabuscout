"""証券会社の共通の形。株数は「実際の株数」（分割前に戻した値段で数える）。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass


class LiveTradingRefused(RuntimeError):
    """実際のお金での発注の条件がそろっていない。"""


@dataclass
class Order:
    symbol: str
    side: str                 # "buy" / "sell"
    qty: float                # 株数（買いは整数。売りは保有の全部なら端数もあり）
    est_price: float          # 見積りに使った値段（判定日の終値）
    type: str = "market"
    tif: str = "opg"          # opg = 寄付の成行（Alpaca の time_in_force）
    note: str = ""

    @property
    def value(self) -> float:
        return self.qty * self.est_price

    def to_dict(self) -> dict:
        return dict(asdict(self), est_price=round(self.est_price, 4), value=round(self.value, 2))


class Broker(ABC):
    name = "base"
    live = False

    @abstractmethod
    def account(self) -> dict:
        """{"cash": ドル, "equity": ドル（分かれば）}"""

    @abstractmethod
    def positions(self) -> dict[str, float]:
        """銘柄 → 株数"""

    @abstractmethod
    def submit(self, orders: list[Order], signal_date: str) -> list[dict]:
        """注文を出す。戻り値は証券会社の受付の記録。"""

    def cancel_all(self) -> None:
        pass

    def settle(self, panel) -> list[dict]:
        """（ペーパー口座だけ）出してある注文を、届いた株価データで約定させる。"""
        return []
