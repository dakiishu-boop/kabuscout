"""発注の窓口（証券会社ごとの違いをここで吸収する）。

- paper: 手元の作り物の口座（既定）。注文は「次の営業日の寄付」で、株価データが届いた後に約定させる
- alpaca: Alpaca（米国の証券会社）の REST API のひな形。**この環境からは接続できず、動作は未確認**。
  日本の居住者が口座を開けるかも確認していない（docs/US.md）
新しい証券会社は base.Broker を継承して account / positions / submit / cancel_all を書き、make() に足す。

実際のお金での発注は、usscout.yaml の trading.live: true と環境変数 USSCOUT_LIVE=1 の両方がそろったときだけ。
"""
from __future__ import annotations

import os

from .base import Broker, LiveTradingRefused, Order


def live_allowed(cfg: dict) -> bool:
    return bool(cfg.get("trading", {}).get("live")) and os.environ.get("USSCOUT_LIVE") == "1"


def make(cfg: dict, name: str | None = None) -> Broker:
    name = name or cfg["trading"].get("broker", "paper")
    if cfg["trading"].get("live") and os.environ.get("USSCOUT_LIVE") != "1":
        raise LiveTradingRefused("usscout.yaml は live: true ですが、環境変数 USSCOUT_LIVE=1 がありません。実際のお金での発注は止めました")
    if name == "paper":
        from .paper import PaperBroker
        return PaperBroker(cfg["data"]["dir"] / "paper_account.json", cfg)
    if name == "alpaca":
        from .alpaca import AlpacaBroker
        return AlpacaBroker(live=live_allowed(cfg))
    raise ValueError(f"知らない証券会社: {name}")


__all__ = ["Broker", "Order", "LiveTradingRefused", "make", "live_allowed"]
