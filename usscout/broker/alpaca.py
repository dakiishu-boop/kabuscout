"""Alpaca（米国の証券会社）の REST API v2 のひな形。

**未検証**: この開発環境からは Alpaca に接続できず、一度も実際に動かしていない。日本の居住者が口座を開けるかも確認していない。
使う前に、Alpaca のペーパー口座（作り物のお金）で必ず試すこと。

- 鍵は環境変数 ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY（.env でも可。リポジトリには絶対に書かない）
- 既定はペーパー口座の URL。実際のお金の URL は live=True（usscout.yaml の trading.live: true ＋ USSCOUT_LIVE=1）のときだけ
- 注文は寄付の成行（time_in_force=opg）。整数の株数だけ（opg では端数株を出せない）
"""
from __future__ import annotations

import os

import requests

from .base import Broker, Order

PAPER_URL = "https://paper-api.alpaca.markets"
LIVE_URL = "https://api.alpaca.markets"


class AlpacaBroker(Broker):
    name = "alpaca"

    def __init__(self, live: bool = False, key: str | None = None, secret: str | None = None, timeout: int = 30):
        self.live = bool(live)
        self.base = LIVE_URL if self.live else PAPER_URL
        self.key = key or os.environ.get("ALPACA_API_KEY_ID", "")
        self.secret = secret or os.environ.get("ALPACA_API_SECRET_KEY", "")
        if not self.key or not self.secret:
            raise RuntimeError("ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY が設定されていません")
        self.timeout = timeout

    def _h(self) -> dict:
        return {"APCA-API-KEY-ID": self.key, "APCA-API-SECRET-KEY": self.secret}

    def _req(self, method: str, path: str, **kw):
        r = requests.request(method, self.base + path, headers=self._h(), timeout=self.timeout, **kw)
        r.raise_for_status()
        return r.json() if r.content else None

    def account(self) -> dict:
        a = self._req("GET", "/v2/account")
        return {"cash": float(a["cash"]), "equity": float(a["equity"]), "status": a.get("status")}

    def positions(self) -> dict[str, float]:
        return {p["symbol"]: float(p["qty"]) for p in self._req("GET", "/v2/positions")}

    def submit(self, orders: list[Order], signal_date: str) -> list[dict]:
        out = []
        for o in orders:
            qty = int(o.qty) if o.side == "buy" else o.qty
            if qty <= 0:
                continue
            body = {"symbol": o.symbol, "qty": str(qty), "side": o.side, "type": "market", "time_in_force": o.tif,
                    "client_order_id": f"usscout-{signal_date}-{o.symbol}-{o.side}"}
            out.append(self._req("POST", "/v2/orders", json=body))
        return out

    def cancel_all(self) -> None:
        self._req("DELETE", "/v2/orders")
