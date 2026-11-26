"""kabuステーションAPI（三菱UFJ eスマート証券）の窓口の雛形。

⚠️ 未検証: 口座が無いため、一度も実際の API に接続していない。項目名・値の意味は公式の資料
（https://kabucom.github.io/kabusapi/ptal/ ）で必ず確かめてから使うこと。ここに書いた値は記憶に基づく下書き。

安全のための約束:
- 既定は dry_run=True（注文の中身をログに出すだけで送らない）
- 実際に送るのは、環境変数 AUTOTRADE_LIVE=1 かつ dry_run=False のときだけ
- 現物の買い・売りの「寄付の成行」だけ。信用取引・指値の変更などは作っていない
- kabuステーション（Windows のアプリ）が動いている PC の上でしか使えない（localhost の API）
"""
from __future__ import annotations

import logging
import os

from ..broker import Broker, Order, Position

log = logging.getLogger(__name__)


class KabusAPIBroker(Broker):
    def __init__(self, password_env: str = "KABUS_API_PASSWORD", port: int = 18081, dry_run: bool = True,
                 account_type: int = 4):
        # 18080 = 本番、18081 = 検証用（公式の資料で確認すること）
        self.base = f"http://localhost:{port}/kabusapi"
        self.password_env = password_env
        self.dry_run = dry_run or os.environ.get("AUTOTRADE_LIVE") != "1"
        self.account_type = account_type  # 4 = 特定口座（要確認）
        self._token = None

    def _session(self):
        import requests
        if self._token is None:
            r = requests.post(f"{self.base}/token", json={"APIPassword": os.environ[self.password_env]}, timeout=10)
            r.raise_for_status()
            self._token = r.json()["Token"]
        s = requests.Session()
        s.headers["X-API-KEY"] = self._token
        return s

    def _order_body(self, o: Order) -> dict:
        buy = o.side == "buy"
        return {
            "Symbol": o.code, "Exchange": 1, "SecurityType": 1,
            "Side": "2" if buy else "1",          # 2 = 買い、1 = 売り（要確認）
            "CashMargin": 1,                      # 1 = 現物（要確認）
            "DelivType": 2 if buy else 0,         # 要確認
            "FundType": "AA" if buy else "  ",    # 要確認
            "AccountType": self.account_type,
            "Qty": int(o.shares), "Price": 0,
            "FrontOrderType": 13,                 # 寄成（前場）のつもり（要確認）
            "ExpireDay": 0,
        }

    def submit(self, orders: list[Order]) -> list[Order]:
        sent = []
        for o in orders:
            body = self._order_body(o)
            if self.dry_run:
                log.info("[dry-run] 送らない注文: %s", body)
                continue
            r = self._session().post(f"{self.base}/sendorder", json=body, timeout=10)
            r.raise_for_status()
            log.info("注文を送信: %s → %s", body, r.json())
            sent.append(o)
        return sent

    def cash(self) -> float:
        if self.dry_run:
            raise RuntimeError("dry-run 中は残高を照会しない")
        return float(self._session().get(f"{self.base}/wallet/cash", timeout=10).json()["StockAccountWallet"])

    def positions(self) -> dict[str, Position]:
        if self.dry_run:
            raise RuntimeError("dry-run 中は保有を照会しない")
        out = {}
        for p in self._session().get(f"{self.base}/positions", params={"product": 1}, timeout=10).json():
            out[p["Symbol"]] = Position(p["Symbol"], int(p["LeavesQty"]), float(p["Price"]), "")
        return out

    def open_orders(self) -> list[Order]:
        return []
