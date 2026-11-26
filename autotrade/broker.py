"""証券会社とのやりとりを1つの形にまとめる（バックテスト・ペーパートレード・将来の実口座で共通）。

- Broker: 共通の窓口（現金・保有の照会、注文、約定の反映）
- PaperBroker: 仮想の口座。注文は「翌営業日の寄付の成行」で、その日の始値に片道コストを乗せて約定する
  （kabuscout の過去ペーパートレード backtest.simulate と同じ約定の前提）。状態は JSON に保存できる
- 実口座（kabuステーションAPI など）は brokers/ に置く。口座が無い間は使わない（既定で無効）

株数・株価は「実際の株数・実際の株価」で持つ（分割調整後の株価ではない）。分割・併合があった日は株数と買値を直す。
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Position:
    code: str
    shares: int
    entry_px: float          # 1株あたりの取得価格（コスト込み・実際の株価）
    entry_date: str
    peak_px: float = 0.0     # 保有中の最高値（終値）。トレーリングの損切などに使う
    meta: dict = field(default_factory=dict)


@dataclass
class Order:
    code: str
    side: str                # "buy" / "sell"
    shares: int              # 実際の株数（単元の倍数）
    ref_px: float            # 判断に使った株価（前日の終値）。異常な約定の検出に使う
    reason: str = ""
    kind: str = "market_open"  # 寄付の成行


@dataclass
class Fill:
    date: str
    code: str
    side: str
    shares: int
    px: float                # 約定値（コスト込み）
    raw_px: float            # 市場の値段（コスト前）
    reason: str = ""
    pnl: float | None = None
    net: float | None = None  # 売りの損益率（コスト込み）
    entry_date: str | None = None  # 売りのとき: その保有を買った日


class Broker:
    """共通の窓口。実口座の実装もこれに合わせる。"""

    def cash(self) -> float:
        raise NotImplementedError

    def positions(self) -> dict[str, Position]:
        raise NotImplementedError

    def submit(self, orders: list[Order]) -> list[Order]:
        raise NotImplementedError

    def open_orders(self) -> list[Order]:
        raise NotImplementedError


class PaperBroker(Broker):
    """仮想の口座。submit した注文は次に execute_open を呼んだ日（翌営業日）の始値で約定する。"""

    def __init__(self, capital: float, cost: float = 0.002, lot: int = 100):
        self._cash = float(capital)
        self.capital = float(capital)
        self.half = cost / 2
        self.lot = lot
        self._pos: dict[str, Position] = {}
        self._pending: list[Order] = []
        self.fills: list[Fill] = []

    # ---- 照会
    def cash(self) -> float:
        return self._cash

    def positions(self) -> dict[str, Position]:
        return self._pos

    def open_orders(self) -> list[Order]:
        return list(self._pending)

    def equity(self, px: dict[str, float]) -> float:
        return self._cash + sum(p.shares * px.get(p.code, p.entry_px) for p in self._pos.values())

    # ---- 注文
    def submit(self, orders: list[Order]) -> list[Order]:
        ok = [o for o in orders if o.shares > 0 and o.shares % self.lot == 0 and o.side in ("buy", "sell")]
        self._pending.extend(ok)
        return ok

    def cancel_all(self) -> None:
        self._pending = []

    # ---- 1日の処理
    def apply_splits(self, ratios: dict[str, float]) -> None:
        """分割・併合（比率 r: 1株→r株）。株数を r 倍、取得価格を 1/r にする（端数は切り捨て、現金での精算は無視）。"""
        for code, r in ratios.items():
            p = self._pos.get(code)
            if p and r and r > 0 and abs(r - 1) > 1e-9:
                p.shares = int(math.floor(p.shares * r))
                p.entry_px /= r
                p.peak_px /= r
                if p.shares <= 0:
                    del self._pos[code]
        for o in self._pending:
            r = ratios.get(o.code)
            if r and r > 0 and abs(r - 1) > 1e-9:
                o.ref_px /= r
                o.shares = int(o.shares * r // self.lot * self.lot) if o.side == "buy" else int(o.shares * r)

    def execute_open(self, date: str, open_px: dict[str, float]) -> list[Fill]:
        """前日までの注文を今日の始値で約定する。売りを先に（その代金で買う）。値段が無い（売買停止など）注文は売りだけ翌日に回す。"""
        done, keep = [], []
        sells = [o for o in self._pending if o.side == "sell"]
        buys = [o for o in self._pending if o.side == "buy"]
        for o in sells:
            p = self._pos.get(o.code)
            px = open_px.get(o.code)
            if p is None:
                continue
            if not (px and px > 0 and math.isfinite(px)):
                keep.append(o)
                continue
            n = min(o.shares, p.shares)
            fill_px = px * (1 - self.half)
            self._cash += n * fill_px
            pnl = n * (fill_px - p.entry_px)
            f = Fill(date, o.code, "sell", n, fill_px, px, o.reason, pnl, fill_px / p.entry_px - 1, p.entry_date)
            self.fills.append(f)
            done.append(f)
            p.shares -= n
            if p.shares <= 0:
                del self._pos[o.code]
        for o in buys:
            px = open_px.get(o.code)
            if not (px and px > 0 and math.isfinite(px)):
                continue  # 寄らなかった買いは取り消し
            fill_px = px * (1 + self.half)
            n = min(o.shares, int(self._cash / fill_px // self.lot) * self.lot)  # 寄付が高く現金が足りなければ減らす
            if n <= 0:
                continue
            self._cash -= n * fill_px
            p = self._pos.get(o.code)
            if p:
                tot = p.shares + n
                p.entry_px = (p.entry_px * p.shares + fill_px * n) / tot
                p.shares = tot
            else:
                self._pos[o.code] = Position(o.code, n, fill_px, date, peak_px=px, meta={"reason": o.reason})
            f = Fill(date, o.code, "buy", n, fill_px, px, o.reason)
            self.fills.append(f)
            done.append(f)
        self._pending = keep
        return done

    def force_sell(self, date: str, code: str, px: float, reason: str) -> Fill | None:
        """値段が途切れた保有（上場廃止など）を、その値段で売ったものとして現金にする（注文を通さない）。"""
        p = self._pos.pop(code, None)
        if p is None:
            return None
        fill_px = px * (1 - self.half)
        self._cash += p.shares * fill_px
        f = Fill(date, code, "sell", p.shares, fill_px, px, reason, p.shares * (fill_px - p.entry_px), fill_px / p.entry_px - 1,
                 p.entry_date)
        self.fills.append(f)
        self._pending = [o for o in self._pending if o.code != code]
        return f

    def add_dividends(self, per_share: dict[str, float], tax: float = 0.20315) -> float:
        """権利落ち日に、保有株数 × 1株配当（税引き後）を現金に入れる。"""
        got = 0.0
        for code, d in per_share.items():
            p = self._pos.get(code)
            if p and d and d > 0 and math.isfinite(d):
                got += p.shares * d * (1 - tax)
        self._cash += got
        return got

    def mark(self, close_px: dict[str, float]) -> float:
        for p in self._pos.values():
            c = close_px.get(p.code)
            if c and math.isfinite(c):
                p.peak_px = max(p.peak_px, c)
        return self.equity(close_px)

    # ---- 保存
    def to_dict(self) -> dict:
        return dict(cash=self._cash, capital=self.capital, half=self.half, lot=self.lot,
                    positions=[asdict(p) for p in self._pos.values()], pending=[asdict(o) for o in self._pending],
                    fills=[asdict(f) for f in self.fills[-2000:]])

    @classmethod
    def from_dict(cls, d: dict) -> "PaperBroker":
        b = cls(d["capital"], d["half"] * 2, d["lot"])
        b._cash = d["cash"]
        b._pos = {p["code"]: Position(**p) for p in d["positions"]}
        b._pending = [Order(**o) for o in d["pending"]]
        b.fills = [Fill(**f) for f in d.get("fills", [])]
        return b

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=1))
        tmp.replace(path)  # 書きかけの状態を残さない

    @classmethod
    def load(cls, path: Path) -> "PaperBroker":
        return cls.from_dict(json.loads(Path(path).read_text()))
