"""手元のペーパー口座（作り物のお金）。data/us_scout/paper_account.json に保存する。

本番と同じ流れ: 判定日の夜に寄付の成行を出す → 次に株価を取ったとき、判定日の次の営業日の「実際の寄付の値段」で約定。
コストは usscout.yaml の costs（手数料・気配差）。配当は入れていない（実際の口座より少し低く出る）。
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from ..backtest import order_cost
from .base import Broker, Order


class PaperBroker(Broker):
    name = "paper"
    live = False

    def __init__(self, path: Path, cfg: dict):
        self.path = Path(path)
        self.costs = cfg["costs"]
        if self.path.exists():
            self.s = json.loads(self.path.read_text())
        else:
            cap = float(cfg["backtest"]["capital_usd"])
            self.s = {"capital": cap, "cash": cap, "positions": {}, "pending": [], "fills": [], "last_prices": {},
                      "peak_equity": cap, "last_signal": None, "history": []}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.s, ensure_ascii=False, indent=1))

    def account(self) -> dict:
        lp = self.s["last_prices"]
        eq = self.s["cash"] + sum(q * lp.get(k, 0.0) for k, q in self.s["positions"].items())
        return {"cash": self.s["cash"], "equity": eq, "peak_equity": self.s["peak_equity"], "pending": len(self.s["pending"])}

    def positions(self) -> dict[str, float]:
        return dict(self.s["positions"])

    def submit(self, orders: list[Order], signal_date: str) -> list[dict]:
        out = []
        for o in orders:
            d = dict(o.to_dict(), signal_date=signal_date)
            self.s["pending"].append(d)
            out.append(dict(d, status="受付（次の営業日の寄付で約定）"))
        self.s["last_signal"] = signal_date
        self.save()
        return out

    def cancel_all(self) -> None:
        self.s["pending"] = []
        self.save()

    def settle(self, p) -> list[dict]:
        """判定日の次の営業日の寄付（実際の値段）で約定させる。売りを先に、現金が足りなければ買いの株数を減らす。"""
        done, keep = [], []
        cal = p.close["SPY"].dropna().index
        pend = sorted(self.s["pending"], key=lambda o: o["side"] != "sell")
        for o in pend:
            later = cal[cal > pd.Timestamp(o["signal_date"])]
            sym = o["symbol"]
            if not len(later) or sym not in p.close:
                keep.append(o)
                continue
            d = later[0]
            adj_o, adj_c, raw_c = p.open.at[d, sym], p.close.at[d, sym], p.raw.at[d, sym]
            if pd.isna(adj_c) or pd.isna(raw_c) or adj_c <= 0:
                keep.append(o)
                continue
            px = float(adj_o * raw_c / adj_c) if pd.notna(adj_o) and adj_o > 0 else float(raw_c)
            q = float(o["qty"])
            held = self.s["positions"].get(sym, 0.0)
            if o["side"] == "sell":
                q = min(q, held)
                v = q * px
                c = order_cost(v, self.costs, sym)
                self.s["cash"] += v - c
                left = held - q
                if left > 1e-9:
                    self.s["positions"][sym] = left
                else:
                    self.s["positions"].pop(sym, None)
            else:
                c_rate = self.costs.get("commission", 0) + self.costs.get("spread", 0)
                q = min(q, math.floor(self.s["cash"] / (px * (1 + c_rate))))
                if q <= 0:
                    done.append(dict(o, status="現金不足で見送り", fill_date=str(d.date())))
                    continue
                v = q * px
                c = order_cost(v, self.costs, sym)
                self.s["cash"] -= v + c
                self.s["positions"][sym] = held + q
            f = dict(symbol=sym, side=o["side"], qty=q, price=round(px, 4), cost=round(c, 4), fill_date=str(d.date()),
                     signal_date=o["signal_date"], status="約定")
            self.s["fills"].append(f)
            done.append(f)
        self.s["pending"] = keep
        self.save()
        return done

    def mark(self, p) -> dict:
        """最新の実際の終値で評価し、最高値を更新する。"""
        for k in set(self.s["positions"]) | set(self.s["last_prices"]):
            if k in p.raw:
                v = p.raw[k].dropna()
                if len(v):
                    self.s["last_prices"][k] = float(v.iloc[-1])
        a = self.account()
        self.s["peak_equity"] = max(self.s["peak_equity"], a["equity"])
        day = str(p.close.index[-1].date())
        if not self.s["history"] or self.s["history"][-1]["date"] != day:
            self.s["history"].append({"date": day, "equity": round(a["equity"], 2)})
        self.save()
        return self.account()
