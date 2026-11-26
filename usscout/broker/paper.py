"""手元のペーパー口座（作り物のお金）。data/us_scout/paper_account.json に保存する。

本番と同じ流れ: 判定日の夜に寄付の成行を出す → 次に株価を取ったとき、判定日の次の営業日の「実際の寄付の値段」で約定。
コストは usscout.yaml の costs（手数料・気配差）。

保有は過去の検証（backtest）と同じく「調整後の株数」（units）で持つ。実際の株数 = units × 調整後の終値 ÷ 実際の終値。
こうすると分割（4対1 など）で実際の株数が自動で増え、配当は同じ銘柄に再投資したとみなされる（過去の検証と同じ扱い）。
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
            self.s = {"capital": cap, "cash": cap, "units": {}, "last_close": {}, "last_ratio": {}, "pending": [], "fills": [],
                      "peak_equity": cap, "last_signal": None, "history": []}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.s, ensure_ascii=False, indent=1))

    def account(self) -> dict:
        lc = self.s["last_close"]
        eq = self.s["cash"] + sum(u * lc.get(k, 0.0) for k, u in self.s["units"].items())
        return {"cash": self.s["cash"], "equity": eq, "peak_equity": self.s["peak_equity"], "pending": len(self.s["pending"])}

    def positions(self) -> dict[str, float]:
        """実際の株数（直近の評価の日の値段で換算）。"""
        return {k: u / self.s["last_ratio"].get(k, 1.0) for k, u in self.s["units"].items()}

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
        for o in sorted(self.s["pending"], key=lambda o: o["side"] != "sell"):
            later = cal[cal > pd.Timestamp(o["signal_date"])]
            sym = o["symbol"]
            if not len(later) or sym not in p.close:
                keep.append(o)
                continue
            d = later[0]
            adj_o, adj_c, raw_c = p.open.at[d, sym], p.close.at[d, sym], p.raw.at[d, sym]
            if pd.isna(adj_c) or pd.isna(raw_c) or adj_c <= 0 or raw_c <= 0:
                keep.append(o)
                continue
            ratio = float(raw_c / adj_c)                       # 実際の値段 ÷ 調整後の値段（その日）
            px = float(adj_o) * ratio if pd.notna(adj_o) and adj_o > 0 else float(raw_c)
            q = float(o["qty"])
            held = self.s["units"].get(sym, 0.0) / ratio       # その日の実際の株数
            if o["side"] == "sell":
                q = min(q, held)
                v = q * px
                c = order_cost(v, self.costs, sym)
                self.s["cash"] += v - c
                if held - q > 1e-9:
                    self.s["units"][sym] = (held - q) * ratio
                else:
                    self.s["units"].pop(sym, None)
            else:
                c_rate = self.costs.get("commission", 0) + self.costs.get("spread", 0)
                q = min(q, math.floor(self.s["cash"] / (px * (1 + c_rate))))
                if q <= 0:
                    done.append(dict(o, status="現金不足で見送り", fill_date=str(d.date())))
                    continue
                v = q * px
                c = order_cost(v, self.costs, sym)
                self.s["cash"] -= v + c
                self.s["units"][sym] = (held + q) * ratio
            self.s["last_close"][sym], self.s["last_ratio"][sym] = float(adj_c), ratio
            f = dict(symbol=sym, side=o["side"], qty=q, price=round(px, 4), cost=round(c, 4), fill_date=str(d.date()),
                     signal_date=o["signal_date"], status="約定")
            self.s["fills"].append(f)
            done.append(f)
        self.s["pending"] = keep
        self.save()
        return done

    def mark(self, p, save: bool = True) -> dict:
        """最新の値段で評価し、最高値を更新する（分割・配当は調整後の値段で自動的に入る）。"""
        for k in self.s["units"]:
            if k in p.close:
                c, r = p.close[k].dropna(), p.raw[k].dropna()
                if len(c) and len(r) and c.iloc[-1] > 0:
                    self.s["last_close"][k] = float(c.iloc[-1])
                    self.s["last_ratio"][k] = float(r.iloc[-1] / c.iloc[-1])
        a = self.account()
        if save:
            self.s["peak_equity"] = max(self.s["peak_equity"], a["equity"])
            day = str(p.close.index[-1].date())
            if not self.s["history"] or self.s["history"][-1]["date"] != day:
                self.s["history"].append({"date": day, "equity": round(a["equity"], 2)})
            self.save()
        return self.account()
