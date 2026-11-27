"""戦略: 判定日の引けまでのデータだけで「各銘柄に資金の何割を置くか」（目標の比率）を決める。

どれも月1回の判定・売買の少ない戦略（米国株の短期売買は手数料に負けた＝docs/JOURNAL.md §57）。
条件は文献の定番の値をそのまま使い、過去データで調整していない（usscout.yaml の strategies）。
「1か月」は21営業日として数える（判定日をずらした検証でも同じ意味になるように）。

新しい戦略は Strategy を継承して target() を書き、REGISTRY に足すだけで、backtest・screen・rebalance から使える。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import Panel

M = 21  # 1か月の営業日数


def _ret(close: pd.DataFrame, months: float, skip: float = 0) -> pd.Series:
    """最終行（判定日）から見た騰落率。skip か月前 ÷ months か月前 − 1。足りない銘柄は NaN。"""
    n, k = int(round(months * M)), int(round(skip * M))
    if len(close) <= n:
        return pd.Series(np.nan, index=close.columns)
    return close.iloc[-1 - k] / close.iloc[-1 - n] - 1


def _monthly_sma(close: pd.DataFrame, months: int) -> pd.Series:
    """判定日から21営業日ずつさかのぼった months 個の終値の平均（Faber の10か月平均の、判定日をずらせる版）。"""
    pos = [len(close) - 1 - i * M for i in range(months)]
    if pos[-1] < 0:
        return pd.Series(np.nan, index=close.columns)
    return close.iloc[pos].mean()


def cash_return(p: Panel, months: int, cash: str = "BIL") -> float:
    """短期国債の months か月の利回り。BIL の履歴が足りない時期（2007〜2008年）は ^IRX（13週国債の利回り）で近似。"""
    c = p.close
    if cash in c and len(c) > months * M and pd.notna(c[cash].iloc[-1 - months * M]):
        return float(c[cash].iloc[-1] / c[cash].iloc[-1 - months * M] - 1)
    if "^IRX" in c:
        irx = c["^IRX"].iloc[-months * M:].dropna()
        if len(irx):
            return float(irx.mean() / 100 * months / 12)
    return 0.0


class Strategy:
    name = "base"

    def __init__(self, params: dict, cfg: dict | None = None):
        self.p = params
        self.cfg = cfg or {}
        self.prev: dict[str, float] = {}  # 前回の目標（持ち続ける判定に使う）

    def target(self, p: Panel, members: set[str] | None = None) -> dict[str, float]:
        raise NotImplementedError

    def __call__(self, p: Panel, members: set[str] | None = None) -> dict[str, float]:
        w = {k: float(v) for k, v in self.target(p, members).items() if v > 1e-9}
        self.prev = w
        return w

    def explain(self) -> list[dict]:
        """直近の判定の理由（screen で表示）。"""
        return getattr(self, "_why", [])


class Hold(Strategy):
    """1銘柄を持ち続ける（物差し: SPY・QQQ）。"""
    name = "hold"

    def target(self, p, members=None):
        return {self.p["asset"]: 1.0}


class GEM(Strategy):
    """Antonacci のデュアルモメンタム: 米国株の12か月騰落が短期国債より良ければ、米国株と先進国株の強い方。悪ければ債券。"""
    name = "gem"

    def target(self, p, members=None):
        L = self.p.get("lookback_months", 12)
        r = _ret(p.close[list(self.p["risk"]) + [self.p["safe"]]], L)
        cr = cash_return(p, L, self.p.get("cash", "BIL"))
        lead = self.p["risk"][0]
        if pd.notna(r[lead]) and r[lead] > cr:
            pick = r[list(self.p["risk"])].idxmax()
        else:
            pick = self.p["safe"]
        self._why = [{"symbol": k, "ret_%dm" % L: round(float(v), 4)} for k, v in r.items()] + [{"symbol": "cash", "ret_%dm" % L: round(cr, 4)}]
        return {pick: 1.0}


class ETFRotation(Strategy):
    """ETF の中から 1・3・6・12か月の騰落の平均（勢い）が高い top 本を同じ金額ずつ。勢いがマイナスの枠は短期国債。"""
    name = "etf_rotation"

    def target(self, p, members=None):
        uni = [u for u in self.p["universe"] if u in p.close]
        c = p.close[uni]
        score = pd.concat([_ret(c, m) for m in self.p.get("windows_months", [1, 3, 6, 12])], axis=1).mean(axis=1, skipna=False)
        score = score.dropna().sort_values(ascending=False)
        top = int(self.p.get("top", 3))
        w: dict[str, float] = {}
        for sym, sc in score.iloc[:top].items():
            k = sym if sc > 0 else self.p["cash"]
            w[k] = w.get(k, 0) + 1.0 / top
        missing = top - min(top, len(score))
        if missing:
            w[self.p["cash"]] = w.get(self.p["cash"], 0) + missing / top
        self._why = [{"symbol": s, "score": round(float(v), 4), "rank": i + 1} for i, (s, v) in enumerate(score.items())]
        return w


class TrendHold(Strategy):
    """1つの指数 ETF を、判定日の終値が10か月平均より上のときだけ持つ。下なら短期国債。"""
    name = "trend"

    def target(self, p, members=None):
        a = self.p["asset"]
        g = self.p.get("signal", a)  # 判定に使う指数（既定は持つものと同じ。2倍の ETF を持つときは SPY で判定）
        c = p.close[[g]]
        sma = _monthly_sma(c, self.p.get("sma_months", 10))[g]
        last = c[g].iloc[-1]
        up = pd.notna(sma) and last > sma
        self._why = [{"symbol": g, "close": round(float(last), 2), "sma": round(float(sma), 2) if pd.notna(sma) else None, "up": bool(up), "hold": a if up else self.p["cash"]}]
        return {a: 1.0} if up else {self.p["cash"]: 1.0}


class VolManaged(Strategy):
    """値動きが落ち着いているときほど多く持つ（Moreira & Muir、ジャーナル §92。**事前の基準では不合格**、前向きのペーパー口座で見守る用）。
    指数への投資比率 = その日までの全期間の日々の騰落の分散 ÷ 直近21営業日の分散（0〜cap 倍）。2倍の ETF を 比率/2、残りを現金で持つ。"""
    name = "vol_managed"

    def target(self, p, members=None):
        P = self.p
        r = p.close[P["signal"]].dropna().pct_change().dropna()
        n = int(P.get("window_days", 21))
        if len(r) < max(n, 252):
            return {P["cash"]: 1.0}
        e = float(min(r.var() / r.iloc[-n:].var(), float(P.get("cap", 2.0))))
        lev = float(P.get("leverage", 2.0))
        w = {P["asset"]: e / lev}
        if e / lev < 1:
            w[P["cash"]] = 1 - e / lev
        self._why = [{"symbol": P["signal"], "exposure": round(e, 3), "recent_vol": round(float(r.iloc[-n:].std() * np.sqrt(252)), 4),
                      "long_vol": round(float(r.std() * np.sqrt(252)), 4)}]
        return w


class StockMomentum(Strategy):
    """S&P500 のその時点の構成銘柄から、12−1か月の騰落が高い銘柄を top 個（同じ金額ずつ）。

    - 個別株の条件: 終値が200日平均より上・株価5ドル以上・1年以上の値動きがある
    - 持っている銘柄は順位が keep_rank 以内なら持ち続ける（売買の回数を減らす）
    - 地合い: SPY が10か月平均より下なら全部を短期国債へ
    - score="random" にすると、同じ条件を満たす銘柄からランダムに選ぶ（比較用。ランダムより良いかを確かめる）
    """
    name = "stock_mom"

    def __init__(self, params, cfg=None, seed: int | None = None):
        super().__init__(params, cfg)
        self.rng = np.random.default_rng(seed) if seed is not None else None

    def target(self, p, members=None):
        P = self.p
        cash, top, keep = P["cash"], int(P.get("top", 10)), int(P.get("keep_rank", 20))
        mkt = p.close[[P["market"]]]
        msma = _monthly_sma(mkt, P["market_sma_months"])[P["market"]] if P.get("market_sma_months") else None
        if msma is not None and (pd.isna(msma) or mkt[P["market"]].iloc[-1] <= msma):
            self._why = [{"symbol": P["market"], "note": "地合いが弱い（10か月平均より下）→ 短期国債"}]
            return {cash: 1.0}
        # 並びを固定する（set の順番は実行ごとに変わり、ランダム比較・同点の順位が再現できなくなる）
        cols = sorted(c for c in (members if members is not None else p.close.columns) if c in p.close.columns)
        c = p.close[cols]
        L, skip, n_sma = P.get("lookback_months", 12), P.get("skip_months", 1), int(P.get("stock_sma_days", 200))
        tail = c.iloc[-max(n_sma, int(L * M) + 1):]
        last = tail.iloc[-1]
        ok = last.notna() & tail.iloc[0].notna() & (p.raw[cols].iloc[-1] >= 5)
        if n_sma:
            ok &= last > tail.iloc[-n_sma:].mean()
        mom = _ret(c, L, skip)[ok].dropna()
        score = pd.Series(self.rng.random(len(mom)), index=mom.index) if self.rng is not None else mom
        ranked = score.sort_values(ascending=False)
        rank = pd.Series(np.arange(1, len(ranked) + 1), index=ranked.index)
        held = [s for s in self.prev if s in rank.index and rank[s] <= keep]
        held = sorted(held, key=lambda s: rank[s])[:top]
        picks = held + [s for s in ranked.index if s not in held][: top - len(held)]
        w = {s: 1.0 / top for s in picks}
        if len(picks) < top:
            w[cash] = (top - len(picks)) / top
        self._why = [{"symbol": s, "mom_12_1": round(float(mom[s]), 4), "rank": int(rank[s]), "kept": s in held} for s in picks]
        return w


class EqualWeight(Strategy):
    """その時点の構成銘柄（株価のある銘柄）を全部同じ金額ずつ（物差し: 同じ母集団・同じ生存者バイアス）。"""
    name = "equal_weight"

    def target(self, p, members=None):
        cols = sorted(c for c in (members or []) if c in p.close.columns and pd.notna(p.close[c].iloc[-1]))
        return {c: 1.0 / len(cols) for c in cols} if cols else {"BIL": 1.0}


class Blend(Strategy):
    """複数の戦略に資金を分ける（各戦略の目標の比率を、配分で重み付けして足す）。"""
    name = "blend"

    def __init__(self, params, cfg=None, parts: dict | None = None):
        super().__init__(params, cfg)
        self.parts = parts or {}  # name -> (Strategy, share)

    def target(self, p, members=None):
        w: dict[str, float] = {}
        self._why = []
        for name, (s, share) in self.parts.items():
            for k, v in s(p, members).items():
                w[k] = w.get(k, 0) + share * v
            self._why += [dict(part=name, **x) for x in s.explain()]
        return w


REGISTRY = {"gem": GEM, "etf_rotation": ETFRotation, "trend_spy": TrendHold, "trend_qqq": TrendHold, "stock_mom": StockMomentum,
            "lev_trend_spy": TrendHold, "vol_managed_spy": VolManaged}
NEEDS_MEMBERS = {"stock_mom", "equal_weight", "blend"}


def build(name: str, cfg: dict, overrides: dict | None = None, seed: int | None = None) -> Strategy:
    """設定（usscout.yaml の strategies）から戦略を作る。overrides で条件の一部を差し替え（感度の確認用）。"""
    st = cfg["strategies"]
    if name.startswith("hold_"):
        return Hold({"asset": name[5:].upper()}, cfg)
    if name == "equal_weight":
        return EqualWeight({}, cfg)
    if name == "blend":
        parts = {k: (build(k, cfg, (overrides or {}).get(k), seed), float(v)) for k, v in st["blend"]["parts"].items()}
        return Blend(st["blend"], cfg, parts)
    params = dict(st[name], **(overrides or {}))
    if name == "stock_mom":
        return StockMomentum(params, cfg, seed=seed if params.get("score") == "random" else None)
    return REGISTRY[name](params, cfg)
