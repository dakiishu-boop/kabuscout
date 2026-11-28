"""自動売買: 今月の目標の比率 → 注文 → 安全装置 → 証券会社へ（既定は表示だけ・手元のペーパー口座）。

使い方（docs/US.md）:
  python -m usscout fetch                      # 株価を取り直す（判定日の引けの後＝日本時間の朝）
  python -m usscout rebalance                  # 注文の案を表示するだけ
  python -m usscout rebalance --execute        # ペーパー口座に注文を出す（次の営業日の寄付で約定）
判定日（月の最後の営業日）の引けの後、次の営業日の寄付（日本時間 23:30、夏時間は 22:30）までに実行する。
"""
from __future__ import annotations

import json
import math
from datetime import date, time, timedelta

import pandas as pd

from . import broker as B
from . import data as D
from . import universe
from .backtest import _members_at
from .broker.base import Order
from .strategies import NEEDS_MEMBERS, build


# ---- 営業日（NYSE）。月末の判定に要る分だけ: 土日・聖金曜日・メモリアルデー（月末に当たりうる休み） ----
def _easter(y: int) -> date:
    a, b, c = y % 19, y // 100, y % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7  # noqa: E741
    m = (a + 11 * h + 22 * l) // 451
    mo = (h + l - 7 * m + 114) // 31
    return date(y, mo, (h + l - 7 * m + 114) % 31 + 1)


def last_trading_day(y: int, m: int) -> date:
    d = (date(y + (m == 12), m % 12 + 1, 1) - timedelta(days=1))
    good_friday = _easter(y) - timedelta(days=2)
    while True:
        memorial = m == 5 and d.weekday() == 0 and d.day > 24
        if d.weekday() < 5 and d != good_friday and not memorial:
            return d
        d -= timedelta(days=1)


def signal_date(dates: pd.DatetimeIndex) -> pd.Timestamp:
    """直近の「確定した」判定日。データの最終日がその月の最後の営業日ならその日、そうでなければ前の月の最後の営業日。"""
    last = dates[-1]
    if last.date() == last_trading_day(last.year, last.month):
        return last
    prev = dates[dates < last.replace(day=1)]
    return prev[-1]


def _drop_unfinished(p: D.Panel) -> D.Panel:
    """米国の取引時間中に取ったデータの「今日」の行（まだ終値でない）を落とす。"""
    now = pd.Timestamp.now(tz="America/New_York")
    if p.dates[-1].date() == now.date() and now.time() < time(16, 30):
        return p.upto(p.dates[-2])
    return p


def load(cfg: dict):
    root = cfg["data"]["dir"]
    p = _drop_unfinished(D.with_cash(D.load(root)))
    cur, ch = universe.fetch_sp500(root)
    mem = universe.membership(cur, ch, D.month_ends(D.calendar(p)))
    return p, mem


def _strategy(cfg: dict, name: str, p: D.Panel, mem, s: pd.Timestamp):
    """戦略を作り、直近3年の判定をやり直して「前回の目標」（持ち続ける判定に使う）を本番と同じ状態にする。"""
    strat = build(name, cfg)
    me = D.month_ends(D.calendar(p))
    for d in me[(me < s) & (me >= s - pd.DateOffset(years=3))]:
        strat(p.upto(d), _have(p, mem, d) if name in NEEDS_MEMBERS else None)
    return strat


def _have(p, mem, d):
    m = _members_at(mem, d)
    return {x for x in m if x in p.close.columns and pd.notna(p.close.at[d, x])} if m is not None else None


def targets(cfg: dict, name: str | None = None, data=None):
    """data=(株価, 構成銘柄) を渡すと読み込まない（過去をたどる試し運転 replay 用）。"""
    name = name or cfg["trading"]["strategy"]
    p, mem = data if data is not None else load(cfg)
    s = signal_date(D.calendar(p))
    strat = _strategy(cfg, name, p, mem, s)
    w = strat(p.upto(s), _have(p, mem, s) if name in NEEDS_MEMBERS else None)
    return p, s, name, w, strat


def screen(cfg: dict, name: str | None = None) -> dict:
    p, s, name, w, strat = targets(cfg, name)
    return {"signal_date": str(s.date()), "data_end": str(p.dates[-1].date()), "strategy": name,
            "targets": {k: round(v, 4) for k, v in sorted(w.items(), key=lambda x: -x[1])},
            "reasons": strat.explain()[:40],
            "note": "CASH = 米ドルのまま（米ドルMMF）。BIL = 短期国債の ETF。判定日の次の営業日の寄付で組み直す"}


def plan_orders(w: dict, pos: dict, px: dict, equity: float, cfg: dict, cash: float | None = None) -> list[Order]:
    """目標の比率と今の保有から注文を作る（売りが先）。1株単位、比率のずれが小さい銘柄は売買しない。
    cash を渡すと、売った後の現金で足りない分は買いを同じ割合で減らす（過去の検証 backtest と同じ）。"""
    band = float(cfg["backtest"].get("rebalance_band", 0))
    min_usd = float(cfg["trading"].get("min_order_usd", 0))
    c = cfg["costs"]
    buffer = 1 - (c.get("commission", 0) + c.get("spread", 0))
    orders = []
    for k in sorted(set(pos) | set(w)):
        if k == D.CASH:
            continue
        price = px.get(k)
        if price is None or not price > 0:
            continue
        q0 = float(pos.get(k, 0.0))
        tq = math.floor(w.get(k, 0.0) * equity * buffer / price)
        if k in pos and w.get(k, 0) > 0 and abs(tq - q0) * price < band * equity:
            continue
        dq = tq - q0
        if w.get(k, 0) <= 0:
            dq = -q0  # 目標から外れた銘柄は全部売る
        if abs(dq) < 1e-9 or (abs(dq) * price < min_usd and w.get(k, 0) > 0):
            continue
        orders.append(Order(k, "buy" if dq > 0 else "sell", abs(dq), price, note="新規" if q0 == 0 else ("全部売る" if tq == 0 else "調整")))
    if cash is not None:
        rate = c.get("commission", 0) + c.get("spread", 0)
        avail = cash + sum(o.value * (1 - rate) for o in orders if o.side == "sell")
        need = sum(o.value * (1 + rate) for o in orders if o.side == "buy")
        if need > avail > 0:
            scale = avail / need
            for o in orders:
                if o.side == "buy":
                    o.qty = float(math.floor(o.qty * scale))
            orders = [o for o in orders if o.qty > 0]
    return sorted(orders, key=lambda o: o.side != "sell")


def _now(now=None) -> pd.Timestamp:
    return pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="America/New_York").tz_localize(None)


def checks(cfg: dict, orders: list[Order], w: dict, equity: float, peak: float, p: D.Panel, s: pd.Timestamp,
           now=None) -> tuple[list[str], list[str], list[Order]]:
    """安全装置。(止める理由, 注意, 出してよい注文)"""
    t = cfg["trading"]
    blocked, warn = [], []
    if (cfg["data"]["dir"] / "STOP").exists():
        blocked.append(f"緊急停止のファイル {cfg['data']['dir'] / 'STOP'} があります（消すと再開）")
    age = (_now(now).date() - p.dates[-1].date()).days
    if age > int(t.get("max_data_age_days", 4)):
        blocked.append(f"株価データが古い（最終日 {p.dates[-1].date()}、{age}日前）。先に fetch を")
    if equity <= 0:
        blocked.append("口座の評価額が0以下")
        return blocked, warn, []
    turn = sum(o.value for o in orders) / equity
    if turn > float(t.get("max_turnover", 1.1)):
        blocked.append(f"売買の合計が資金の {turn:.2f} 倍（上限 {t.get('max_turnover')}）。データの異常を疑って止めました")
    lim = max(float(t.get("max_order_fraction", 0.4)), max(w.values(), default=0) + 0.02)
    for o in orders:
        if o.side == "buy" and o.value > lim * equity:
            blocked.append(f"{o.symbol} の買い {o.value:,.0f} ドルが資金の {lim:.0%} を超える")
        if o.side == "buy" and o.symbol in p.close:  # 調整後の値段で見る（分割での値段の変化を誤って止めない）
            r = p.close[o.symbol].loc[:s].dropna().iloc[-2:]
            if len(r) == 2 and abs(r.iloc[1] / r.iloc[0] - 1) > 0.5:
                blocked.append(f"{o.symbol} の値段が前日から50%以上動いている（データの誤りの疑い）")
    if peak > 0 and equity < (1 - float(t.get("stop_dd", 0.35))) * peak:
        warn.append(f"口座が最高値から {1 - equity / peak:.0%} 下落（目安 {t.get('stop_dd'):.0%}）。新規の買いを止め、売りだけ出します")
        orders = [o for o in orders if o.side == "sell"]
    return blocked, warn, orders


def _state_path(cfg):
    return cfg["data"]["dir"] / "trader_state.json"


def rebalance(cfg: dict, strategy: str | None = None, broker: str | None = None, execute: bool = False, force: bool = False,
              data=None, now=None, br=None) -> dict:
    p, s, name, w, _ = targets(cfg, strategy, data)
    br = br or B.make(cfg, broker)
    fills = br.settle(p) if execute else []
    acct = br.mark(p, save=execute) if hasattr(br, "mark") else br.account()
    pos = br.positions()
    px = {k: float(p.raw[k].loc[:s].dropna().iloc[-1]) for k in set(pos) | set(w) if k in p.raw and len(p.raw[k].loc[:s].dropna())}
    eq = acct.get("equity") or acct["cash"] + sum(q * px.get(k, 0) for k, q in pos.items())
    st = json.loads(_state_path(cfg).read_text()) if _state_path(cfg).exists() else {}
    out = {"signal_date": str(s.date()), "strategy": name, "broker": br.name, "live": br.live, "execute": execute,
           "equity": round(eq, 2), "positions": pos, "targets": {k: round(v, 4) for k, v in w.items()}, "fills": fills}
    if st.get(br.name) == str(s.date()) and not force:
        out.update(orders=[], note=f"{s.date()} の判定ではもう注文済み（やり直すなら --force）")
        return out
    orders = plan_orders(w, pos, px, eq, cfg, cash=acct.get("cash"))
    blocked, warn, orders = checks(cfg, orders, w, eq, float(acct.get("peak_equity", eq)), p, s, now)
    late = len(pd.bdate_range(s, _now(now).normalize())) - 2
    if late > 2:
        warn.append(f"判定日 {s.date()} から約{late}営業日たっています（過去の検証は判定日の次の営業日の寄付で組み直す前提）")
    out.update(orders=[o.to_dict() for o in orders], blocked=blocked, warnings=warn)
    if blocked or not execute:
        if not execute:
            out["note"] = "表示だけ（--execute で注文を出す）"
        return out
    out["submitted"] = br.submit(orders, str(s.date()))
    st[br.name] = str(s.date())
    _state_path(cfg).write_text(json.dumps(st))
    return out


def status(cfg: dict, broker: str | None = None) -> dict:
    br = B.make(cfg, broker)
    out = {"broker": br.name, "live": br.live, "account": br.account(), "positions": br.positions()}
    if br.name == "paper":
        out.update(pending=br.s["pending"], fills=br.s["fills"][-20:], history=br.s["history"][-12:])
    return out


def replay(cfg: dict, start: str, end: str | None = None, strategy: str | None = None, data=None) -> dict:
    """過去をたどる自動売買の試し運転: 毎月の判定日の引けの後に「その日までの株価」だけで rebalance --execute を実行し、
    次の月の実行で手元のペーパー口座が寄付の値段で約定する。過去の検証（backtest）と同じ結果になるかを確かめる。
    口座は cfg["data"]["dir"] / "replay" に作る（本物のペーパー口座には触らない）。"""
    import copy
    from .broker.paper import PaperBroker
    p, mem = data if data is not None else load(cfg)
    c = copy.deepcopy(cfg)
    c["data"]["dir"] = cfg["data"]["dir"] / "replay"
    c["data"]["dir"].mkdir(parents=True, exist_ok=True)
    for f in ("paper_account.json", "trader_state.json"):
        (c["data"]["dir"] / f).unlink(missing_ok=True)
    br = PaperBroker(c["data"]["dir"] / "paper_account.json", c)
    cal = D.calendar(p)
    me = D.month_ends(cal)
    me = me[(me >= pd.Timestamp(start)) & (me <= pd.Timestamp(end or cal[-1]))]
    me = pd.DatetimeIndex([d for d in me if d.date() == last_trading_day(d.year, d.month)])  # 終わっていない月は除く
    log = []
    for s in me:
        res = rebalance(c, strategy, execute=True, data=(p.upto(s), mem), now=s + pd.Timedelta(hours=20), br=br)
        log.append({"signal_date": str(s.date()), "equity": res["equity"], "orders": len(res.get("orders", [])), "targets": res["targets"],
                    "blocked": res.get("blocked", []), "fills": len(res.get("fills", []))})
    br.settle(p)
    final = br.mark(p)
    return {"months": len(me), "final_equity": final["equity"], "log": log, "history": br.s["history"], "fills": br.s["fills"]}
