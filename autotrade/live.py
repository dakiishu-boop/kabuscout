"""毎日の自動運用（今は仮想の口座＝ペーパートレード。証券口座ができたら brokers/ の実口座に差し替える）。

kabuscout の毎日の実行（株価の更新）の後に動かす:
  1) 前回の続きの営業日から今日まで、1日ずつ: 分割の反映 → 前日の注文を寄付で約定 → 終値で評価
  2) 今日の引けのデータでAIの点数を付け、strategy.decide で翌営業日の寄付の注文を作る（バックテストと同じ関数）
  3) 安全装置（risk.RiskGuard）を通った注文だけを出す
  4) 状態（data/autotrade/state.json）・毎日の記録・レポートを保存し、設定があれば Discord に送る

AI は weekly に学び直す（train。全期間のデータで最終のモデルを作る）。モデルが無い・古いときは daily の中で学ぶ。
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import data as D
from . import features as F
from . import model as M
from .broker import PaperBroker
from .risk import RiskGuard, RiskLimits
from .sim import _Row
from .strategy import Params, decide

log = logging.getLogger(__name__)
JST = timezone(timedelta(hours=9))
CONFIG = Path(__file__).with_name("config.yaml")


def load_settings(path: Path = CONFIG) -> dict:
    return yaml.safe_load(Path(path).read_text())


def state_dir(cfg: dict) -> Path:
    d = Path(cfg["data"]["db_path"]).parent / "autotrade"
    d.mkdir(parents=True, exist_ok=True)
    return d


def params_of(s: dict) -> Params:
    return Params(**s["strategy"])


# ---------------------------------------------------------------- model
def train(cfg: dict, s: dict) -> dict:
    """全期間のデータで最終のモデルを作って保存する。"""
    import joblib
    hs = s["model"]["horizons"]
    ds, _, _ = D.build(cfg, horizons=hs)
    feats = F.FEATURES
    bundle = dict(features=feats, horizons=hs, models={}, trained=datetime.now(JST).isoformat(timespec="seconds"),
                  data_through=str(ds["date"].max().date()))
    for H in hs:
        tr = ds[ds["elig"]]
        bundle["models"][H] = M.fit(tr, feats, f"y{H}", s["model"].get("params"), seeds=tuple(range(s["model"].get("seeds", 1))),
                                    sample_every=s["model"].get("sample_every", 3))
        log.info("モデル H=%d を学習（%d行）", H, int(tr[f"y{H}"].notna().sum()))
    joblib.dump(bundle, state_dir(cfg) / "model.joblib")
    return bundle


def load_model(cfg: dict, s: dict, max_age_days: int = 8):
    import joblib
    p = state_dir(cfg) / "model.joblib"
    if p.exists():
        b = joblib.load(p)
        age = datetime.now(JST) - datetime.fromisoformat(b["trained"])
        if age.days < max_age_days and b["horizons"] == s["model"]["horizons"]:
            return b
        log.info("モデルが古い（%d日前）ので学び直します", age.days)
    return train(cfg, s)


def score(bundle: dict, rows: pd.DataFrame) -> pd.Series:
    """各 H のモデルの予測を、その日の中での順位にして平均する（研究の s_mix と同じ）。"""
    parts = []
    for H, models in bundle["models"].items():
        p = pd.Series(M.predict(models, rows, bundle["features"]), index=rows.index)
        parts.append(p.groupby(rows["date"]).rank(pct=True))
    return pd.concat(parts, axis=1).mean(axis=1)


# ---------------------------------------------------------------- daily
def _busdays_between(a: pd.Timestamp, b: pd.Timestamp) -> int:
    return int(np.busday_count(a.date(), b.date()))


def daily(cfg: dict, s: dict, notify: bool = True, today: pd.Timestamp | None = None) -> dict:
    p = params_of(s)
    sd = state_dir(cfg)
    st_path = sd / "state.json"
    from kabuscout.data import PriceStore
    proxy = cfg["universe"]["market_proxy"]
    end = PriceStore(cfg["data"]["price_dir"]).load(fields=["close"], codes=[proxy])["close"][proxy].dropna().index[-1]
    start = (end - pd.Timedelta(days=900)).strftime("%Y-%m-%d")  # 特徴量に1年強の履歴が要る（データの最終日から数える）
    ds, market, masks = D.build(cfg, horizons=(), start=start)
    cal = market.cal
    cix = {c: j for j, c in enumerate(market.codes)}
    if st_path.exists():
        st = json.loads(st_path.read_text())
        b = PaperBroker.from_dict(st["broker"])
        guard = RiskGuard(RiskLimits(**s.get("risk", {})), **st.get("guard", {}))
    else:
        st = dict(last_date=None, since_reb=10 ** 6, equity=[], peak=s["account"]["capital"], log=[])
        b = PaperBroker(s["account"]["capital"], p.cost, p.lot)
        guard = RiskGuard(RiskLimits(**s.get("risk", {})))
        log.info("自動売買の口座を新しく作りました（資金 %s円）", f"{s['account']['capital']:,}")
    last = pd.Timestamp(st["last_date"]) if st["last_date"] else cal[-2]
    todo = [i for i, d in enumerate(cal) if d > last]
    events = []
    if not todo:
        log.info("新しい営業日のデータがありません（最新 %s）", cal[-1].date())
    for i in todo:
        d = cal[i].strftime("%Y-%m-%d")
        sp = market.splits[i]
        ratios = {c: float(sp[cix[c]]) for c in {*b.positions(), *(o.code for o in b.open_orders())}
                  if c in cix and sp[cix[c]] == sp[cix[c]] and sp[cix[c]] not in (0, 1)}
        if ratios:
            b.apply_splits(ratios)
            events.append(f"{d} 分割・併合を反映: {ratios}")
        for f in b.execute_open(d, _Row(market.open[i], cix)):
            events.append(f"{d} 約定 {f.side} {f.code} {f.shares}株 @{f.raw_px:,.1f}" +
                          (f" 損益 {f.pnl:+,.0f}円（{f.net:+.1%}）{f.reason}" if f.side == "sell" else f" {f.reason}"))
        eq = b.mark(_Row(market.close[i], cix))
        st["equity"].append([d, round(eq)])
        st["peak"] = max(st["peak"], eq)
        st["since_reb"] += 1
        st["last_date"] = d
    # 翌営業日の注文（最新日の引けのデータで判断）
    orders, blocked = [], {}
    i = len(cal) - 1
    crow = _Row(market.close[i], cix)
    eq = b.equity(crow)
    if todo and todo[-1] == i:
        reb = st["since_reb"] >= p.rebalance_every
        rows = ds[ds["date"] == cal[i]]
        if reb and len(rows):
            bundle = load_model(cfg, s)
            sc = score(bundle, rows)
            scores = dict(zip(rows["code"], sc))
            elig = set(rows.loc[rows["elig"], "code"])
        else:
            scores, elig, reb = {c: 0.0 for c in b.positions()}, set(), False
        orders = decide(b, scores, elig, crow, reb, bool(market.weak[i]), p)
        now = today or pd.Timestamp(datetime.now(JST).date())
        age = _busdays_between(cal[i], now)
        orders, blocked = guard.check(orders, eq, st["peak"], {c: crow.get(c) for c in {o.code for o in orders}}, age)
        b.submit(orders)
        if reb and age <= guard.limits.data_max_age_days:
            st["since_reb"] = 0  # 古いデータで止めた日は入れ替えたことにしない（次の実行でもう一度判断する）
    st["broker"], st["guard"] = b.to_dict(), guard.to_dict()
    st["log"] = (st.get("log", []) + events)[-500:]
    tmp = st_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, ensure_ascii=False, indent=1))
    tmp.replace(st_path)
    rep = report(st, b, crow, orders, blocked, events, cal[i], s)
    (sd / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    (sd / "report.md").write_text(rep["text"])
    print(rep["text"])
    if notify:
        send(cfg, rep)
    return rep


def report(st, b: PaperBroker, crow, orders, blocked, events, date, s) -> dict:
    eq = b.equity(crow)
    cap = s["account"]["capital"]
    names = _names(s)
    lines = [f"# 自動売買（ペーパートレード） {date.date()} 引け後",
             f"資産 {eq:,.0f}円（{eq / cap - 1:+.1%}）・現金 {b.cash():,.0f}円・最高値から {eq / st['peak'] - 1:+.1%}"]
    if st["guard"].get("halted"):
        lines.append(f"⚠️ 新規の買いを停止中: {st['guard']['halt_reason']}（data/autotrade/state.json の guard.halted を false にすると再開）")
    lines.append("\n## 保有")
    for p in sorted(b.positions().values(), key=lambda x: x.code):
        c = crow.get(p.code) or p.entry_px
        lines.append(f"- {p.code} {names.get(p.code, '')} {p.shares}株 取得 {p.entry_px:,.1f} → {c:,.1f}（{c / p.entry_px - 1:+.1%}）{p.entry_date}〜")
    if not b.positions():
        lines.append("- なし")
    lines.append("\n## 翌営業日の寄付の成行注文")
    for o in orders:
        lines.append(f"- {'買い' if o.side == 'buy' else '売り'} {o.code} {names.get(o.code, '')} {o.shares}株（目安 {o.ref_px:,.1f}円・{o.reason}）")
    if not orders:
        lines.append("- なし")
    for code, why in blocked.items():
        lines.append(f"- 安全装置で止めた注文: {code} {why}")
    if events:
        lines.append("\n## 今回の処理")
        lines += [f"- {e}" for e in events[-20:]]
    lines.append("\n※ 仮想の口座の記録です。投資助言ではありません。")
    return dict(date=str(date.date()), equity=eq, cash=b.cash(), peak=st["peak"], halted=st["guard"].get("halted"),
                positions=[dict(code=p.code, shares=p.shares, entry=p.entry_px) for p in b.positions().values()],
                orders=[dict(code=o.code, side=o.side, shares=o.shares, ref=o.ref_px, reason=o.reason) for o in orders],
                blocked=blocked, text="\n".join(lines))


def _names(s) -> dict:
    try:
        import sqlite3
        from kabuscout.config import load_config
        cfg = load_config(Path("config.yaml"))
        with sqlite3.connect(str(cfg["data"]["db_path"])) as con:
            return dict(con.execute("SELECT code, name FROM universe").fetchall())
    except Exception:
        return {}


def send(cfg: dict, rep: dict) -> None:
    """kabuscout と同じ Discord / ntfy の設定があれば送る（無ければ何もしない）。失敗しても止めない。"""
    try:
        from kabuscout import notify
        n = cfg.get("notify", {})
        text = rep["text"][:3900]
        if n.get("discord_webhook"):
            notify.discord(n["discord_webhook"], {"content": text[:1990]})
        if n.get("ntfy_topic"):
            notify.ntfy(n.get("ntfy_server", "https://ntfy.sh"), n["ntfy_topic"], "自動売買（ペーパー）", text)
    except Exception as e:
        log.warning("通知に失敗（運用は続けます）: %s", e)
