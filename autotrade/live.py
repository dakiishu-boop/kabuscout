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
from .blend import Sleeve, index_levels
from .broker import PaperBroker
from .risk import RiskGuard, RiskLimits
from .sim import _Row
from .strategy import Params, decide

log = logging.getLogger(__name__)
JST = timezone(timedelta(hours=9))
CONFIG = Path(__file__).with_name("config.yaml")
# 口座の状態ファイルの版。2（2026-11-28）: 株価が揃っていない最新日を使わない。
# 1 の口座は、深夜の実行で株価が3/3694銘柄しか届いていない日のデータで判断していた（候補が1銘柄だけになった）ので、
# 一度だけ作り直す（元のファイルは state_*.v1.json として残す）
STATE_VERSION = 2


def load_settings(path: Path = CONFIG) -> dict:
    return yaml.safe_load(Path(path).read_text())


def state_dir(cfg: dict) -> Path:
    d = Path(cfg["data"]["db_path"]).parent / "autotrade"
    d.mkdir(parents=True, exist_ok=True)
    return d


def params_of(s: dict) -> Params:
    return Params(**s["strategy"])


# ---------------------------------------------------------------- model
def train(cfg: dict, s: dict, end: pd.Timestamp | None = None) -> dict:
    """全期間のデータで最終のモデルを作って保存する。end: この日までのデータだけを使う（株価が揃っていない最新日を除く）"""
    import joblib
    hs = s["model"]["horizons"]
    ds, _, _ = D.build(cfg, horizons=hs, end=end.strftime("%Y-%m-%d") if end is not None else None)
    feats = F.FEATURES
    bundle = dict(features=feats, horizons=hs, models={}, trained=datetime.now(JST).isoformat(timespec="seconds"),
                  data_through=str(ds["date"].max().date()))
    k = s["model"].get("sample_every", 3)
    for H in hs:
        tr = ds[ds["elig"]]
        # 学習に使う日を k 日おきに間引き、間引き方（0〜k−1日目から）を変えた k 個のモデルの平均にする（研究の s_ens と同じ）
        bundle["models"][H] = [m for off in range(k) for m in M.fit(tr, feats, f"y{H}", s["model"].get("params"),
                                                                      seeds=tuple(range(s["model"].get("seeds", 1))),
                                                                      sample_every=k, offset=off)]
        log.info("モデル H=%d を学習（%d行・%d個の平均）", H, int(tr[f"y{H}"].notna().sum()), len(bundle["models"][H]))
    joblib.dump(bundle, state_dir(cfg) / "model.joblib")
    return bundle


def load_model(cfg: dict, s: dict, max_age_days: int = 8, end: pd.Timestamp | None = None):
    import joblib
    p = state_dir(cfg) / "model.joblib"
    if p.exists():
        b = joblib.load(p)
        age = datetime.now(JST) - datetime.fromisoformat(b["trained"])
        if age.days < max_age_days and b["horizons"] == s["model"]["horizons"]:
            return b
        log.info("モデルが古い（%d日前）ので学び直します", age.days)
    return train(cfg, s, end)


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


def usable_end(cfg: dict) -> tuple[pd.Timestamp, str]:
    """使ってよいデータの最終日と、最新日を除いたときの注意書き。
    kabuscout の毎日の実行と同じ判定（kabuscout.pipeline._usable_date: 全銘柄の8割以上の株価が揃っている最新日）。
    GitHub の定時実行が深夜にずれると、Yahoo の最新日の足が数銘柄分しか無いことがある（2026-11-25 は 2/3694、11-27 は 3/3694）。"""
    from kabuscout.data import PriceStore
    from kabuscout.pipeline import _usable_date
    store = PriceStore(cfg["data"]["price_dir"])
    proxy = cfg["universe"]["market_proxy"]
    p_last = store.load(fields=["close"], codes=[proxy])["close"][proxy].dropna().index[-1]
    # 最新日は「どれかの銘柄に株価がある最後の日」（2026-11-27 は指数のETFに無く、3銘柄だけにあった）
    c = store.load(start=(p_last - pd.Timedelta(days=40)).strftime("%Y-%m-%d"), fields=["close"])["close"]
    cnt = c.notna().sum(axis=1)
    cnt = cnt[cnt > 0]
    last = cnt.index[-1]
    end = _usable_date(store, last)
    note = ""
    if end < last:
        note = (f"⚠️ {last.date()} の株価がまだ揃っていない（{int(cnt.iloc[-1])}/{int(cnt.max())}銘柄・Yahoo）ため、"
                f"{end.date()} までのデータで判断しました。{last.date()} の分は、揃ってから次の実行で処理します")
    return end, note


def daily(cfg: dict, s: dict, notify: bool = True, today: pd.Timestamp | None = None) -> dict:
    """本番の口座（state.json）と、比較用の口座（config の shadow_accounts。state_<名前>.json）を同じデータで1日進める。
    比較用の口座は売買の形だけが違い、AIの点数は共通。通知は本番の口座のレポートに比較用の口座の1行を足して送る。
    株価が揃っていない最新日は使わない（usable_end）。新しく処理した営業日が無い実行（同じ日の2回目など）は通知しない。"""
    end, note = usable_end(cfg)
    start = (end - pd.Timedelta(days=900)).strftime("%Y-%m-%d")  # 特徴量に1年強の履歴が要る（データの最終日から数える）
    ds, market, masks = D.build(cfg, horizons=(), start=start, end=end.strftime("%Y-%m-%d"))
    cache: dict = {"end": end}
    rep = _account(cfg, s, params_of(s), "state.json", "report", ds, market, cache, today)
    if note:
        head, _, rest = rep["text"].partition("\n")
        rep["text"] = f"{head}\n{note}\n{rest}"
    for name, diff in (s.get("shadow_accounts") or {}).items():
        try:
            diff = dict(diff)
            risk = diff.pop("risk", None)  # 比較用の口座ごとの安全装置（無ければ本番と同じ）
            s2 = {**s, "risk": {**s.get("risk", {}), **(risk or {})}}
            r = _account(cfg, s2, Params(**{**s["strategy"], **diff}), f"state_{name}.json", f"report_{name}", ds, market, cache, today)
            line = (f"比較用の口座「{name}」（{diff}）: 資産 {r['equity']:,.0f}円（{r['equity'] / s['account']['capital'] - 1:+.1%}）・"
                    f"保有 {len(r['positions'])}銘柄・翌朝の注文 {len(r['orders'])}件")
            rep["text"] += "\n" + line
            rep.setdefault("shadows", {})[name] = dict(equity=r["equity"], positions=r["positions"], orders=r["orders"])
        except Exception as e:  # 比較用の口座が失敗しても本番の口座は止めない
            log.warning("比較用の口座 %s の処理に失敗: %s", name, e)
    sd = state_dir(cfg)
    (sd / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    (sd / "report.md").write_text(rep["text"])
    print(rep["text"])
    if notify and rep.get("new_day"):
        send(cfg, rep)
    elif notify:
        log.info("新しく処理した営業日が無いので、通知は送りません")
    return rep


def _account(cfg: dict, s: dict, p: Params, state_name: str, report_name: str, ds, market, cache: dict,
             today: pd.Timestamp | None) -> dict:
    sd = state_dir(cfg)
    st_path = sd / state_name
    cal = market.cal
    cix = {c: j for j, c in enumerate(market.codes)}
    st = json.loads(st_path.read_text()) if st_path.exists() else None
    if st is not None and st.get("version", 1) < STATE_VERSION:
        old = st_path.with_name(st_path.stem + ".v1.json")
        st_path.replace(old)
        log.warning("口座 %s は前の版（株価が揃っていない日も使っていた）で作られたので作り直します（元のファイルは %s）",
                    state_name, old.name)
        st = None
    if st is not None:
        b = PaperBroker.from_dict(st["broker"])
        guard = RiskGuard(RiskLimits(**s.get("risk", {})), **st.get("guard", {}))
    else:
        st = dict(version=STATE_VERSION, last_date=None, since_reb=10 ** 6, equity=[], peak=s["account"]["capital"], log=[])
        b = PaperBroker(s["account"]["capital"], p.cost, p.lot)
        guard = RiskGuard(RiskLimits(**s.get("risk", {})))
        log.info("自動売買の口座 %s を新しく作りました（資金 %s円）", state_name, f"{s['account']['capital']:,}")
    last = pd.Timestamp(st["last_date"]) if st["last_date"] else cal[-2]
    todo = [i for i, d in enumerate(cal) if d > last]
    events = []
    # 指数の部分（AIと指数を組み合わせた口座だけ。autotrade/blend.py。過去の検証と同じ Sleeve を使う。毎日の運用では税はかけない）
    sleeve, lvl, index_note = None, None, ""
    if p.index_share > 0 or p.tax_ai > 0:  # 過去の検証（sim.run）と同じ条件で Sleeve を作る（税だけのときも）
        sleeve = Sleeve.from_dict(st["sleeve"]) if st.get("sleeve") else Sleeve(p.index_share, p.tax_ai)
        sleeve.share, sleeve.tax = p.index_share, p.tax_ai
    if p.index_share <= 0 and sleeve is not None:
        lvl = np.ones(len(cal))  # 指数を持たない口座は値段1（口数0）として税の計算だけに使う
    elif p.index_share > 0:
        lvl, src = _index_levels(cfg, cal, p.index_name, cache)
        if lvl is None:
            index_note = f"⚠️ 指数（{p.index_name}）のデータを取れませんでした。指数の部分は前回の値のまま"
        else:
            lag = (cal[-1] - pd.Timestamp(src[-1])).days if src[-1] == src[-1] else 999
            if lag > 5:  # 米国の終値は日本の翌営業日に入る。連休を見込んでも5日を超えたら古い
                index_note = f"⚠️ 指数（{p.index_name}）のデータが古い（{pd.Timestamp(src[-1]).date()} の値）。届けば口数から正しい値に追いつく"
    def idx_val(i):
        if sleeve is None:
            return 0.0
        if lvl is None or not lvl[i] == lvl[i]:
            return float(st.get("sleeve_value", 0.0))
        return sleeve.value(float(lvl[i]))
    if not todo:
        log.info("新しい営業日のデータがありません（最新 %s）", cal[-1].date())
    for i in todo:
        d = cal[i].strftime("%Y-%m-%d")
        if sleeve is not None and lvl is not None:  # 年の最初の営業日の寄付の前: 比率を戻す（同じ年に2回はしない）
            prev = _Row(market.close[i - 1], cix) if i > 0 else None
            n0 = len(sleeve.events)
            sleeve.new_year(cal[i].year, b, b.equity(prev) if prev is not None else b.cash(), float(lvl[i]), d)
            events += sleeve.events[n0:]
        sp = market.splits[i]
        ratios = {c: float(sp[cix[c]]) for c in {*b.positions(), *(o.code for o in b.open_orders())}
                  if c in cix and sp[cix[c]] == sp[cix[c]] and sp[cix[c]] not in (0, 1)}
        if ratios:
            b.apply_splits(ratios)
            events.append(f"{d} 分割・併合を反映: {ratios}")
        for f in b.execute_open(d, _Row(market.open[i], cix)):
            events.append(f"{d} 約定 {f.side} {f.code} {f.shares}株 @{f.raw_px:,.1f}" +
                          (f" 損益 {f.pnl:+,.0f}円（{f.net:+.1%}）{f.reason}" if f.side == "sell" else f" {f.reason}"))
        if market.div is not None and b.positions():  # 権利落ち日の配当（税引き後。過去の検証の dividends=True と同じ）
            got = b.add_dividends({c: float(market.div[i, cix[c]]) for c in b.positions() if c in cix})
            if got > 0:
                events.append(f"{d} 配当 {got:,.0f}円（税引き後）")
        lc = st.setdefault("last_close", {})
        for c in list(b.positions()):
            v = market.close[i, cix[c]] if c in cix else float("nan")
            if v == v:
                lc[c] = float(v)
            elif c in lc:  # 6営業日以上値段が無い（上場廃止など）: 最後の終値で現金にする（バックテストと同じ）
                f = b.force_sell(d, c, lc.pop(c), "値段の途切れ（上場廃止など）")
                events.append(f"{d} {c} を最後の終値 {f.raw_px:,.1f} で現金化（値段の途切れ）")
        eq = b.mark(_Row(market.close[i], cix)) + idx_val(i)
        st["equity"].append([d, round(eq)])
        st["peak"] = max(st["peak"], eq)
        st["since_reb"] += 1
        st["last_date"] = d
    # 翌営業日の注文（最新日の引けのデータで判断）
    orders, blocked = [], {}
    i = len(cal) - 1
    crow = _Row(market.close[i], cix)
    eq = b.equity(crow) + idx_val(i)
    if todo and todo[-1] == i:
        reb = st["since_reb"] >= p.rebalance_every
        rows = ds[ds["date"] == cal[i]]
        if reb and len(rows):
            if "scores" not in cache:  # AIの点数はすべての口座で共通（1日に1回だけ計算）
                bundle = load_model(cfg, s, end=cache.get("end"))
                cache["scores"] = dict(zip(rows["code"], score(bundle, rows)))
            scores = cache["scores"]
            elig = set(rows.loc[rows["elig"], "code"])
        else:
            scores, elig, reb = {c: 0.0 for c in b.positions()}, set(), False
        weak = bool(market.weak_ma.get(p.regime_ma, market.weak)[i]) if p.regime_ma else False
        danger = _danger(cfg)  # 監理・整理銘柄と上場廃止予定（kabuscout の毎日の実行が JPX から取って保存したもの）
        elig -= set(danger)
        orders = decide(b, scores, elig, crow, reb, weak, p,
                        _Row(market.turn20[i], cix) if market.turn20 is not None else None)
        have = {o.code for o in orders}
        from .broker import Order
        orders += [Order(c, "sell", ps.shares, crow.get(c) or ps.entry_px, danger[c]) for c, ps in b.positions().items()
                   if c in danger and c not in have]  # 持っている銘柄が上場廃止の恐れになったら翌朝売る
        now = today or pd.Timestamp(datetime.now(JST).date())
        age = _busdays_between(cal[i], now)
        orders, blocked = guard.check(orders, eq, st["peak"], {c: crow.get(c) for c in {o.code for o in orders}}, age)
        b.submit(orders)
        if reb and age <= guard.limits.data_max_age_days:
            st["since_reb"] = 0  # 古いデータで止めた日は入れ替えたことにしない（次の実行でもう一度判断する）
    else:
        orders = list(b.open_orders())  # 新しい営業日が無い: 前回に出した翌朝の注文がそのまま残っている
    st["broker"], st["guard"] = b.to_dict(), guard.to_dict()
    if sleeve is not None:
        st["sleeve"], st["sleeve_value"] = sleeve.to_dict(), idx_val(len(cal) - 1)
    st["log"] = (st.get("log", []) + events)[-500:]
    tmp = st_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, ensure_ascii=False, indent=1))
    tmp.replace(st_path)
    rep = report(st, b, crow, orders, blocked, events, cal[i], s,
                 index=(p.index_name if p.index_share > 0 else "", idx_val(len(cal) - 1), index_note) if sleeve is not None else None)
    rep["new_day"] = bool(todo)
    (sd / f"{report_name}.md").write_text(rep["text"])
    return rep


def report(st, b: PaperBroker, crow, orders, blocked, events, date, s, index=None) -> dict:
    eq = b.equity(crow) + (index[1] if index else 0.0)
    cap = s["account"]["capital"]
    names = _names(s)
    lines = [f"# 自動売買（ペーパートレード） {date.date()} 引け後",
             f"資産 {eq:,.0f}円（{eq / cap - 1:+.1%}）・現金 {b.cash():,.0f}円・最高値から {eq / st['peak'] - 1:+.1%}"]
    if index and index[0]:
        lines.append(f"うち指数（{index[0]}）{index[1]:,.0f}円・AIの部分 {eq - index[1]:,.0f}円")
        if index[2]:
            lines.append(index[2])
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


def _index_levels(cfg: dict, cal, name: str, cache: dict):
    """指数の値段（円）と元の米国の日付。kabuscout.core.load_indices（取れなければ前回の保存分）から。1日に1回だけ読む。"""
    if "ix" not in cache:
        try:
            from kabuscout import core
            cache["ix"] = core.load_indices(cfg)
        except Exception as e:
            log.warning("指数のデータを読めませんでした: %s", e)
            cache["ix"] = None
    if cache["ix"] is None:
        return None, None
    return index_levels(cache["ix"], cal, name)


def _danger(cfg: dict) -> dict[str, str]:
    p = Path(cfg["data"]["db_path"]).parent / "alerts.json"
    try:
        return json.loads(p.read_text())["codes"] if p.exists() else {}
    except Exception:
        return {}


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
