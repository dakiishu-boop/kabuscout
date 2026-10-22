"""生存者バイアスの測定（上場廃止になった銘柄を入れると、過去ペーパートレードの成績はどう変わるか）。

Yahoo からは今も上場している銘柄の株価しか取れないため、過去ペーパートレードは「生き残った銘柄」だけで計算している。
J-Quants（Free は過去約2年）なら、その期間に上場廃止になった銘柄の株価も取れるので、
1. 対象期間の初めに上場していて、終わりには一覧から消えた銘柄を探し、株価（分割調整済み）を取得する
2. 生き残った銘柄と合わせて同じ手順でシグナルを作り、同じAI・同じ口座シミュレーションで「入れた場合/入れない場合」を比べる
上場廃止後は最後の値段で据え置く（TOB・MBO なら買付価格で現金化、倒産なら最後の安値で売れたものとみなす）。
元データは再配布できないので、結果の数字だけを data/survivorship.json とログに出す。
"""
from __future__ import annotations

import json
import logging
import time

import numpy as np
import pandas as pd

from . import backtest, jquants, learner, pool

log = logging.getLogger(__name__)
MARKETS = ("プライム", "スタンダード", "グロース")


def _paths(cfg):
    d = jquants._dir(cfg)
    return d / "delisted_list.parquet", d / "delisted_bars.parquet"


def collect(cfg: dict, max_minutes: float = 90) -> dict:
    if not jquants._key():
        return dict(skipped="no key")
    lst_p, bars_p = _paths(cfg)
    cov = jquants.coverage(cfg)
    if cov is None:
        return dict(error="coverage")
    lo, hi = (x.strftime("%Y%m%d") for x in cov)
    if not lst_p.exists():
        _, a = jquants._get(cfg, "/equities/master", {"date": lo})
        _, b = jquants._get(cfg, "/equities/master", {"date": hi})
        a, b = pd.DataFrame(a), pd.DataFrame(b)
        a = a[a["MktNm"].astype(str).str.contains("|".join(MARKETS))]
        gone = a[~a["Code"].isin(set(b["Code"]))][["Code", "CoName", "S33Nm", "MktNm"]]
        gone.to_parquet(lst_p)
        log.info("上場廃止（%s〜%s に一覧から消えた銘柄）: %d社（期間の初めの上場 %d社）", lo, hi, len(gone), len(a))
    gone = pd.read_parquet(lst_p)
    have = set(pd.read_parquet(bars_p, columns=["Code"])["Code"]) if bars_p.exists() else set()
    t0, frames = time.time(), []
    for code in gone["Code"]:
        if code in have:
            continue
        if time.time() - t0 > max_minutes * 60:
            break
        st, rows = jquants._get(cfg, "/equities/bars/daily", {"code": code})
        if st != 200:
            break
        frames.append(pd.DataFrame(rows) if rows else pd.DataFrame({"Code": [code]}))
        if len(frames) >= 25:
            _save(bars_p, frames)
            frames = []
    if frames:
        _save(bars_p, frames)
    n = len(set(pd.read_parquet(bars_p, columns=["Code"])["Code"])) if bars_p.exists() else 0
    log.info("上場廃止銘柄の株価: %d / %d社 取得済み", n, len(gone))
    return dict(delisted=len(gone), fetched=n)


def _save(path, frames):
    df = pd.concat(frames, ignore_index=True).astype(str)
    if path.exists():
        df = pd.concat([pd.read_parquet(path), df], ignore_index=True)
    df.drop_duplicates(["Code", "Date"] if "Date" in df else None).to_parquet(path)


class _StoreWithDelisted:
    """生き残った銘柄の株価（手元の PriceStore）に、上場廃止銘柄の株価を継ぎ足して見せる。"""

    def __init__(self, store, extra: dict[str, pd.DataFrame]):
        self.store, self.extra = store, extra
        self.codes = set(extra["close"].columns)

    def load(self, start=None, end=None, codes=None, fields=None, clean=True):
        own = None if codes is None else [c for c in codes if c not in self.codes]
        p = self.store.load(start=start, end=end, codes=own, fields=fields, clean=clean)
        add = [c for c in (codes or self.codes) if c in self.codes]
        for k in p:
            if k in self.extra and add:
                p[k] = p[k].join(self.extra[k][add].reindex(p[k].index), how="left")
        return p

    def split_factor(self, index, codes):
        own = [c for c in codes if c not in self.codes]
        f = self.store.split_factor(index, own)
        return f.reindex(columns=list(codes)).fillna(1.0)  # 上場廃止銘柄は J-Quants の調整済み株価なので倍率1とみなす

    def load_dividends(self, start=None, end=None, codes=None):
        return self.store.load_dividends(start=start, end=end, codes=None if codes is None else [c for c in codes if c not in self.codes])


def _panel(bars: pd.DataFrame, index: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    b = bars.dropna(subset=["Date"]).copy()
    b["code"] = b["Code"].str[:4]
    b["Date"] = pd.to_datetime(b["Date"])
    out = {}
    for f, col in (("open", "AdjO"), ("high", "AdjH"), ("low", "AdjL"), ("close", "AdjC"), ("volume", "AdjVo")):
        w = b.assign(v=pd.to_numeric(b[col], errors="coerce")).pivot_table(index="Date", columns="code", values="v", aggfunc="last")
        out[f] = w.reindex(index)
    last = out["close"].ffill()  # 上場廃止後は最後の値段で据え置く（売買ルールの決済に使う）
    for f in ("open", "high", "low", "close"):
        out[f] = out[f].fillna(last.where(out["close"].isna()))
    out["volume"] = out["volume"].fillna(0.0)
    return out


def analyze(cfg: dict) -> dict:
    from .data import refresh_universe
    from .db import connect
    from .pipeline import dataset, store_of
    _, bars_p = _paths(cfg)
    lst_p = _paths(cfg)[0]
    if not bars_p.exists():
        return dict(error="no data")
    gone = pd.read_parquet(lst_p)
    bars = pd.read_parquet(bars_p)
    store = store_of(cfg)
    uni = refresh_universe(connect(cfg["data"]["db_path"]), cfg)
    idx = store.load(fields=["close"], codes=[cfg["universe"]["market_proxy"]])["close"].index
    start = pd.Timestamp(bars["Date"].dropna().min())
    extra = _panel(bars, idx[idx >= start - pd.Timedelta(days=pool.WARMUP_DAYS)])
    extra = {k: v.drop(columns=[c for c in v.columns if c in set(uni["code"])]) for k, v in extra.items()}  # コード再利用などで今も上場中のものは除く
    gone4 = gone.assign(code=gone["Code"].str[:4])
    uni2 = pd.concat([uni, pd.DataFrame({"code": list(extra["close"].columns),
                                         "sector33": gone4.set_index("code")["S33Nm"].reindex(list(extra["close"].columns)).to_numpy()})],
                     ignore_index=True)
    s2 = _StoreWithDelisted(store, extra)
    end = idx[-1]
    _, sig = pool.build_window(s2, cfg, uni2, (start - pd.Timedelta(days=pool.WARMUP_DAYS)).strftime("%Y-%m-%d"),
                               end.strftime("%Y-%m-%d"), keep_from=start.strftime("%Y-%m-%d"))
    sig["delisted"] = sig["code"].isin(set(extra["close"].columns))
    log.info("期間 %s〜: シグナル %d件（うち上場廃止銘柄 %d件・%d社）", start.date(), len(sig), int(sig["delisted"].sum()),
             sig.loc[sig["delisted"], "code"].nunique())

    # 学習は期間の前まで（いつもの25年分のシグナル表）、判定は期間中の両方の候補
    pl = pool.load(cfg)
    if pl is None:
        pl = pool.build_full(store, cfg, uni)
    ds = dataset(cfg, pl)
    tr = learner.with_plan_outcomes(ds[ds["complete"] & (ds["date"] < start - pd.Timedelta(days=90))], cfg)
    tr = tr[tr["complete_plan"]]
    models = learner.fit_models(tr, cfg)
    te = learner.with_plan_outcomes(dataset(cfg, sig.drop(columns=["delisted"])).assign(
        delisted=lambda x: x["code"].isin(set(extra["close"].columns))), cfg, store=s2)
    te = te[te["complete"] & te["complete_plan"]]
    te = te.assign(prob=learner.predict(models, te)[1], ret=te["ret_plan"], exit_days=te["exit_plan"])
    out = dict(start=str(start.date()), end=str(end.date()), delisted_companies=int(len(gone)),
               delisted_with_signals=int(te.loc[te["delisted"], "code"].nunique()),
               signals=dict(survivors=int((~te["delisted"]).sum()), delisted=int(te["delisted"].sum())),
               avg_ret=dict(survivors=float(te.loc[~te["delisted"], "ret"].mean()), delisted=float(te.loc[te["delisted"], "ret"].mean())))
    te = backtest.with_stops(te, cfg)
    cal = idx[idx >= te["date"].min()]
    for label, df in (("生き残りのみ", te[~te["delisted"]]), ("上場廃止も含む", te)):
        sim = backtest.simulate(df, cal, cfg, "ai", store=s2)
        st = backtest._stats(sim["equity"], sim["trades"])
        out[label] = dict(total=st["total"], cagr=st["cagr"], max_dd=st["max_dd"], trades=st.get("trades", 0))
        log.info("過去ペーパートレード（%s）: 期間合計 %+.1f%%・年率 %+.1f%%・最大下落 %.1f%%・取引 %d回",
                 label, st["total"] * 100, st["cagr"] * 100, st["max_dd"] * 100, st.get("trades", 0))
    log.info("1シグナルあたりの平均リターン: 生き残り %+.2f%% ／ 上場廃止 %+.2f%%",
             out["avg_ret"]["survivors"] * 100, out["avg_ret"]["delisted"] * 100)
    (cfg["data"]["db_path"].parent / "survivorship.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float))
    return out
