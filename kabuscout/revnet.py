"""比較用の口座その3「修正好反応」（ジャーナル 84・85。本番の判断には使わない）。

網: 業績予想の修正・予想と実績の差異（訂正を除く、tdnet.REVISION）が終値に入った日（events.known_dates）から5営業日以内で、
その日の騰落率が日経平均比 +5%以上・出来高が直前20日平均の1.5倍以上。売買ルールは決算好反応と同じ。
条件は研究（research/event_nets.py）と同じ数字に固定する（研究ループでは調整しない）。

本番を変えないための仕組み:
- この網は strategies.STRATEGIES に入れない。シグナル表（pool）には rev_* の列と、この網だけに当たる行も入るが、
  本番の候補表（strategies.apply_flags）はどの本番の網にも当たらない行を落とすので、本番の候補・AI は変わらない
- 比較用のAIは、本番と同じ作り方（全期間・同じ種の数・新しいデータほど重く）で、印 s_rev_drift を1つ足して別に学習する
  （毎日の実行の後の shadow ジョブ。本番と同じ間隔で学習し直す）
- 業績修正の題名の過去分（2010年〜）がそろうまで（tdnet.backfill_revisions）は学習も売買もしない
"""
from __future__ import annotations

import copy
import json
import logging
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import requests
import time

from . import events, tdnet
from .strategies import Strategy, apply_flags

log = logging.getLogger(__name__)
KEY = "rev_drift"
FLAG = f"s_{KEY}"
REV_FEATURES = ["rev_age", "rev_react", "rev_vol"]
PARAMS = dict(age=5, react=0.05, vol=1.5)   # 研究と同じ（決算好反応の既定値と同じ数字）
ACCT = "paper_rev"
MODEL_FILE = "revnet.joblib"
RECENT_FILE = "tdnet_rev_recent.parquet"   # スクリーニングの前に取った、前の営業日〜今日の業績修正の題名（当日の発表を拾うため）


def detect(i, p=PARAMS):
    return (i["rev_age"] >= 0) & (i["rev_age"] <= p["age"]) & (i["rev_react"] >= p["react"]) & (i["rev_vol"] >= p["vol"])


STRATEGY = Strategy(KEY, "修正好反応", "業績予想の修正に市場が好反応（市場比+5%以上・大商い）した銘柄の、発表後数日以内", detect)


# ---------------------------------------------------------------- 題名
def rows(cfg: dict, td: pd.DataFrame | None) -> pd.DataFrame:
    """業績修正の題名（保存済みの適時開示 ＋ スクリーニングの前に取った直近の分）。"""
    parts = []
    if td is not None and len(td):
        parts.append(td.loc[td["events"].str.contains("revision"), ["pubdate", "code"]])
    p = cfg["data"]["db_path"].parent / RECENT_FILE
    if p.exists():
        try:
            parts.append(pd.read_parquet(p)[["pubdate", "code"]])
        except Exception as e:
            log.warning("直近の業績修正の題名の読み込みに失敗: %s", e)
    if not parts:
        return pd.DataFrame(columns=["pubdate", "code"])
    return pd.concat(parts, ignore_index=True).drop_duplicates().reset_index(drop=True)


def fetch_recent(cfg: dict, last: pd.Timestamp, today: pd.Timestamp | None = None) -> int:
    """前の営業日〜今日の一覧を取り、業績修正の題名だけを別のファイルに置く（本番の適時開示の表には書かない。
    本番の取得は通知の後なので、前の営業日の夕方以降と今日の場中の発表がスクリーニングの時点でまだ無い）。取った件数を返す。"""
    today = (today or pd.Timestamp.today()).normalize()
    out = []
    for d in pd.date_range(min(last.normalize(), today) - pd.Timedelta(days=1), today):
        try:
            r = requests.get(tdnet.API.format(d=d.strftime("%Y%m%d")), headers=tdnet.UA, timeout=30)
            r.raise_for_status()
            for it in r.json().get("items", []):
                t = it.get("Tdnet", it)
                if "revision" in tdnet.classify(t["title"]):
                    out.append(dict(pubdate=pd.Timestamp(t["pubdate"]), code=str(t["company_code"])[:4]))
        except Exception as e:
            log.warning("  直近の業績修正の取得失敗 %s: %s", d.date(), e)
        time.sleep(1.0)
    pd.DataFrame(out, columns=["pubdate", "code"]).to_parquet(cfg["data"]["db_path"].parent / RECENT_FILE)
    return len(out)


# ---------------------------------------------------------------- 特徴量・候補
def wide_features(rev: pd.DataFrame | None, close: pd.DataFrame, volume: pd.DataFrame, mkt: pd.Series, max_age: int = 60) -> dict:
    """修正の開示が終値に入った最初の日 k での反応。k より先の値は使わない（研究の rev_wide と同じ式）。
    rev_react = k の日の騰落率 − 日経平均、rev_vol = k の出来高 ÷ それ以前20日の平均、rev_age = k からの営業日数。"""
    idx, cols = close.index, close.columns
    T, N = len(idx), len(cols)
    out = {f: pd.DataFrame(np.nan, index=idx, columns=cols, dtype="float32") for f in REV_FEATURES}
    if rev is None or rev.empty:
        return out
    rev = rev[rev["code"].isin(set(cols)) & (rev["pubdate"] >= idx[0] - pd.Timedelta(days=100)) & (rev["pubdate"] <= idx[-1] + pd.Timedelta(days=1))]
    if rev.empty:
        return out
    cix = {k: i for i, k in enumerate(cols)}
    k = events.known_dates(rev["pubdate"], idx)
    j = rev["code"].map(cix).to_numpy()
    ok = (k >= 1) & (k < T)
    k, j = k[ok], j[ok]
    c, v, m = close.to_numpy("float64"), volume.to_numpy("float64"), mkt.to_numpy("float64")
    vavg = pd.DataFrame(v).rolling(20, min_periods=10).mean().shift(1).to_numpy()
    react = np.full((T, N), np.nan, dtype="float32")
    vol, known = react.copy(), react.copy()
    react[k, j] = c[k, j] / c[k - 1, j] - 1 - (m[k] / m[k - 1] - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        vol[k, j] = np.where(vavg[k, j] > 0, v[k, j] / vavg[k, j], np.nan)
    known[k, j] = k
    f = lambda a: pd.DataFrame(a, index=idx, columns=cols).ffill(limit=max_age)
    kn = f(known)
    out["rev_react"], out["rev_vol"] = f(react).astype("float32"), f(vol).astype("float32")
    out["rev_age"] = (pd.DataFrame(np.arange(T)[:, None].repeat(N, 1), index=idx, columns=cols) - kn).where(kn.notna()).astype("float32")
    return out


def pool_mask(ind: dict) -> pd.DataFrame:
    """シグナル表に残す行（この網に当たる行）。"""
    return detect(ind).fillna(False).astype(bool)


def dataset(pl: pd.DataFrame, params: dict) -> pd.DataFrame:
    """比較用の候補表: 本番の網の印 ＋ s_rev_drift。どれかに当たる行だけ（本番の候補 ＋ この網だけに当たる行）。"""
    hit = pd.Series(detect(pl).fillna(False).to_numpy().astype(bool), index=pl.index)
    prod = apply_flags(pl, params)                       # 本番と同じ候補表（行の番号はシグナル表のまま）
    extra = pl[hit & ~pl.index.isin(prod.index)].copy()  # この網だけに当たる行
    for c in prod.columns:
        if c not in extra:
            extra[c] = "" if c == "strategies" else np.int8(0)
    d = pd.concat([prod, extra[prod.columns]]).sort_index()
    d[FLAG] = hit.loc[d.index].astype("int8")
    d["n_strat"] = d["n_strat"] + d[FLAG]
    d["strategies"] = np.where(d[FLAG] == 1, np.where(d["strategies"] == "", KEY, d["strategies"] + "," + KEY), d["strategies"])
    return d.reset_index(drop=True)


def cfg_for(cfg: dict) -> dict:
    """比較用の設定: 売買ルールは決算好反応と同じ（無ければ共通ルール）・AI に印を1つ足す。"""
    c = copy.deepcopy(cfg)
    per = dict(c["labeling"].get("per_strategy") or {})
    if "earnings_drift" in per:
        per[KEY] = per["earnings_drift"]
    c["labeling"]["per_strategy"] = per
    c["learning"]["extra_features"] = list(c["learning"].get("extra_features") or []) + [FLAG]
    return c


# ---------------------------------------------------------------- 比較用のAI
def _path(cfg: dict):
    return cfg["data"]["model_dir"] / MODEL_FILE


def load(cfg: dict) -> dict | None:
    try:
        return joblib.load(_path(cfg)) if _path(cfg).exists() else None
    except Exception as e:
        log.warning("修正好反応の比較用AIの読み込みに失敗: %s", e)
        return None


def _spec(cfg: dict) -> str:
    c = cfg_for(cfg)
    return json.dumps(dict(params=PARAMS, learning={k: c["learning"].get(k) for k in ("model_params", "extra_features", "specialists", "seeds")},
                           plan=c["labeling"]["per_strategy"].get(KEY)), sort_keys=True, default=str)


def due(cfg: dict, dates: pd.Series) -> bool:
    """学習し直しが要るか（過去の題名がそろっていて、無い・定義が変わった・本番と同じ間隔を過ぎた）。"""
    if not tdnet.revisions_ready(cfg):
        return False
    b = load(cfg)
    return b is None or b.get("spec") != _spec(cfg) or int((dates > pd.Timestamp(b["data_end"])).sum()) >= cfg["learning"]["retrain_every_days"]


def train(cfg: dict, pl: pd.DataFrame, params: dict, data_end: str) -> dict | None:
    """本番の最終モデルと同じ作り方で、印 s_rev_drift を足したAIを学習して保存する。"""
    from . import learner
    c = cfg_for(cfg)
    ds = dataset(pl, params)
    d = learner.with_plan_outcomes(ds[ds["complete"]], c)
    d = d[d["complete_plan"]].sort_values("date").reset_index(drop=True)
    log.info("修正好反応の比較用AIを学習中（%d件、うち修正好反応 %d件）", len(d), int(d[FLAG].sum()))
    models = learner.fit_models(d, c, seeds=int(c["learning"].get("seeds", 1)))
    b = dict(models=models, features=learner.model_features(c), spec=_spec(cfg), data_end=data_end, metrics={}, alpha=1.0,
             version=f"rev-{datetime.now().strftime('%Y%m%d-%H%M')}", n_rev=int(d[FLAG].sum()))
    joblib.dump(b, _path(cfg))
    return b


def picks(cfg: dict, pl: pd.DataFrame, params: dict, last: pd.Timestamp, exclude: set, bundle: dict, prod_bundle: dict | None,
          weights: dict) -> pd.DataFrame:
    """その日の比較用の候補に点を付けて並べる（本番と同じ並べ方。AIとルールの配分は本番のモデルと同じにする）。"""
    from . import learner
    today = dataset(pl[pl["date"] == last], params)
    today = today[~today["code"].isin(exclude)]
    b = dict(bundle)
    if prod_bundle is not None:  # 本番がルールと半々にしている時期は、こちらも同じ配分にする（差を網の違いだけにする）
        m = prod_bundle.get("metrics", {})
        b["alpha"] = learner.ai_weight(m) if "daily_top_ret" in m else prod_bundle.get("alpha", 1.0)
    return learner.score(today, b, weights)
