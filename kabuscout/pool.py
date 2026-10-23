"""シグナル表（プール）。

25年分の全銘柄で「最も緩い条件」のシグナルを拾い、その日の特徴量と、その後の結果（ラベル）を
1行=1銘柄日で保存する。戦略の条件をどう変えても、その条件に当てはまる行はこの表の中にあるので、
研究ループは指標を計算し直さずに、何百通りもの条件を一瞬で試せる。

25年分の指標を一度にメモリに載せると重いので、3年ずつ区切って計算する。
毎日は直近だけ計算し直して差し替える（結果が確定した行が少しずつ増えていく）。
"""
from __future__ import annotations

import hashlib
import json
import logging

import numpy as np
import pandas as pd

from . import data, dividends, earnings, edinet, events, forecast, jquants, learner, macro, regimes, tdnet
from .indicators import CANDIDATE_FEATURES, FEATURES, RAW_FIELDS, compute
from .strategies import detect_all, loose_params

log = logging.getLogger(__name__)
POOL_VERSION = 12
FIELDS = list(dict.fromkeys(FEATURES + RAW_FIELDS + earnings.EARN_FEATURES + events.EVENT_FEATURES
                            + dividends.DIV_FEATURES + edinet.FIN_FEATURES + forecast.FC_FEATURES + macro.MACRO_FEATURES + jquants.JQ_FEATURES + CANDIDATE_FEATURES))
WARMUP_DAYS = 600   # 250日線などの計算に必要な助走（暦日）
RECENT_DAYS = 200    # 毎日計算し直す直近期間（暦日）


def _lookahead(cfg: dict) -> int:
    """ラベル付けに必要な先の期間（暦日）。最長保有日数から決める。"""
    return int(cfg["labeling"]["horizon_days"] * 1.6) + 20


def _earn_stamp(cfg: dict) -> str:
    """決算データが増えたらシグナル表を作り直すための目印（週単位）。"""
    ev, td, fin, jq = earnings.load(cfg), tdnet.load(cfg), edinet.load(cfg), jquants.load_fins(cfg)
    return (f"{0 if ev is None else ev['code'].nunique() // 50}-{0 if td is None else len(td) // 20000}"
            f"-{0 if fin is None else len(fin) // 3000}-{0 if jq is None else len(jq) // 20000}")


def _signature(cfg: dict) -> str:
    key = json.dumps(dict(v=POOL_VERSION, loose=loose_params(),
                          lab={**{k: cfg["labeling"][k] for k in ("horizon_days", "tp_atr", "sl_atr")},
                               "stop": cfg["labeling"].get("stop_mode", "intraday")},
                          start=cfg["data"]["history_start"], uni=cfg["universe"], earn=_earn_stamp(cfg)),
                     sort_keys=True, default=str)
    return hashlib.sha1(key.encode()).hexdigest()[:12]


def _meta_path(cfg):
    return cfg["data"]["pool_path"].with_suffix(".json")


def build_window(store, cfg: dict, uni: pd.DataFrame, start: str | None, end: str | None,
                 keep_from: str | None = None, keep_to: str | None = None):
    """期間の指標を計算し、緩い条件のシグナル行（ラベル付き）を返す。"""
    proxy = cfg["universe"]["market_proxy"]
    codes = uni["code"].tolist()
    panel = data.load_panel(store, start=start, end=end, codes=codes + [proxy])
    if panel["close"].empty:
        return None, pd.DataFrame()
    # メモリ節約: この期間に一度も流動性基準を満たさなかった銘柄は除外
    tv = (panel["close"] * panel["volume"]).rolling(20, min_periods=10).mean().max()
    keep = [c for c in panel["close"].columns if tv.get(c, 0) >= cfg["universe"]["min_turnover_jpy"] or c == proxy]
    panel = {k: v[keep].astype("float32") for k, v in panel.items()}
    if hasattr(store, "split_factor"):
        panel["split_factor"] = store.split_factor(panel["close"].index, panel["close"].columns)

    ind = compute(panel, proxy, uni.set_index("code")["sector33"])
    td = tdnet.load(cfg)
    ev = earnings.combine(earnings.load(cfg), events.earnings_dates(td))
    ind.update(earnings.wide_features(ev, ind["close"], ind["volume"], panel["close"][proxy].ffill()))
    ind.update(events.wide_features(td, ind["close"]))
    ind.update(dividends.wide_features(store.load_dividends(codes=list(ind["close"].columns)), ind["close"]))
    px = ind.get("px_actual", ind["close"])  # 1株あたりの財務の値（報告書の値）は、分割前の実際の株価と比べる
    ind.update(edinet.wide_features(edinet.load(cfg), px))
    ind.update(forecast.wide_features(forecast.predictions(cfg, store, uni), ind["close"]))  # 自前の利益の予想（調整後の株価で時価総額）
    ind.update(macro.wide_features(macro.load(cfg), ind["close"]))
    ind.update(jquants.wide_features(jquants.load_fins(cfg), px, cfg.get("jquants", {}).get("lag_days", 84)))
    ind = {k: v.astype("float32") for k, v in ind.items()}
    in_uni = pd.Series([c in set(codes) for c in ind["close"].columns], index=ind["close"].columns)
    mask = ((ind["turnover"] >= cfg["universe"]["min_turnover_jpy"])
            & (ind["px_actual"] >= cfg["universe"]["min_price"])) & in_uni
    sigs = detect_all(ind, mask, loose_params())
    any_sig = None
    for m in sigs.values():
        any_sig = m if any_sig is None else (any_sig | m)
    arr = any_sig.to_numpy().copy()
    idx = ind["close"].index
    if keep_from:
        arr[idx < pd.Timestamp(keep_from)] = False
    if keep_to:
        arr[idx >= pd.Timestamp(keep_to)] = False
    ti, ci = np.nonzero(arr)
    rows = {"date": idx[ti], "code": ind["close"].columns[ci], "_ti": ti, "_ci": ci}
    for f in FIELDS:
        rows[f] = ind[f].to_numpy()[ti, ci]
    df = learner.label(ind, pd.DataFrame(rows), cfg)
    df = df.drop(columns=["_ti", "_ci"])
    df["crisis"] = regimes.crisis_flag(df["date"], df["stress"] > 0.5).astype("int8")
    for c in df.columns:
        if df[c].dtype == "float64":
            df[c] = df[c].astype("float32")
    return ind, df


def build_full(store, cfg: dict, uni: pd.DataFrame) -> pd.DataFrame:
    start = pd.Timestamp(cfg["data"]["history_start"])
    end = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
    parts, y = [], start
    while y < end:
        y2 = min(y + pd.DateOffset(years=3), end)
        log.info("  シグナル表を作成中: %s〜%s", y.date(), y2.date())
        _, df = build_window(store, cfg, uni, (y - pd.Timedelta(days=WARMUP_DAYS)).strftime("%Y-%m-%d"),
                             (y2 + pd.Timedelta(days=_lookahead(cfg))).strftime("%Y-%m-%d"),
                             keep_from=y.strftime("%Y-%m-%d"), keep_to=y2.strftime("%Y-%m-%d"))
        parts.append(df)
        y = y2
    pool = pd.concat(parts, ignore_index=True)
    save(pool, cfg)
    return pool


def save(pool: pd.DataFrame, cfg: dict) -> None:
    pool = pool.sort_values(["date", "code"]).reset_index(drop=True)
    pool.to_parquet(cfg["data"]["pool_path"], compression="zstd")
    _meta_path(cfg).write_text(json.dumps(dict(signature=_signature(cfg), rows=len(pool))))


def load(cfg: dict) -> pd.DataFrame | None:
    p = cfg["data"]["pool_path"]
    if not p.exists():
        return None
    meta = json.loads(_meta_path(cfg).read_text()) if _meta_path(cfg).exists() else {}
    if meta.get("signature") != _signature(cfg):
        log.info("シグナル表の定義が変わったので作り直します")
        return None
    return pd.read_parquet(p)


def refresh(store, cfg: dict, uni: pd.DataFrame, cutoff: pd.Timestamp):
    """直近を計算し直してシグナル表を更新。(直近の指標, シグナル表全体) を返す。"""
    pool = load(cfg)
    if pool is None:
        log.info("25年分のシグナル表を作成します（初回のみ・数分）")
        pool = build_full(store, cfg, uni)
    keep_from = (cutoff - pd.Timedelta(days=RECENT_DAYS)).strftime("%Y-%m-%d")
    ind, recent = build_window(store, cfg, uni, (cutoff - pd.Timedelta(days=RECENT_DAYS + WARMUP_DAYS)).strftime("%Y-%m-%d"),
                               cutoff.strftime("%Y-%m-%d"), keep_from=keep_from)
    pool = pd.concat([pool[pool["date"] < pd.Timestamp(keep_from)], recent], ignore_index=True)
    save(pool, cfg)
    return ind, pool.sort_values(["date", "code"]).reset_index(drop=True)
