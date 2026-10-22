"""海外市場・為替・金利・商品（Yahoo Finance・2002年〜）から、銘柄ごとの「追い風/向かい風」を作る。

市場全体の値をそのまま特徴量にすると特定の時期の丸暗記になりやすい（ラウンド0の教訓）ので、
銘柄ごとの感応度（過去1年の日次リターンの β）× その要因の直近20日の動き、という形にする。
例: 円安が進んだとき、円安で上がりやすい輸出株には追い風、輸入株には向かい風。

為替は同じ日、米国市場は翌営業日の日本株と対応させて感応度を測り、出来上がった値は1日遅らせて使う（先読み防止）。
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)
SYMBOLS = {"usdjpy": "JPY=X", "spx": "^GSPC", "sox": "^SOX", "oil": "CL=F", "rate": "^TNX"}
MACRO_FEATURES = [f"beta_{k}" for k in SYMBOLS] + [f"tail_{k}" for k in SYMBOLS]
# AIに渡すのは原油と金利だけ。全部入れると過去ペーパートレードが +11.1% → −0.4% に悪化し（時期の丸暗記）、
# 原油・金利だけなら +11.7%・最大下落 −46% → −34%（ジャーナル ラウンド7）
MODEL_MACRO_FEATURES = ["beta_oil", "tail_oil", "beta_rate", "tail_rate"]


def _path(cfg):
    return cfg["data"]["db_path"].parent / "macro.parquet"


def update(cfg: dict) -> pd.DataFrame:
    """全期間を取り直しても数秒なので毎回まとめて取得する。"""
    raw = yf.download(list(SYMBOLS.values()), start=cfg["data"]["history_start"], progress=False,
                      auto_adjust=False, group_by="ticker")
    out = pd.DataFrame({k: raw[s]["Close"] for k, s in SYMBOLS.items()})
    out.index = pd.to_datetime(out.index).tz_localize(None).normalize()
    out = out.dropna(how="all")
    if len(out) > 1000:
        out.to_parquet(_path(cfg))
    return out


def load(cfg: dict) -> pd.DataFrame | None:
    p = _path(cfg)
    return pd.read_parquet(p) if p.exists() else None


def wide_features(macro: pd.DataFrame | None, close: pd.DataFrame, window: int = 250) -> dict[str, pd.DataFrame]:
    idx, cols = close.index, close.columns
    if macro is None or macro.empty:
        return {f: pd.DataFrame(np.nan, index=idx, columns=cols, dtype="float32") for f in MACRO_FEATURES}
    # 感応度の推定では、その要因が動いた時間帯に合わせて日本株の日次リターンと対応させる:
    #   為替は東京の取引時間中にも動くので同じ日、米国市場（株・半導体・原油・金利）は日本の翌営業日と対応
    # そのうえで、出来上がった値はすべて1日遅らせる（その日の日本の引けの時点で確定している情報だけにする）
    m = macro.reindex(macro.index.union(idx)).ffill().reindex(idx)
    r = close.pct_change(fill_method=None)
    lag = {"usdjpy": 0, "spx": 1, "sox": 1, "oil": 1, "rate": 1}
    w2 = window // 2
    out = {}
    for k in SYMBOLS:
        lv = m[k].shift(lag[k])
        mr = lv.diff() if k == "rate" else lv.pct_change(fill_method=None)  # 金利は変化幅、それ以外は騰落率
        cov = (r.mul(mr, axis=0).rolling(window, min_periods=w2).mean()
               - r.rolling(window, min_periods=w2).mean().mul(mr.rolling(window, min_periods=w2).mean(), axis=0))
        beta = cov.div(mr.rolling(window, min_periods=w2).var().replace(0, np.nan), axis=0).clip(-5, 5).shift(1)
        known = m[k].shift(1)  # 日本の引けの時点で確定している値
        move20 = (known - known.shift(20)) if k == "rate" else (known / known.shift(20) - 1)
        out[f"beta_{k}"] = beta
        out[f"tail_{k}"] = beta.mul(move20, axis=0)
    return {k: v.astype("float32") for k, v in out.items()}
