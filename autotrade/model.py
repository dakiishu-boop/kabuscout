"""横並びの順位を予測するAI（勾配ブースティング）と、1年ずつ進めるウォークフォワード。

学習の正解: 翌営業日の寄付で買い、H 営業日後の寄付で売ったときの騰落率の、その日の対象銘柄の中での順位（0〜1）。
相場全体が上がるか下がるかではなく「他の銘柄より上がるか」だけを当てにいく。

ウォークフォワード: 年 Y の予測には、Y の前年末から H+1 営業日さかのぼった日までのデータだけで学習したモデルを使う
（正解の期間がテストの年にかからないように空ける）。kabuscout の backtest.walk_forward と同じ考え方。
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

DEFAULT = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=2000,
               l2_regularization=1.0, max_bins=64)


def make_model(params: dict | None = None, seed: int = 0):
    from sklearn.ensemble import HistGradientBoostingRegressor
    kw = {**DEFAULT, **(params or {})}
    return HistGradientBoostingRegressor(early_stopping=False, random_state=seed, **kw)


def recency_weight(dates: pd.Series, ref: pd.Timestamp, half_life_days: float | None) -> np.ndarray | None:
    if not half_life_days:
        return None
    age = (ref - pd.to_datetime(dates)).dt.days.to_numpy()
    return np.power(0.5, age / half_life_days)


def fit(train: pd.DataFrame, features: list[str], target: str, params: dict | None = None, seeds=(0,),
        half_life_days: float | None = None, sample_every: int = 1):
    d = train
    if sample_every > 1:
        days = np.sort(d["date"].unique())
        d = d[d["date"].isin(set(days[::sample_every]))]
    d = d[d[target].notna()]
    w = recency_weight(d["date"], d["date"].max(), half_life_days)
    X, y = d[features].to_numpy("float32"), d[target].to_numpy("float32") - 0.5
    return [make_model(params, s).fit(X, y, sample_weight=w) for s in seeds]


def predict(models, df: pd.DataFrame, features: list[str]) -> np.ndarray:
    X = df[features].to_numpy("float32")
    return np.mean([m.predict(X) for m in models], axis=0)


def walk_forward(ds: pd.DataFrame, features: list[str], target: str, horizon: int, start_year: int,
                 params: dict | None = None, seeds=(0,), half_life_days: float | None = None,
                 sample_every: int = 3, end_year: int | None = None, train_col: str | None = "elig",
                 min_train_days: int = 500) -> pd.Series:
    """年ごとに学習し直して、その年の全行の点数を返す（index は ds と同じ）。
    学習には train_col が True の行（新規に買える銘柄）だけを使い、予測は全行に付ける（保有を続けるかの判断に使う）。"""
    dates = np.sort(ds["date"].unique())
    out = pd.Series(np.nan, index=ds.index, dtype="float32")
    last = end_year or pd.Timestamp(dates[-1]).year
    for y in range(start_year, last + 1):
        y0 = np.datetime64(f"{y}-01-01")
        before = dates[dates < y0]
        if len(before) < min_train_days:
            continue
        purge = before[-(horizon + 2)]  # 正解の期間（H+1営業日）がテストの年にかからないように
        tr = ds[(ds["date"] < purge) & (ds[train_col] if train_col else True)]
        te_mask = (ds["date"] >= y0) & (ds["date"] < np.datetime64(f"{y + 1}-01-01"))
        if not te_mask.any():
            continue
        models = fit(tr, features, target, params, seeds, half_life_days, sample_every)
        out[te_mask] = predict(models, ds[te_mask], features)
        log.info("  %d年: 学習 %d行（〜%s）→ %d行を予測", y, int(tr[target].notna().sum()), str(purge)[:10], int(te_mask.sum()))
    return out
