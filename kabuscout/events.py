"""適時開示イベントを「日付×銘柄」の特徴量にする。

ev_<種類>_age = その開示が株価に織り込まれ始めてからの営業日数（0=その日。60営業日を過ぎたら空欄）。
15:00以降の開示は翌営業日から（後場の引けまでに反応できないため）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EVENT_TYPES = ["buyback", "forecast_up", "forecast_down", "dividend_up", "split", "yutai_new", "offering"]
EVENT_FEATURES = [f"ev_{k}_age" for k in EVENT_TYPES]


def known_dates(pub: pd.Series, idx: pd.DatetimeIndex) -> np.ndarray:
    """開示時刻 → その情報が引け値に入る最初の営業日の位置。"""
    d = pub.dt.normalize()
    after_close = pub.dt.hour >= 15
    pos = idx.searchsorted(d.to_numpy())
    # 営業日の15時以降の開示、または休日の開示（searchsorted が次の営業日を返す）
    same_day = (pos < len(idx)) & (idx[np.minimum(pos, len(idx) - 1)] == d.to_numpy())
    return np.where(same_day & after_close.to_numpy(), pos + 1, pos)


def wide_features(td: pd.DataFrame | None, close: pd.DataFrame, max_age: int = 60) -> dict[str, pd.DataFrame]:
    idx, cols = close.index, close.columns
    T, N = len(idx), len(cols)
    out = {f: pd.DataFrame(np.nan, index=idx, columns=cols, dtype="float32") for f in EVENT_FEATURES}
    if td is None or td.empty:
        return out
    td = td[td["code"].isin(set(cols)) & (td["pubdate"] >= idx[0] - pd.Timedelta(days=100))
            & (td["pubdate"] <= idx[-1] + pd.Timedelta(days=1))]
    if td.empty:
        return out
    cix = {k: i for i, k in enumerate(cols)}
    k_pos = known_dates(td["pubdate"], idx)
    rows = np.arange(T)[:, None]
    for typ in EVENT_TYPES:
        m = td["events"].str.contains(typ).to_numpy() & (k_pos < T)
        if not m.any():
            continue
        known = np.full((T, N), np.nan, dtype="float32")
        known[k_pos[m], [cix[c] for c in td.loc[m, "code"]]] = k_pos[m]
        kn = pd.DataFrame(known, index=idx, columns=cols).ffill(limit=max_age)
        out[f"ev_{typ}_age"] = (pd.DataFrame(rows.repeat(N, 1), index=idx, columns=cols) - kn).where(kn.notna()).astype("float32")
    return out


def earnings_dates(td: pd.DataFrame | None) -> pd.DataFrame:
    """TDnet の決算短信から、全銘柄の決算発表日（Yahoo の決算情報を補う）。"""
    if td is None or td.empty:
        return pd.DataFrame(columns=["code", "date", "surprise"])
    e = td[td["events"].str.contains("earnings")]
    # 15時以降の発表は翌営業日に反応する。earnings.wide_features は発表日の前日〜翌日の2日間で反応を測るので日付のままでよい
    return pd.DataFrame({"code": e["code"], "date": e["pubdate"].dt.normalize(), "surprise": np.nan})
