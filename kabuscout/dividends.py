"""配当の特徴量（Yahoo の配当履歴・2002年〜・全銘柄）。

- div_yield: 直近1年（250営業日）の配当合計 ÷ 株価
- div_growth: 直近1年の配当合計 ÷ その前の1年 − 1（増配・減配）
- days_to_ex: 次の権利落ち日までの推定営業日数。「1年前の権利落ち日の暦の上で1年後」で推定する
  （日本は3月末・9月末に集中。過去の日付だけで推定するので先読みにならない）

注意: 株価は配当で調整していない（権利落ち日に配当分だけ下がる）。売買の成績には受け取る配当を
含めていないので、権利落ち日をまたいで持つ戦略は少し不利に評価される（保守的）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DIV_FEATURES = ["div_yield", "div_growth", "days_to_ex"]


def _ttm(dates: np.ndarray, amts: np.ndarray):
    """配当の回ごとに「直近1年分（年間の支払回数ぶん）の合計」と「その前の1年分」を返す。
    権利落ち日が年によって数日ずれても抜け・二重計上が起きないよう、日数ではなく回数で数える。"""
    n = len(dates)
    ttm, prev = np.full(n, np.nan), np.full(n, np.nan)
    for j in range(n):
        recent = (dates[: j + 1] > dates[j] - np.timedelta64(1095, "D")).sum()
        f = int(np.clip(round(recent / 3), 1, 4))
        if j + 1 >= f:
            ttm[j] = amts[j + 1 - f: j + 1].sum()
        if j + 1 >= 2 * f:
            prev[j] = amts[j + 1 - 2 * f: j + 1 - f].sum()
    return ttm, prev


def wide_features(div: pd.DataFrame, close: pd.DataFrame) -> dict[str, pd.DataFrame]:
    idx, cols = close.index, close.columns
    if div is None or div.empty:
        return {f: pd.DataFrame(np.nan, index=idx, columns=cols, dtype="float32") for f in DIV_FEATURES}
    d = div.reindex(index=idx, columns=cols).fillna(0.0)
    full = div.reindex(columns=cols).fillna(0.0)  # 窓より前の配当も使って年間合計を出す
    ttm_w = pd.DataFrame(np.nan, index=idx, columns=cols)
    prev_w = ttm_w.copy()
    for c in cols:
        s = full[c]
        s = s[s > 0]
        if s.empty:
            continue
        t, p = _ttm(s.index.to_numpy(), s.to_numpy(float))
        e = pd.DataFrame({"t": t, "p": p}, index=s.index)
        e = e[(e.index >= idx[0] - pd.Timedelta(days=500)) & (e.index <= idx[-1])]
        pos = idx.searchsorted(e.index)
        ok = pos < len(idx)
        ttm_w.iloc[pos[ok], cols.get_loc(c)] = e["t"].to_numpy()[ok]
        prev_w.iloc[pos[ok], cols.get_loc(c)] = e["p"].to_numpy()[ok]
    ttm_w = ttm_w.ffill(limit=400)
    prev_w = prev_w.ffill(limit=400)
    out = {
        "div_yield": (ttm_w / close).fillna(0.0).where(close.notna()),
        "div_growth": (ttm_w / prev_w.replace(0, np.nan) - 1).clip(-1, 3),
    }
    # 次の権利落ち日の推定: 1年前の権利落ち日の「暦の上で1年後」。営業日数で数えると年によって数日ずれて、
    # 実際の権利落ち日をまたいでしまう（配当分の値下がりを拾ってしまう）ため、暦日で推定する。
    # 最新日でも先の推定日が見えるよう、未来側に営業日を足してから位置を求める（中身は過去の実績なので先読みではない）
    T = len(idx)
    ext = idx.append(pd.bdate_range(idx[-1] + pd.Timedelta(days=1), periods=80))
    est = np.zeros((len(ext), len(cols)), dtype=bool)
    for c in cols:
        s_ = full[c]
        exd = s_[s_ > 0].index
        exd = exd[(exd >= idx[0] - pd.Timedelta(days=400)) & (exd <= idx[-1])]
        if len(exd):
            pos_ = ext.searchsorted(exd + pd.Timedelta(days=365))
            est[pos_[pos_ < len(ext)], cols.get_loc(c)] = True
    pos = np.where(est, np.arange(len(ext))[:, None], np.nan)
    nxt = pd.DataFrame(pos[::-1]).ffill().to_numpy()[::-1][:T]  # 自分以降で最初の推定権利落ち日
    days = nxt - np.arange(T)[:, None]
    out["days_to_ex"] = pd.DataFrame(np.where(days <= 60, days, np.nan), index=idx, columns=cols)
    return {k: v.astype("float32") for k, v in out.items()}
