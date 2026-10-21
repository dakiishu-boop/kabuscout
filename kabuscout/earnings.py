"""決算情報: Yahoo Finance の決算発表日・EPS予想・実績・サプライズ率（2004年頃〜）。

アナリスト予想がある銘柄（主にプライムの中・大型株）だけが対象で、全銘柄の約4分の1。
発表日の前後2日間の値動き（市場全体の動きを差し引いたもの）＝「決算への市場の反応」も計算し、
決算後の値動きの続きやすさ（決算後ドリフト）を特徴量・戦略として使う。

先読みを防ぐため、決算の情報は「発表日の翌営業日」から使えるものとして扱う。
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)
EARN_FEATURES = ["earn_age", "earn_react", "earn_surprise", "earn_vol", "earn_to_next"]


def _fetch_one(code: str) -> pd.DataFrame:
    x = yf.Ticker(f"{code}.T").get_earnings_dates(limit=100)
    if x is None or x.empty:
        return pd.DataFrame()
    x = x.dropna(how="all", subset=["EPS Estimate", "Reported EPS"]).reset_index()
    d = pd.to_datetime(x.iloc[:, 0], utc=True).dt.tz_convert("Asia/Tokyo")
    return pd.DataFrame({
        "code": code, "date": d.dt.tz_localize(None).dt.normalize(),
        "eps_est": x["EPS Estimate"].astype(float), "eps": x["Reported EPS"].astype(float),
        "surprise": x["Surprise(%)"].astype(float),
    })


def update(cfg: dict, codes: list[str], max_minutes: float | None = None) -> pd.DataFrame:
    """未取得の銘柄と、決算発表が近い/直後の銘柄を取り直す（全銘柄の初回取得は約30分）。"""
    path = cfg["data"]["db_path"].parent / "earnings.parquet"
    meta_path = path.with_suffix(".json")
    old = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["code", "date", "eps_est", "eps", "surprise"])
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    today = pd.Timestamp.today().normalize()
    # 次の発表日の前後（-3日〜+10日）の銘柄、または30日以上確認していない銘柄
    upcoming = old[(old["date"] >= today - pd.Timedelta(days=10)) & (old["date"] <= today + pd.Timedelta(days=3))]["code"]
    stale = [c for c in codes if c not in meta or meta[c] < (today - pd.Timedelta(days=30)).strftime("%Y-%m-%d")]
    todo = list(dict.fromkeys([c for c in codes if c not in meta] + list(upcoming) + stale))
    if not todo:
        return old
    log.info("決算情報を取得: %d 銘柄", len(todo))
    t0, new = time.time(), []
    for i, c in enumerate(todo):
        if max_minutes and time.time() - t0 > max_minutes * 60:
            log.info("  時間切れ（残りは次回）: %d / %d", i, len(todo))
            break
        try:
            new.append(_fetch_one(c))
            meta[c] = today.strftime("%Y-%m-%d")
        except Exception as e:
            if "Too Many Requests" in str(e) or "Rate" in str(e):
                log.warning("  Yahoo の回数制限。60秒待ちます")
                time.sleep(60)
            else:
                meta[c] = today.strftime("%Y-%m-%d")  # データ無しの銘柄は30日後に再確認
        if i % 200 == 0:
            log.info("  決算情報 %d / %d", i, len(todo))
        time.sleep(0.15)
    new = [n for n in new if not n.empty]
    if new:
        got = pd.concat(new, ignore_index=True)
        old = pd.concat([old[~old["code"].isin(got["code"].unique())], got], ignore_index=True)
    old = old.sort_values(["code", "date"]).drop_duplicates(["code", "date"], keep="last")
    old.to_parquet(path)
    meta_path.write_text(json.dumps(meta))
    return old


def load(cfg: dict) -> pd.DataFrame | None:
    path = cfg["data"]["db_path"].parent / "earnings.parquet"
    return pd.read_parquet(path) if path.exists() else None


def wide_features(ev: pd.DataFrame | None, close: pd.DataFrame, volume: pd.DataFrame, mkt: pd.Series,
                  max_age: int = 60) -> dict[str, pd.DataFrame]:
    """決算イベントを「日付×銘柄」の行列にする。値は発表日の翌営業日から max_age 営業日まで有効。
    - earn_react: 発表前日終値→発表翌日終値の騰落率から、同期間の市場の騰落率を引いたもの
    - earn_surprise: EPS サプライズ率（%）
    - earn_vol: 発表翌日までの出来高の、それ以前20日平均に対する倍率
    - earn_age: 情報が使えるようになってからの営業日数（0=今日初めて使える）
    - earn_to_next: 次の決算発表までの暦日数（発表日は事前に公表されるので先読みには当たらない）"""
    idx, cols = close.index, close.columns
    out = {k: pd.DataFrame(np.nan, index=idx, columns=cols, dtype="float32") for k in EARN_FEATURES}
    if ev is None or ev.empty:
        return out
    ev = ev[ev["code"].isin(set(cols)) & (ev["date"] >= idx[0] - pd.Timedelta(days=120))]
    c, v, m = close.to_numpy(), volume.to_numpy(), mkt.to_numpy()
    cix = {k: i for i, k in enumerate(cols)}
    T = len(idx)
    react = np.full((T, len(cols)), np.nan, dtype="float32")
    surp, vol, known = react.copy(), react.copy(), react.copy()
    vavg = pd.DataFrame(v).rolling(20, min_periods=10).mean().shift(1).to_numpy()
    for code, d, s in ev[["code", "date", "surprise"]].itertuples(index=False):
        j = cix[code]
        i0 = idx.searchsorted(d)          # 発表日（休日なら次の営業日）
        k = i0 + 1                        # 情報が使えるのは翌営業日の引け後から
        if i0 < 1 or k >= T:
            continue
        r = c[k, j] / c[i0 - 1, j] - 1 - (m[k] / m[i0 - 1] - 1)
        react[k, j] = r
        surp[k, j] = s if s == s else np.nan
        vol[k, j] = np.nanmax([v[i0, j], v[k, j]]) / vavg[i0, j] if vavg[i0, j] and vavg[i0, j] == vavg[i0, j] else np.nan
        known[k, j] = k
    f = lambda a: pd.DataFrame(a, index=idx, columns=cols).ffill(limit=max_age)
    kn = f(known)
    out["earn_react"], out["earn_surprise"], out["earn_vol"] = f(react), f(surp), f(vol)
    out["earn_age"] = (pd.DataFrame(np.arange(T)[:, None].repeat(len(cols), 1), index=idx, columns=cols) - kn).where(kn.notna())
    # 次の発表日までの日数
    nxt = pd.DataFrame(np.nan, index=idx, columns=cols)
    for code, g in ev.groupby("code"):
        ds = np.sort(g["date"].to_numpy())
        pos = np.searchsorted(ds, idx.to_numpy(), side="right")
        valid = pos < len(ds)
        days = np.full(T, np.nan)
        days[valid] = (ds[pos[valid]] - idx.to_numpy()[valid]) / np.timedelta64(1, "D")
        nxt[code] = days
    out["earn_to_next"] = nxt.where(nxt <= 120)
    return {k: v.astype("float32") for k, v in out.items()}
