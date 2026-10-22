"""資金の一部を指数（S&P500・全世界株式）に固定して持つ「コア」（残りをAIで運用する）。

ジャーナル ラウンド9: 資金の約30%を S&P500（またはオルカン）に固定すると、AIだけより年率・最大下落・最悪の年・元本割れの確率が
すべて良くなった（AI＝日本の小型株・下げ相場に強い、米国株の指数＝長期の成長、で値動きの性質が違う）。
余った現金を毎日指数に入れる方式は、AIが悪い相場ほど現金を多く持つため逆効果だった。

- ペーパートレード: 東証に上場している円建てのETFの終値で、指数の部分を追う（2558 = S&P500、2559 = 全世界株式）。
  最初に買って持ち続ける（実際の運用では年1回だけ比率を戻す）
- 過去ペーパートレード: S&P500（配当込み）・全世界株式（ACWI・配当込み）を円に換算して、資金の share を固定した場合も参考に出す
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)
ETF_NAMES = {"2558": "S&P500", "2559": "全世界株式（オルカン）"}


def share(cfg: dict) -> float:
    return float((cfg["paper_trading"].get("core") or {}).get("share") or 0.0)


def ai_capital(cfg: dict) -> float:
    """AIで運用する資金（コアを持つ場合は、その分を除いた残り）。"""
    return cfg["paper_trading"]["capital_jpy"] * (1 - share(cfg))


def _state_path(cfg):
    return cfg["data"]["db_path"].parent / "core.json"


def step(cfg: dict, data_date: pd.Timestamp) -> dict | None:
    """ペーパートレードの指数の部分（初回に買って持ち続ける）。"""
    sh = share(cfg)
    if sh <= 0:
        return None
    code = str((cfg["paper_trading"].get("core") or {}).get("etf", "2558"))
    try:
        h = yf.Ticker(f"{code}.T").history(period="10d", auto_adjust=False)["Close"].dropna()
        h.index = pd.to_datetime(h.index).tz_localize(None).normalize()
        h = h[h.index <= data_date]
        price = float(h.iloc[-1])
    except Exception as e:
        log.warning("指数ETF %s の株価を取れませんでした: %s", code, e)
        return None
    p = _state_path(cfg)
    st = json.loads(p.read_text()) if p.exists() else {}
    budget = cfg["paper_trading"]["capital_jpy"] * sh
    if st.get("etf") != code or st.get("budget") != budget:  # 初回・設定の変更 → 買い直す
        units = int(budget // price)
        st = dict(etf=code, budget=budget, units=units, cash=budget - units * price, bought_at=price,
                  since=data_date.strftime("%Y-%m-%d"))
        log.info("コア: %s を %d口 @%.0f で購入（%s円）", code, units, price, f"{budget:,.0f}")
    value = st["units"] * price + st["cash"]
    st.update(price=price, value=value, date=data_date.strftime("%Y-%m-%d"))
    p.write_text(json.dumps(st, ensure_ascii=False))
    return dict(name=ETF_NAMES.get(code, code), etf=code, value=value, budget=budget, ret=value / budget - 1, since=st["since"])


def load_indices(cfg: dict) -> pd.DataFrame | None:
    p = cfg["data"]["db_path"].parent / "core_indices.parquet"
    try:
        raw = yf.download(["^SP500TR", "ACWI", "JPY=X"], start=cfg["data"]["history_start"], auto_adjust=True,
                          progress=False, group_by="ticker")
        out = pd.DataFrame({"spx_tr": raw["^SP500TR"]["Close"], "acwi": raw["ACWI"]["Close"], "usdjpy": raw["JPY=X"]["Close"]})
        out.index = pd.to_datetime(out.index).tz_localize(None).normalize()
        if len(out.dropna(how="all")) > 1000:
            out.to_parquet(p)
    except Exception as e:
        log.warning("指数の取得に失敗（前回の値を使います）: %s", e)
    return pd.read_parquet(p) if p.exists() else None


def index_returns_jpy(ix: pd.DataFrame, cal: pd.DatetimeIndex) -> pd.DataFrame:
    """円建ての指数の日次リターン（米国の終値は日本の翌営業日に反映、信託報酬として年0.1%を引く）。"""
    us = ix.ffill()
    out = {}
    for name, col in (("S&P500", "spx_tr"), ("全世界株式（オルカン）", "acwi")):
        v = (us[col] * us["usdjpy"]).dropna()
        pos = v.index.searchsorted(cal, side="left") - 1
        val = pd.Series(np.where(pos >= 0, v.to_numpy()[np.clip(pos, 0, None)], np.nan), index=cal)
        out[name] = val.pct_change().fillna(0) - 0.001 / 250
    return pd.DataFrame(out)


def combined(ai_eq: pd.Series, idx_ret: pd.Series, sh: float) -> pd.Series:
    """資金の sh を指数に固定し（年1回の組み直し）、残りをAIで運用した場合の資産の推移（初めを1とする）。"""
    rs = ai_eq.pct_change().fillna(0)
    parts = []
    for _, g in rs.groupby(rs.index.year):
        gi = idx_ret.reindex(g.index).fillna(0)
        v = sh * np.cumprod(1 + gi) + (1 - sh) * np.cumprod(1 + g)
        parts.append(v / np.r_[1.0, v.to_numpy()[:-1]] - 1)
    return (1 + pd.concat(parts)).cumprod()
