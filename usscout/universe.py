"""銘柄の母集団。

- ETF: 戦略ごとに決めた本数（usscout.yaml）。どれも2007年より前に上場しているので、2008年からの検証で銘柄の後知恵が入らない
- S&P500 の「その時点の」構成銘柄: Wikipedia の今の構成銘柄の表（採用日つき）と、入れ替えの履歴の表から逆算する。
  今の構成銘柄だけで過去を検証すると、後から大きく伸びて採用された会社（例: 2020年採用の TSLA）を昔から選べてしまう（後知恵）。
  残る弱点（docs/JOURNAL.md §90）:
  - 入れ替えの履歴は 2008〜2010年の分に抜けがある（年8〜13件。2011年以降は年16〜30件）
  - 倒産・買収で上場廃止になった会社は Yahoo に株価が無い → その時点の構成銘柄でも検証に入らない（生存者バイアス）。
    何割の銘柄に株価があったかを coverage として毎回表示する
"""
from __future__ import annotations

import io
import logging
from pathlib import Path

import pandas as pd
import requests

from .data import to_yahoo

log = logging.getLogger(__name__)
CURRENT_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
HISTORY_URL = "https://en.wikipedia.org/wiki/Historical_components_of_the_S%26P_500"
UA = {"User-Agent": "usscout-research/0.1 (personal backtest; contact via repository)"}
EXTRA = ["SPY", "QQQ", "BIL", "AGG", "^IRX", "JPY=X"]  # 物差し・逃避先・短期金利（BIL が無い時期の代わり・米ドルMMF の利回り）・ドル円


def fetch_sp500(root: Path, refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(今の構成銘柄, 入れ替えの履歴) を Wikipedia から取り、CSV に残す（2回目からは CSV を読む）。"""
    root = Path(root)
    cur_p, hist_p = root / "sp500_current.csv", root / "sp500_changes.csv"
    if cur_p.exists() and hist_p.exists() and not refresh:
        return pd.read_csv(cur_p, parse_dates=["added"]), pd.read_csv(hist_p, parse_dates=["date"])
    root.mkdir(parents=True, exist_ok=True)
    cur = pd.read_html(io.StringIO(requests.get(CURRENT_URL, headers=UA, timeout=60).text))[0]
    cur = pd.DataFrame({"symbol": cur["Symbol"].map(to_yahoo), "name": cur["Security"], "sector": cur["GICS Sector"],
                        "added": pd.to_datetime(cur["Date added"], errors="coerce")})
    h = pd.read_html(io.StringIO(requests.get(HISTORY_URL, headers=UA, timeout=60).text))[0]
    h = h.iloc[:, :6]
    h.columns = ["date", "add", "add_name", "rem", "rem_name", "reason"]
    h["date"] = pd.to_datetime(h["date"], errors="coerce", format="mixed")
    h = h.dropna(subset=["date"])
    for c in ("add", "rem"):
        h[c] = h[c].where(h[c].notna(), None).map(lambda s: to_yahoo(s) if isinstance(s, str) and s.strip() else None)
    cur.to_csv(cur_p, index=False)
    h.to_csv(hist_p, index=False)
    log.info("S&P500: 今の構成銘柄 %d、入れ替えの履歴 %d 件（%s〜）", len(cur), len(h), h["date"].min().date())
    return cur, h


def membership(cur: pd.DataFrame, changes: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """日付×銘柄の「その日に構成銘柄だったか」（True/False）。

    今の構成銘柄から出発して、入れ替えを新しい順に巻き戻す。入れ替えは発効日の寄付から有効とみなす
    （発効日より前の日は入れ替え前の構成）。今の構成銘柄は、表の採用日より前は構成銘柄でないとする
    （入れ替えの履歴に抜けがある時期の補い）。
    """
    ev = []
    for r in changes.itertuples(index=False):
        if isinstance(r.add, str) and r.add:
            ev.append((pd.Timestamp(r.date), r.add, "add"))
        if isinstance(r.rem, str) and r.rem:
            ev.append((pd.Timestamp(r.date), r.rem, "rem"))
    ev.sort(key=lambda e: e[0], reverse=True)
    members = set(cur["symbol"])
    added = dict(zip(cur["symbol"], pd.to_datetime(cur["added"])))
    names = sorted(members | {e[1] for e in ev})
    out = pd.DataFrame(False, index=dates, columns=names)
    i = 0
    for d in sorted(dates, reverse=True):
        while i < len(ev) and ev[i][0] > d:
            _, sym, kind = ev[i]
            if kind == "add":
                members.discard(sym)
            else:
                members.add(sym)
            i += 1
        ok = [s for s in members if not (s in added and pd.notna(added[s]) and added[s] > d)]
        out.loc[d, ok] = True
    return out


def sp500_symbols(cur: pd.DataFrame, changes: pd.DataFrame) -> list[str]:
    s = set(cur["symbol"]) | {x for x in changes["add"].dropna()} | {x for x in changes["rem"].dropna()}
    return sorted(x for x in s if isinstance(x, str) and x)


def etf_symbols(cfg: dict) -> list[str]:
    st = cfg["strategies"]
    out = list(st["gem"]["risk"]) + [st["gem"]["safe"], st["gem"]["cash"]] + list(st["etf_rotation"]["universe"])
    out += [st["trend_qqq"]["asset"], st["stock_mom"]["market"]]
    return list(dict.fromkeys(out + EXTRA))
