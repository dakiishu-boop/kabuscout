"""データの鮮度の確認: 取得に失敗しても実行自体は成功扱いになる「静かな失敗」を、毎日の通知の先頭で知らせる。

2026-11-05 の実行では適時開示（TDnet）の取得が 0/5日 だったのに、通知には何も出なかった（ジャーナル §72）。
営業日の暦（株価の日付）を基準に、前の営業日までのデータが揃っているかを確かめる。休場日は株価の日付が進まないので知らせない。
"""
from __future__ import annotations

import json
import logging

import pandas as pd

log = logging.getLogger(__name__)
ALERTS_MAX_DAYS = 7  # 監理・整理銘柄の一覧（JPX）がこれより古いままなら知らせる


def check(cfg: dict, cal: pd.DatetimeIndex) -> list[str]:
    """cal: その日までの営業日（株価の日付）。遅れているデータの説明を返す（無ければ空）。"""
    if len(cal) < 2:
        return []
    last, prev = pd.Timestamp(cal[-1]).normalize(), pd.Timestamp(cal[-2]).normalize()
    base = cfg["data"]["db_path"].parent
    out = []
    # 適時開示は通知の後に取るので、今日の実行の時点で揃っているべきなのは前の営業日まで
    p = base / "tdnet.parquet"
    try:
        newest = pd.read_parquet(p, columns=["pubdate"])["pubdate"].max() if p.exists() else None
        if newest is None or pd.isna(newest):
            out.append("適時開示のデータがありません（決算・開示の特徴量が空になります）")
        elif pd.Timestamp(newest).normalize() < prev:
            out.append(f"適時開示が {pd.Timestamp(newest):%m/%d} で止まっています（{prev:%m/%d} の分が無い）。決算後ドリフトなどの判断に影響します")
    except Exception as e:
        out.append(f"適時開示のデータを読めません: {e}")
    # 海外市場・為替は実行の最初に取るので、前の営業日まであるはず（海外の休日で1日遅れることはある）
    p = base / "macro.parquet"
    try:
        if p.exists():
            m = pd.read_parquet(p)
            newest = pd.Timestamp(m.index.max()).normalize() if len(m) else None
            if newest is None or (last - newest).days > 4:
                out.append(f"海外市場・為替が {newest:%m/%d} で止まっています" if newest is not None else "海外市場・為替のデータが空です")
    except Exception as e:
        out.append(f"海外市場・為替のデータを読めません: {e}")
    p = base / "alerts.json"
    try:
        if p.exists():
            d = pd.Timestamp(json.loads(p.read_text())["date"])
            if (last - d).days > ALERTS_MAX_DAYS:
                out.append(f"監理・整理銘柄の一覧が {d:%m/%d} のままです（JPX のページの形が変わった可能性）。古い一覧で除外しています")
        else:
            out.append("監理・整理銘柄の一覧がありません（上場廃止のおそれがある銘柄を除外できていません）")
    except Exception as e:
        out.append(f"監理・整理銘柄の一覧を読めません: {e}")
    for s in out:
        log.warning("データの遅れ: %s", s)
    return out
