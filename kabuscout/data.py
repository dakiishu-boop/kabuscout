"""データ取得層。

- 銘柄一覧: JPX「東証上場銘柄一覧」(無料・毎月更新)
- 株価: Yahoo Finance (yfinance, 無料・日足)。分割調整済みの OHLCV を Parquet にキャッシュし、
  毎日は直近分だけを差分取得する。株式分割を検知したらその銘柄だけ全期間を取り直す。
"""
from __future__ import annotations

import io
import json
import logging
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import requests
import yfinance as yf

log = logging.getLogger(__name__)

JPX_PAGE = "https://www.jpx.co.jp/markets/statistics-equities/misc/01.html"
FIELDS = ["open", "high", "low", "close", "volume"]
# 株価と一緒に取得・保存する項目（配当は権利落ち日に1株あたり配当額、分割調整済み）
STORE_FIELDS = FIELDS + ["dividends"]


# ---------------------------------------------------------------- universe
def fetch_jpx_universe(markets: list[str]) -> pd.DataFrame:
    page = requests.get(JPX_PAGE, timeout=30)
    page.raise_for_status()
    m = re.search(r'href="([^"]+data_j\.xlsx?)"', page.text)
    if not m:
        raise RuntimeError("JPXの銘柄一覧ファイルが見つかりません")
    r = requests.get("https://www.jpx.co.jp" + m.group(1), timeout=60)
    r.raise_for_status()
    df = pd.read_excel(io.BytesIO(r.content), dtype=str)
    df = df.rename(columns={
        "コード": "code", "銘柄名": "name", "市場・商品区分": "market",
        "33業種区分": "sector33", "17業種区分": "sector17", "規模区分": "scale",
    })
    df = df[df["market"].str.contains("内国株式", na=False)]
    df = df[df["market"].apply(lambda s: any(k in s for k in markets))]
    df["code"] = df["code"].str.strip()
    return df[["code", "name", "market", "sector33", "sector17", "scale"]].reset_index(drop=True)


def refresh_universe(con, cfg: dict, force: bool = False) -> pd.DataFrame:
    cur = pd.read_sql("SELECT * FROM universe", con)
    if not cur.empty and not force:
        age = (datetime.now() - pd.to_datetime(cur["updated"]).max()).days
        if age < cfg["universe"]["refresh_days"]:
            return cur
    try:
        df = fetch_jpx_universe(cfg["universe"]["markets"])
    except Exception as e:  # ネットワーク障害時は既存の一覧で続行
        if cur.empty:
            raise
        log.warning("銘柄一覧の更新に失敗（既存を使用）: %s", e)
        return cur
    df["updated"] = datetime.now().isoformat(timespec="seconds")
    con.execute("DELETE FROM universe")
    df.to_sql("universe", con, if_exists="append", index=False)
    con.commit()
    log.info("銘柄一覧を更新: %d 銘柄", len(df))
    return df


# ---------------------------------------------------------------- prices
class PriceStore:
    """株価を「日付×銘柄」のワイド行列として項目ごとに Parquet 保存する（25年×全銘柄でも数百MB）。"""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.root / "meta.json"

    def _path(self, f: str) -> Path:
        return self.root / f"{f}.parquet"

    def exists(self) -> bool:
        return self._path("close").exists()

    def meta(self) -> dict:
        return json.loads(self.meta_path.read_text()) if self.meta_path.exists() else {}

    def save_meta(self, meta: dict) -> None:
        self.meta_path.write_text(json.dumps(meta))

    def load(self, start: str | None = None, end: str | None = None,
             codes: list[str] | None = None, fields: list[str] | None = None, clean: bool = True) -> dict:
        if not self.exists():
            return {f: pd.DataFrame() for f in (fields or FIELDS)}
        cols = None
        if codes is not None:
            have = set(pq.read_schema(self._path("close")).names)
            cols = [c for c in codes if c in have]
        out = {}
        for f in fields or FIELDS:
            df = pd.read_parquet(self._path(f), columns=cols)
            df.index = pd.to_datetime(df.index)
            if start:
                df = df[df.index >= pd.Timestamp(start)]
            if end:
                df = df[df.index <= pd.Timestamp(end)]
            out[f] = df
        return clean_panel(out) if clean and fields is None else out

    def load_dividends(self, start: str | None = None, end: str | None = None, codes: list[str] | None = None) -> pd.DataFrame:
        """権利落ち日ごとの1株配当（日付×銘柄）。未取得なら空。"""
        if not self._path("dividends").exists():
            return pd.DataFrame()
        have = set(pq.read_schema(self._path("dividends")).names)
        df = pd.read_parquet(self._path("dividends"), columns=None if codes is None else [c for c in codes if c in have])
        df.index = pd.to_datetime(df.index)
        # Yahoo の配当には桁違いの誤り（株価の10万倍など）が混じるので、1回で株価の30%を超える配当は除く
        # （普通配当は通常10%未満、特別配当でも30%を超えることはまず無い・ジャーナル ラウンド9）
        close = pd.read_parquet(self._path("close"), columns=[c for c in df.columns if c in set(pq.read_schema(self._path("close")).names)])
        close.index = pd.to_datetime(close.index)
        ref = close.reindex(df.index.union(close.index)).ffill().shift(1).reindex(df.index)  # 権利落ち日の前日の終値
        df = df.mask((df / ref.reindex(columns=df.columns)) > 0.3)
        if start:
            df = df[df.index >= pd.Timestamp(start)]
        if end:
            df = df[df.index <= pd.Timestamp(end)]
        return df

    def save(self, panel: dict) -> None:
        for f in STORE_FIELDS:
            if f not in panel or panel[f] is None or (f == "dividends" and panel[f].empty):
                continue
            df = panel[f].sort_index()
            df.index.name = "date"
            df.columns = [str(c) for c in df.columns]
            df.astype("float32").to_parquet(self._path(f), compression="zstd")

    def upsert(self, new: dict, replace: set[str] = frozenset()) -> None:
        """new の値で上書き。replace の銘柄は既存データを捨ててから入れ直す。"""
        if not new or new["close"].empty:
            return
        old = self.load(clean=False)
        old["dividends"] = self.load_dividends()
        merged = {}
        for f in STORE_FIELDS:
            n = new.get(f, pd.DataFrame())
            o = old[f].drop(columns=[c for c in replace if c in old[f].columns]) if not old[f].empty else old[f]
            merged[f] = n.combine_first(o) if not o.empty and not n.empty else (n if not n.empty else o)
        self.save(merged)

    def last_dates(self) -> pd.Series:
        if not self.exists():
            return pd.Series(dtype=object)
        c = self.load(fields=["close"])["close"]
        return c.apply(lambda s: s.last_valid_index()).dropna().map(lambda d: d.strftime("%Y-%m-%d"))

    def first_dates(self) -> pd.Series:
        if not self.exists():
            return pd.Series(dtype=object)
        c = self.load(fields=["close"])["close"]
        return c.apply(lambda s: s.first_valid_index()).dropna().map(lambda d: d.strftime("%Y-%m-%d"))


def ticker(code: str) -> str:
    """Yahoo のティッカー。指数（^N225 など）はそのまま、個別株は「コード.T」。"""
    return code if code.startswith("^") else f"{code}.T"


def _wide(raw: pd.DataFrame, codes: list[str]) -> dict:
    out = {f: [] for f in STORE_FIELDS}
    if raw is None or raw.empty:
        return out
    if not isinstance(raw.columns, pd.MultiIndex):
        raw.columns = pd.MultiIndex.from_product([[ticker(codes[0])], raw.columns])
    raw.index = pd.to_datetime(raw.index).tz_localize(None).normalize()
    for f in STORE_FIELDS:
        try:
            w = raw.xs(f.capitalize(), axis=1, level=1)
        except KeyError:
            continue
        w.columns = [str(c).replace(".T", "") for c in w.columns]
        out[f].append(w.astype("float32"))
    return out


def _download(codes: list[str], batch: int, **kw) -> dict:
    parts = {f: [] for f in STORE_FIELDS}
    for i in range(0, len(codes), batch):
        chunk = codes[i:i + batch]
        for attempt in range(3):
            try:
                raw = yf.download([ticker(c) for c in chunk], group_by="ticker",
                                  auto_adjust=False, actions=True, progress=False, threads=True, **kw)
                for f, lst in _wide(raw, chunk).items():
                    parts[f] += lst
                break
            except Exception as e:
                log.warning("取得失敗 (%d/%d) 再試行 %d: %s", i, len(codes), attempt + 1, e)
                time.sleep(5 * (attempt + 1))
        if (i // batch) % 5 == 0:
            log.info("  株価取得 %d / %d", min(i + batch, len(codes)), len(codes))
        time.sleep(0.5)
    out = {}
    for f in STORE_FIELDS:
        if parts[f]:
            df = pd.concat(parts[f], axis=1)
            df = df.loc[:, ~df.columns.duplicated()]
            out[f] = df
        else:
            out[f] = pd.DataFrame()
    if not out["close"].empty:
        keep = out["close"].dropna(how="all", axis=1).columns
        out = {f: v.reindex(columns=keep).dropna(how="all", axis=0) if f == "close" else v.reindex(columns=keep)
               for f, v in out.items()}
        idx = out["close"].index
        out = {f: v.reindex(idx) for f, v in out.items()}
    return out


def update_prices(store: PriceStore, codes: list[str], cfg: dict) -> None:
    """差分更新。新規銘柄・分割検知銘柄・履歴が設定より短い銘柄は全期間取得。"""
    batch = cfg["data"]["batch_size"]
    hstart = cfg["data"]["history_start"]
    last = store.last_dates().to_dict()
    meta = store.meta()

    cutoff = (datetime.now() - timedelta(days=45)).strftime("%Y-%m-%d")
    today = datetime.now().strftime("%Y-%m-%d")
    recently_tried = lambda c: meta.get(f"tried:{c}", "") >= (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    # データが1本も無い銘柄（新規上場直後など）は週1回だけ取り直す
    fresh = [c for c in codes if (c not in last and not recently_tried(c))
             or (c in last and (last[c] < cutoff or meta.get(c, "9999") > hstart))]
    if not store.load_dividends(codes=["7203"]).shape[1] and last:
        # 配当データを持っていない旧形式の保存データ → 一度だけ全期間を取り直して配当も揃える
        log.info("配当データを揃えるため、全銘柄の全期間を取り直します（初回のみ）")
        fresh = [c for c in codes if c in last or not recently_tried(c)]
    incremental = [c for c in codes if c in last and c not in set(fresh)]

    replace: set[str] = set()
    if incremental:
        start = (pd.to_datetime(min(last[c] for c in incremental)) - timedelta(days=10)).strftime("%Y-%m-%d")
        log.info("差分取得: %d 銘柄 (%s〜)", len(incremental), start)
        recent = _download(incremental, batch, start=start)
        if not recent["close"].empty:
            # 分割・併合の検知: 既存データと重なる期間の終値が2%以上ずれたら取り直し
            old = store.load(start=start, fields=["close"])["close"]
            common = recent["close"].columns.intersection(old.columns)
            a = recent["close"][common]
            b = old[common].reindex(a.index)
            diff = (a / b - 1).abs().max()
            split_codes = sorted(diff[diff > 0.02].index)
            if len(split_codes) > max(20, 0.2 * len(common)):
                # 大量の銘柄が一斉にずれるのは分割ではなく Yahoo 側の一時的な不具合（日付ずれ等）
                log.warning("株価データの不整合が多すぎます（%d銘柄）。今回の差分は取り込みません", len(split_codes))
                return
            if split_codes:
                log.info("分割等を検知、全期間再取得: %s", split_codes[:20])
                fresh += split_codes
                recent = {f: v.drop(columns=split_codes, errors="ignore") for f, v in recent.items()}
            store.upsert(recent)

    if fresh:
        log.info("全期間取得: %d 銘柄 (%s〜)", len(fresh), hstart)
        full = _download(fresh, batch, start=hstart)
        replace = set(full["close"].columns) if not full["close"].empty else set()
        store.upsert(full, replace=replace)
        for c in fresh:
            meta[c] = hstart  # 取得を試みた開始日（上場が新しい銘柄を毎回取り直さないため）
            meta[f"tried:{c}"] = today
        store.save_meta(meta)


def load_panel(store: PriceStore, start: str | None = None, end: str | None = None,
               codes: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """日付×銘柄のワイド形式 DataFrame の辞書 {open, high, low, close, volume} を返す。"""
    return store.load(start=start, end=end, codes=codes)


def clean_panel(panel: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Yahoo の記録ミス（1日だけ価格が1/10になる等）を除去する。
    東証には値幅制限があり、数日で2.5倍/0.4倍以上動くことはまずないので、
    前後1週間の中央値から2.5倍以上離れた足を欠損扱いにする。"""
    c = panel["close"]
    med = c.rolling(7, center=True, min_periods=3).median()
    ratio = (c / med).abs()
    bad = (ratio > 2.5) | (ratio < 0.4) | (c <= 0)
    n = int(bad.to_numpy().sum())
    if n:
        log.info("異常値を除去: %d 本", n)
        panel = {k: v.mask(bad) for k, v in panel.items()}
    return panel
