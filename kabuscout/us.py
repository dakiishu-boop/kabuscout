"""米国株の研究（過去ペーパートレードのみ。ユーザーの依頼「投資の判断はどの市場でも共通」、ジャーナル 57）。

日本株と同じ仕組み（指標・検出ルール・3つの学習目標のAI・口座シミュレーション）を、設定を差し替えて米国株に当てる。
- 銘柄: NASDAQ の公開ファイル（nasdaqlisted / otherlisted）の普通株（ETF・ワラント・優先株・試験銘柄を除く）
- 株価・配当・分割: Yahoo（今と同じ）。決算日・アナリスト予想・サプライズ率: Yahoo（米国株は2002年ごろから全社分ある）
- 売買: 1株単位、呼値は1セント、SBI証券の米国株の手数料（約定代金の0.495%）を往復で差し引く
- 保存先: data/us/（日本株のデータとは分ける）
"""
from __future__ import annotations

import copy
import io
import logging
import re
import shutil

import pandas as pd
import requests

from . import data, earnings, risk

log = logging.getLogger(__name__)
LISTS = ["https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt", "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"]
SCREENER = "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=10000&download=true"
EXCLUDE = re.compile(r"Warrant|Right|Unit|Preferred|Depositary|Notes|Debenture|Trust Preferred|Acquisition Corp|%", re.I)


def config(cfg: dict) -> dict:
    """日本株の設定を米国株用に差し替える。"""
    c = copy.deepcopy(cfg)
    root = cfg["data"]["db_path"].parent / "us"
    root.mkdir(parents=True, exist_ok=True)
    c["data"].update(db_path=root / "kabuscout.db", price_dir=root / "prices", pool_path=root / "pool.parquet", model_dir=root / "models")
    c["universe"].update(market_proxy="^GSPC", min_turnover_jpy=5e6, min_price=5.0)  # 売買代金500万ドル/日以上・5ドル以上
    c["research"]["cost"] = 0.0099 + 0.001  # 手数料 0.495%×往復 ＋ 気配差・滑り
    c["paper_trading"].update(lot_size=1, capital_jpy=6700)  # 100万円 ≒ 6,700ドル
    # 日本株で採用済みの戦略の条件・売買ルールを、そのまま米国株にも当てる
    src, dst = cfg["data"]["db_path"].parent / "strategy_params.json", root / "strategy_params.json"
    if src.exists() and not dst.exists():
        shutil.copy(src, dst)
    m = cfg["data"]["db_path"].parent / "macro.parquet"  # 原油・金利など世界共通の値動き
    if m.exists():
        shutil.copy(m, root / "macro.parquet")
    return c


def use_us() -> None:
    data.SUFFIX = ""
    risk.TICK = "us"


def universe() -> pd.DataFrame:
    rows = []
    for u in LISTS:
        t = pd.read_csv(io.StringIO(requests.get(u, timeout=60).text), sep="|")
        t = t[~t.iloc[:, 0].astype(str).str.startswith("File Creation")]
        sym = t["Symbol"] if "Symbol" in t else t["ACT Symbol"]
        t = t.assign(code=sym.astype(str), name=t["Security Name"].astype(str))
        t = t[(t["ETF"] == "N") & (t["Test Issue"] == "N") & ~t["name"].str.contains(EXCLUDE)]
        t = t[t["code"].str.fullmatch(r"[A-Z]{1,5}")]  # 別クラス株（BRK.B など）は除く
        rows.append(t[["code", "name"]])
    u = pd.concat(rows).drop_duplicates("code")
    try:  # 業種: NASDAQ のスクリーナーの industry（約140）を東証33業種、sector（約12）を17業種の代わりに使う
        r = requests.get(SCREENER, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"}, timeout=60)
        s = pd.DataFrame(r.json()["data"]["rows"]).set_index("symbol")
    except Exception as e:
        log.warning("米国株の業種を取得できませんでした: %s", e)
        s = pd.DataFrame(columns=["industry", "sector"])
    ind, sec = (u["code"].map(s[k].replace("", None)) for k in ("industry", "sector"))
    # 業種が分からない銘柄は銘柄ごとに別の業種として扱う（同じ業種の保有数の制限を誤ってかけない）
    return u.assign(market="US", sector33=ind.fillna("?" + u["code"]), sector17=sec.fillna("?" + u["code"]),
                    scale="-").reset_index(drop=True)


def save_universe(cfg_us: dict, uni: pd.DataFrame) -> None:
    """過去ペーパートレードが業種を読む銘柄一覧の表を、米国株用のデータベースに書く。"""
    from .db import connect
    con = connect(cfg_us["data"]["db_path"])
    con.execute("DELETE FROM universe")
    con.executemany("INSERT INTO universe(code, name, market, sector33, sector17, scale, updated) VALUES (?,?,?,?,?,?,date('now'))",
                    uni[["code", "name", "market", "sector33", "sector17", "scale"]].itertuples(index=False))
    con.commit()


def prepare(cfg_us: dict, earnings_minutes: float | None = None) -> pd.DataFrame:
    """銘柄一覧・株価・決算日を揃える（初回は株価に約1時間、決算日に数時間）。"""
    use_us()
    uni = universe()
    save_universe(cfg_us, uni)
    store = data.PriceStore(cfg_us["data"]["price_dir"])
    log.info("米国株: %d銘柄", len(uni))
    data.update_prices(store, uni["code"].tolist() + [cfg_us["universe"]["market_proxy"]], cfg_us)
    c = store.load(fields=["close", "volume"])
    tv = (c["close"] * c["volume"]).rolling(20, min_periods=10).mean().max()
    liquid = [k for k in uni["code"] if tv.get(k, 0) >= cfg_us["universe"]["min_turnover_jpy"]]
    log.info("売買代金の条件を一度でも満たした銘柄: %d", len(liquid))
    earnings.update(cfg_us, liquid, max_minutes=earnings_minutes)
    return uni
