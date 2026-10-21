"""Webページ生成。DB の中身を1枚の静的 HTML（site/index.html）に書き出す。
GitHub Actions がこれを GitHub Pages に公開する。外部ライブラリなしで、スマホでも軽く表示できる。"""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from . import regimes
from .config import ROOT
from .strategies import STRATEGIES

TEMPLATE = Path(__file__).with_name("site_template.html")


def _records(df: pd.DataFrame) -> list[dict]:
    out = []
    for row in df.to_dict("records"):
        out.append({k: (None if isinstance(v, float) and (math.isnan(v) or math.isinf(v)) else v) for k, v in row.items()})
    return out


def _prices(cfg: dict, codes: list[str], n: int = 130) -> dict:
    from .data import PriceStore
    if not codes:
        return {}
    start = (pd.Timestamp.today() - pd.Timedelta(days=int((n + 80) * 1.6))).strftime("%Y-%m-%d")
    p = PriceStore(cfg["data"]["price_dir"]).load(start=start, codes=codes)
    out = {}
    for c in codes:
        if c not in p["close"].columns:
            continue
        df = pd.DataFrame({f: p[f][c] for f in ("open", "high", "low", "close")}).dropna()
        if df.empty:
            continue
        df["ma25"] = df["close"].rolling(25).mean()
        df["ma75"] = df["close"].rolling(75).mean()
        df = df.tail(n).astype(float).round(1)
        rec = {k: [None if v != v else v for v in df[k].tolist()] for k in df.columns}
        rec["date"] = [d.strftime("%Y-%m-%d") for d in df.index]
        out[c] = rec
    return out


def _sources(cfg: dict) -> dict:
    from . import earnings, flows
    ev = earnings.load(cfg)
    return dict(flows=flows.history_days(cfg),
                earnings=None if ev is None else dict(codes=int(ev["code"].nunique()), events=int(len(ev)),
                                                      since=str(ev["date"].min())[:10]))


def collect(con, cfg: dict) -> dict:
    q = lambda s, p=(): pd.read_sql(s, con, params=p)
    sums = q("SELECT data_date, json FROM summaries ORDER BY data_date DESC LIMIT 60")
    summaries = [json.loads(s) for s in sums["json"]]
    signals = q("SELECT date, code, name, strategies, prob, rank, close, tp, sl, reasons, earnings, label, ret, "
                "exit_days, labeled FROM signals ORDER BY date DESC, rank LIMIT 1500")
    positions = q("SELECT * FROM paper_positions ORDER BY id DESC")
    models = q("SELECT version, trained_at, data_end, n_train, metrics, active FROM models ORDER BY version DESC LIMIT 30")
    models["metrics"] = models["metrics"].map(json.loads)
    stats = q("SELECT * FROM strategy_stats WHERE date=(SELECT MAX(date) FROM strategy_stats)")

    latest = summaries[0]["date"] if summaries else None
    codes = set(signals.loc[signals["date"] == latest, "code"]) if latest else set()
    codes |= set(positions.loc[positions["status"].isin(["open", "pending"]), "code"])
    prices = _prices(cfg, sorted(codes))

    # 保有中の現在値
    last_close = {c: prices[c]["close"][-1] for c in positions.loc[positions["status"] == "open", "code"] if c in prices}

    base = cfg["data"]["db_path"].parent
    load_json = lambda name: json.loads((base / name).read_text()) if (base / name).exists() else None
    return dict(
        backtest=load_json("backtest.json"), research=load_json("research.json"), events=regimes.events_table(),
        data_sources=_sources(cfg),
        generated=datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y-%m-%d %H:%M"),
        summaries=summaries, signals=_records(signals), positions=_records(positions),
        models=_records(models), stats=_records(stats), prices=prices, last_close=last_close,
        strategies={s.key: dict(name=s.name, desc=s.desc) for s in STRATEGIES},
        config=dict(capital=cfg["paper_trading"]["capital_jpy"], horizon=cfg["labeling"]["horizon_days"],
                    tp_atr=cfg["labeling"]["tp_atr"], sl_atr=cfg["labeling"]["sl_atr"],
                    max_positions=cfg["paper_trading"]["max_positions"],
                    crisis_weight=cfg["learning"].get("crisis_weight", 0.3)),
    )


def build(con, cfg: dict, out_dir: Path | None = None) -> Path:
    out_dir = out_dir or ROOT / "site"
    out_dir.mkdir(parents=True, exist_ok=True)
    data = json.dumps(collect(con, cfg), ensure_ascii=False, default=str, separators=(",", ":"))
    data = data.replace("</", "<\\/")  # <script> 内に安全に埋め込む
    html = TEMPLATE.read_text().replace("/*__DATA__*/null", data)
    path = out_dir / "index.html"
    path.write_text(html)
    (out_dir / ".nojekyll").write_text("")
    return path
