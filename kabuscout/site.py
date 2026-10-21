"""Webページ生成。DB の中身を1枚の静的 HTML（site/index.html）に書き出す。
GitHub Actions がこれを GitHub Pages に公開する。外部ライブラリなしで、スマホでも軽く表示できる。"""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from .config import ROOT
from .strategies import STRATEGIES

TEMPLATE = Path(__file__).with_name("site_template.html")


def _records(df: pd.DataFrame) -> list[dict]:
    out = []
    for row in df.to_dict("records"):
        out.append({k: (None if isinstance(v, float) and (math.isnan(v) or math.isinf(v)) else v) for k, v in row.items()})
    return out


def _prices(con, codes: list[str], n: int = 130) -> dict:
    out = {}
    for c in codes:
        df = pd.read_sql("SELECT date, open, high, low, close, volume FROM prices WHERE code=? ORDER BY date DESC LIMIT ?",
                         con, params=(c, n + 75)).iloc[::-1]
        if df.empty:
            continue
        cl = df["close"]
        df["ma25"] = cl.rolling(25).mean().round(1)
        df["ma75"] = cl.rolling(75).mean().round(1)
        df = df.tail(n)
        out[c] = {k: df[k].round(1).tolist() if k != "date" else df[k].tolist()
                  for k in ("date", "open", "high", "low", "close", "ma25", "ma75")}
    return out


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
    prices = _prices(con, sorted(codes))

    # 保有中の現在値
    last_close = {}
    for c in positions.loc[positions["status"] == "open", "code"]:
        r = con.execute("SELECT close, date FROM prices WHERE code=? ORDER BY date DESC LIMIT 1", (c,)).fetchone()
        if r:
            last_close[c] = r[0]

    return dict(
        generated=datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y-%m-%d %H:%M"),
        summaries=summaries, signals=_records(signals), positions=_records(positions),
        models=_records(models), stats=_records(stats), prices=prices, last_close=last_close,
        strategies={s.key: dict(name=s.name, desc=s.desc) for s in STRATEGIES},
        config=dict(capital=cfg["paper_trading"]["capital_jpy"], horizon=cfg["labeling"]["horizon_days"],
                    tp_atr=cfg["labeling"]["tp_atr"], sl_atr=cfg["labeling"]["sl_atr"],
                    max_positions=cfg["paper_trading"]["max_positions"]),
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
