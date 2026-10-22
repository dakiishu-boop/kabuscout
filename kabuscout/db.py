"""SQLite ストレージ。"""
from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS universe(
  code TEXT PRIMARY KEY, name TEXT, market TEXT, sector33 TEXT, sector17 TEXT,
  scale TEXT, updated TEXT
);
CREATE TABLE IF NOT EXISTS prices(
  code TEXT, date TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL,
  PRIMARY KEY(code, date)
);
CREATE INDEX IF NOT EXISTS idx_prices_date ON prices(date);

-- 実際に通知した候補（ライブ）。horizon 経過後に結果をラベル付けして成績を追跡する
CREATE TABLE IF NOT EXISTS signals(
  date TEXT, code TEXT, name TEXT, strategies TEXT, rule_score REAL, prob REAL,
  final_score REAL, rank INTEGER, model_version TEXT, close REAL, atr REAL,
  tp REAL, sl REAL, reasons TEXT, earnings TEXT,
  label INTEGER, ret REAL, exit_days INTEGER, labeled INTEGER DEFAULT 0,
  PRIMARY KEY(date, code)
);
CREATE TABLE IF NOT EXISTS models(
  version TEXT PRIMARY KEY, trained_at TEXT, data_end TEXT, n_train INTEGER,
  params TEXT, metrics TEXT, path TEXT, active INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS strategy_stats(
  date TEXT, strategy TEXT, n INTEGER, win_rate REAL, avg_ret REAL, weight REAL,
  PRIMARY KEY(date, strategy)
);
CREATE TABLE IF NOT EXISTS paper_positions(
  id INTEGER PRIMARY KEY AUTOINCREMENT, code TEXT, name TEXT, signal_date TEXT,
  status TEXT,  -- pending / open / closed / cancelled / archived（資金設定の変更前の記録）
  entry_date TEXT, entry REAL, shares INTEGER, tp REAL, sl REAL, atr REAL,
  exit_date TEXT, exit REAL, pnl REAL, reason TEXT, prob REAL, days_held INTEGER DEFAULT 0
);
-- ペーパートレードの毎日の評価額（含み損益込み）。最高値からの下落を測る
CREATE TABLE IF NOT EXISTS paper_equity(date TEXT PRIMARY KEY, equity REAL, capital REAL);
-- その日の通知内容（Webページの「過去の結果」表示用）
CREATE TABLE IF NOT EXISTS summaries(data_date TEXT PRIMARY KEY, json TEXT);
CREATE TABLE IF NOT EXISTS runs(
  data_date TEXT PRIMARY KEY, started TEXT, finished TEXT, status TEXT, message TEXT
);
"""


def connect(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(str(path), timeout=60)
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(SCHEMA)
    cols = {r[1] for r in con.execute("PRAGMA table_info(paper_positions)")}
    for col, typ in (("horizon", "INTEGER"), ("tp_atr", "REAL"), ("sl_atr", "REAL"), ("capital", "REAL")):
        if col not in cols:  # 売買ルールが途中で変わっても、注文した時のルールで手仕舞うため
            con.execute(f"ALTER TABLE paper_positions ADD COLUMN {col} {typ}")
    con.commit()
    return con
