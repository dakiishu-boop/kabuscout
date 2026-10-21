"""コマンドライン: python -m kabuscout <command>

  daily        日次スクリーニング（初回は全銘柄の履歴取得と初期学習も自動で行う）
  daily --force       処理済みの日付でも再実行
  daily --retrain     再学習を強制
  daily --no-notify   通知しない
  train        再学習だけ実行
  notify-test  通知チャネルのテスト送信
  status       データ・モデル・成績の状況
  site         Webページだけ作り直す（site/index.html）
"""
from __future__ import annotations

import argparse
import logging
import sys
import warnings

import pandas as pd

from . import data, learner, notify, paper
from .config import load_config
from .db import connect


def main(argv=None):
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(prog="kabuscout")
    ap.add_argument("command", choices=["daily", "train", "notify-test", "status", "site"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--no-notify", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    cfg = load_config()

    if a.command == "daily":
        from .pipeline import run_daily
        run_daily(cfg, force=a.force, retrain=a.retrain, notify_on=not a.no_notify)

    elif a.command == "train":
        from .pipeline import build
        con = connect(cfg["data"]["db_path"])
        uni = data.refresh_universe(con, cfg)
        ind, _, ds, _ = build(con, cfg, uni)
        learner.train(con, cfg, ds, ind["close"].index[-1].strftime("%Y-%m-%d"))

    elif a.command == "notify-test":
        sent = notify.send_text(cfg, "KabuScout テスト通知", "通知の設定は正常です ✅")
        print("送信:", sent or "なし（config.yaml / .env を確認）")

    elif a.command == "site":
        from . import site
        print(site.build(connect(cfg["data"]["db_path"]), cfg))

    elif a.command == "status":
        con = connect(cfg["data"]["db_path"])
        q = lambda s: pd.read_sql(s, con)
        print(q("SELECT COUNT(DISTINCT code) AS 銘柄数, MIN(date) AS 開始, MAX(date) AS 最新, COUNT(*) AS 行数 FROM prices").to_string(index=False))
        print("\n[モデル]")
        print(q("SELECT version, data_end, n_train, active FROM models ORDER BY version DESC LIMIT 5").to_string(index=False))
        print("\n[通知した候補の答え合わせ（上位5件）]")
        print(q("SELECT COUNT(*) AS 件数, AVG(label) AS 勝率, AVG(ret) AS 平均リターン FROM signals WHERE labeled=1 AND rank<=5").to_string(index=False))
        print("\n[ペーパートレード]")
        print(paper.summary(con, cfg))
        print("\n[直近の実行]")
        print(q("SELECT * FROM runs ORDER BY data_date DESC LIMIT 5").to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
