"""コマンドライン: python -m kabuscout <command>

  daily        日次スクリーニング（初回は全銘柄の履歴取得と初期学習も自動で行う）
  daily --force       処理済みの日付でも再実行
  daily --retrain     再学習を強制
  daily --no-notify   通知しない
  train        再学習だけ実行
  notify-test  通知チャネルのテスト送信
  status       データ・モデル・成績の状況
  site         Webページだけ作り直す（site/index.html）
  research     研究ループ（過去データで戦略の条件を調整）→ 再学習 → 過去ペーパートレード → 通知
  backtest     過去ペーパートレードだけ実行
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
    ap.add_argument("command", choices=["daily", "train", "notify-test", "status", "site", "research", "backtest", "edinet", "jquants", "evaluate-jquants", "survivorship", "tdnet", "evaluate-edinet", "prices", "evaluate-forecast", "tdnet-xbrl"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--no-notify", action="store_true")
    ap.add_argument("--no-collect", action="store_true", help="追加データ（決算・需給・適時開示）の収集をしない")
    ap.add_argument("--trials", type=int)
    ap.add_argument("--budget-min", type=float)
    ap.add_argument("--probe", action="store_true", help="J-Quants: どのデータが取れるかを確かめ直す")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    cfg = load_config()

    if a.command == "daily":
        from .pipeline import run_daily
        run_daily(cfg, force=a.force, retrain=a.retrain, notify_on=not a.no_notify, collect=not a.no_collect)

    elif a.command == "train":
        from . import pool
        from .pipeline import dataset
        con = connect(cfg["data"]["db_path"])
        pl = pool.load(cfg)
        if pl is None:
            sys.exit("シグナル表がありません。先に daily を実行してください")
        learner.train(con, cfg, dataset(cfg, pl), pl["date"].max().strftime("%Y-%m-%d"))

    elif a.command == "notify-test":
        sent = notify.send_text(cfg, "KabuScout テスト通知", "通知の設定は正常です ✅")
        print("送信:", sent or "なし（config.yaml / .env を確認）")

    elif a.command in ("research", "backtest"):
        from .pipeline import run_research
        run_research(cfg, research=a.command == "research", trials=a.trials, budget_min=a.budget_min,
                     notify_on=not a.no_notify, collect=not a.no_collect)

    elif a.command == "prices":  # 株価・配当・分割の取得だけ（分割の履歴を揃えるときなど、全期間の取り直しを毎日の実行とは別に済ませる）
        from .pipeline import store_of
        con = connect(cfg["data"]["db_path"])
        uni = data.refresh_universe(con, cfg)
        data.update_prices(store_of(cfg), uni["code"].tolist() + [cfg["universe"]["market_proxy"]], cfg)
    elif a.command == "tdnet":  # 適時開示の過去分の取得だけ（決算好反応の判定に必要。クラウドの初回の取得を早く済ませる）
        from . import tdnet
        tdnet.update(cfg, max_minutes=a.budget_min or 60)
    elif a.command == "jquants":
        from . import jquants
        if a.probe or not (cfg["data"]["db_path"].parent / "jquants" / "probe.json").exists():
            jquants.probe(cfg)
        print(jquants.update_fins(cfg, max_minutes=a.budget_min or 20))
    elif a.command == "evaluate-edinet":  # EDINET の財務データが効くかの確認（診断＋過去ペーパートレード）
        from . import evaluate
        from .edinet import FIN_FEATURES
        evaluate.run(cfg, "edinet", FIN_FEATURES, start_year=2012)
    elif a.command == "evaluate-jquants":
        from . import evaluate, jquants
        evaluate.run(cfg, "jquants", jquants.JQ_FEATURES, start_year=2024)
    elif a.command == "survivorship":
        from . import survivorship
        r = survivorship.collect(cfg, max_minutes=a.budget_min or 60)
        if r.get("fetched") and r["fetched"] >= r.get("delisted", 0):
            survivorship.analyze(cfg)
    elif a.command == "tdnet-xbrl":  # 決算短信の数値（直近30日分の未取得）
        from . import tdnet_xbrl
        print(tdnet_xbrl.update(cfg, days=30, max_minutes=a.budget_min or 40))
    elif a.command == "evaluate-forecast":  # 自前の利益の予想（益回り）が効くかの確認
        from . import evaluate
        from .forecast import FC_FEATURES
        evaluate.run(cfg, "forecast", FC_FEATURES, start_year=2016)
    elif a.command == "edinet":
        from . import edinet
        print(edinet.update(cfg, max_minutes=a.budget_min or 30))
    if a.command in ("jquants", "edinet", "evaluate-jquants", "survivorship", "tdnet", "train", "evaluate-edinet", "prices", "evaluate-forecast", "tdnet-xbrl"):  # データ取得だけの実行でも Webページの公開手順が失敗しないように
        from . import site
        site.build(connect(cfg["data"]["db_path"]), cfg)

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
