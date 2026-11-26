"""コマンドライン: python -m usscout <command>

  fetch        銘柄表（S&P500 の構成銘柄と入れ替えの履歴）と株価を取り直す（初回は数分）
  backtest     過去の口座のシミュレーション（全戦略・物差し・感度・金庫）→ data/us_scout/report.json と report.md
  screen       今月の目標の比率と、その理由（モメンタムの順位など）を表示
  rebalance    目標の比率に合わせる注文を作る。既定は「表示だけ」（--execute で口座に送る。既定の口座は手元のペーパー口座）
  status       ペーパー口座の状況
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings

from . import config


def main(argv=None):
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(prog="usscout")
    ap.add_argument("command", choices=["fetch", "backtest", "screen", "rebalance", "status"])
    ap.add_argument("--config", help="設定ファイル（既定: usscout.yaml）")
    ap.add_argument("--strategy", help="戦略（既定: usscout.yaml の trading.strategy）")
    ap.add_argument("--broker", help="paper / alpaca（既定: usscout.yaml の trading.broker）")
    ap.add_argument("--execute", action="store_true", help="rebalance: 注文を実際に口座へ送る（無ければ表示だけ）")
    ap.add_argument("--force", action="store_true", help="rebalance: 今月すでに組み直していても実行する")
    ap.add_argument("--refresh", action="store_true", help="fetch: Wikipedia の銘柄表も取り直す")
    ap.add_argument("--quick", action="store_true", help="backtest: 感度・ランダム比較を省く")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    cfg = config.load(a.config)

    if a.command == "fetch":
        from . import data, universe
        cur, ch = universe.fetch_sp500(cfg["data"]["dir"], refresh=a.refresh)
        syms = universe.etf_symbols(cfg) + universe.sp500_symbols(cur, ch)
        p = data.download(syms, cfg["data"]["start"])
        data.save(p, cfg["data"]["dir"])
        print(f"保存しました: {p.close.shape[1]} 銘柄 / 要求 {len(syms)}、{p.dates[0].date()}〜{p.dates[-1].date()}")

    elif a.command == "backtest":
        from . import research
        rep = research.run_all(cfg, quick=a.quick)
        print(research.markdown(rep))

    elif a.command == "screen":
        from . import trader
        out = trader.screen(cfg, a.strategy)
        print(json.dumps(out, ensure_ascii=False, indent=1, default=str))

    elif a.command == "rebalance":
        from . import trader
        res = trader.rebalance(cfg, strategy=a.strategy, broker=a.broker, execute=a.execute, force=a.force)
        print(json.dumps(res, ensure_ascii=False, indent=1, default=str))
        if res.get("blocked"):
            sys.exit(2)

    elif a.command == "status":
        from . import trader
        print(json.dumps(trader.status(cfg, a.broker), ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
