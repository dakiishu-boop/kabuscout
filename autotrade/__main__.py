"""使い方:
  python -m autotrade daily      # 毎日の引け後: 約定の反映・時価評価・翌朝の注文（ペーパートレード）
  python -m autotrade train      # AIを全期間で学び直す（毎週）
  python -m autotrade status     # 口座の状態を表示
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path


def main(argv: list[str]) -> None:
    import warnings
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")
    from kabuscout.config import load_config
    from . import live
    cmd = argv[0] if argv else "daily"
    cfg = load_config(Path(argv[argv.index("--config") + 1]) if "--config" in argv else Path("config.yaml"))
    s = live.load_settings()
    if cmd == "daily":
        live.daily(cfg, s, notify="--no-notify" not in argv)
    elif cmd == "train":
        live.train(cfg, s, live.usable_end(cfg)[0])  # 株価が揃っていない最新日は使わない
    elif cmd == "status":
        p = live.state_dir(cfg) / "report.md"
        print(p.read_text() if p.exists() else "まだ実行されていません")
    else:
        print(__doc__)
        sys.exit(2)


if __name__ == "__main__":
    main(sys.argv[1:])
