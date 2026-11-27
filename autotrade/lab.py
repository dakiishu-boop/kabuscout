"""研究用の入口（GitHub Actions の autotrade-lab で実行。結果はログに出す）。"""
from __future__ import annotations

import sys


def probe() -> None:
    import json
    import sqlite3
    from pathlib import Path

    import pandas as pd
    import pyarrow.parquet as pq

    root = Path("data")
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.stat().st_size > 200_000:
            print(f"FILE {p} {p.stat().st_size/1e6:.1f}MB")
    pr = root / "prices"
    for f in sorted(pr.glob("*.parquet")):
        s = pq.read_schema(f)
        md = pq.read_metadata(f)
        print(f"PRICES {f.name} cols={len(s.names)} rows={md.num_rows}")
    c = pd.read_parquet(pr / "close.parquet")
    c.index = pd.to_datetime(c.index)
    print("CLOSE range", c.index.min(), c.index.max(), c.shape)
    print("CLOSE sample cols", [x for x in c.columns[:5]], [x for x in c.columns if x.startswith("^") or not x[:1].isdigit()][:40])
    last = c.iloc[-1].notna().sum()
    print("codes with last-day price", int(last))
    for y in (2005, 2008, 2012, 2016, 2019, 2022, 2025):
        print("codes alive", y, int(c[c.index.year == y].notna().any().sum()))
    con = sqlite3.connect("data/kabuscout.db")
    print("TABLES", [r[0] for r in con.execute("select name from sqlite_master where type='table'")])
    u = pd.read_sql("select * from universe", con)
    print("UNIVERSE", u.shape, list(u.columns), u.head(3).to_dict("records"))
    for t in ("paper_positions", "paper_trades", "signals"):
        try:
            print(t, pd.read_sql(f"select * from {t} limit 2", con).to_dict("records"))
        except Exception as e:
            print(t, "ERR", e)
    for f in ("pool.parquet",):
        p = root / f
        if p.exists():
            m = pq.read_metadata(p)
            print("POOL rows", m.num_rows, pq.read_schema(p).names)
    jq = root / "jquants" / "delisted_bars.parquet"
    if jq.exists():
        d = pd.read_parquet(jq)
        print("DELISTED", d.shape, list(d.columns), d.head(2).to_dict("records"))
    for f in ("strategy_params.json",):
        p = root / f
        if p.exists():
            print(f, p.read_text()[:1500])


def main(argv: list[str]) -> None:
    cmd = argv[0] if argv else "probe"
    if cmd == "probe":
        probe()
    elif cmd == "bench":
        from . import bench
        bench.main(argv[1:])
    elif cmd == "improve":
        from . import improve
        improve.main(argv[1:])
    elif cmd == "verify":
        from . import verify
        verify.main(argv[1:])
    elif cmd == "live":
        # 本番のデータで毎日の運用を1回通す（通知なし。このジョブの中だけの口座で、どこにも保存しない）
        import time
        from . import __main__ as cli
        t0 = time.time()
        cli.main(["daily", "--no-notify"])
        print(f"LIVE daily (with training) {time.time() - t0:.0f}s")
        t0 = time.time()
        cli.main(["daily", "--no-notify"])
        print(f"LIVE daily (model cached) {time.time() - t0:.0f}s")
    else:
        from . import research
        research.main(argv)


if __name__ == "__main__":
    main(sys.argv[1:])
