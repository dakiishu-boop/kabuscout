"""tests/fixtures/prices.csv を作る（作り物の株価。乱数の種を固定しているので、何度作っても同じ）。
4銘柄 ＋ 指数（^N225）、2024-01-01 から320営業日。実データではない。"""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
dates = pd.bdate_range("2024-01-01", periods=320)
rows = []
for code, start, vol in (("^N225", 30000.0, 0.012), ("1001", 1500.0, 0.02), ("1002", 820.0, 0.025),
                         ("1003", 4200.0, 0.018), ("1004", 260.0, 0.03)):
    c = start * np.exp(np.cumsum(rng.normal(0.0004, vol, len(dates))))
    o = c * np.exp(rng.normal(0, vol / 3, len(dates)))
    h = np.maximum(o, c) * np.exp(np.abs(rng.normal(0, vol / 2, len(dates))))
    l = np.minimum(o, c) * np.exp(-np.abs(rng.normal(0, vol / 2, len(dates))))
    v = np.round(rng.lognormal(12, 0.5, len(dates)))
    rows.append(pd.DataFrame(dict(date=dates, code=code, open=o.round(1), high=h.round(1), low=l.round(1), close=c.round(1), volume=v)))
pd.concat(rows).to_csv(Path(__file__).with_name("prices.csv"), index=False)
