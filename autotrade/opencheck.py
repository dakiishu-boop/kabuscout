"""Yahoo の始値と、取引所の公式の始値（J-Quants の日足の O）のずれ（ユーザーの依頼、2026-11-27）。

過去の検証は Yahoo の始値で約定したとみなしている。寄付の成行は公式の始値で約定するので、2つがずれていれば、その分は
検証が実際と違う（コストの④「データのずれ」）。J-Quants の Free プランで取れる約2年（約12週遅れ）から日付を選んで比べる。
キーは環境変数 JQUANTS_API_KEY（GitHub の Secrets）。値はログに出さない。公式の値そのものも出さず、ずれの統計だけを出す。
"""
from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

log = logging.getLogger("autotrade.opencheck")


def main(argv: list[str]) -> None:
    import warnings
    from pathlib import Path
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")
    from kabuscout import jquants
    from kabuscout.config import load_config
    from . import data as D
    from . import features as F
    cfg = load_config(Path(argv[argv.index("--config") + 1]) if "--config" in argv else Path("config.yaml"))
    n_dates = int(argv[argv.index("--dates") + 1]) if "--dates" in argv else 40
    if not jquants._key():
        print("OPEN JQUANTS_API_KEY がありません")
        return
    cov = jquants.coverage(cfg)
    print(f"OPEN J-Quants で取れる期間: {cov}")
    if cov is None:
        return
    lo, hi = cov
    start = (lo - pd.Timedelta(days=420)).strftime("%Y-%m-%d")
    px, mkt, fac, splits, div = D.load_panel(cfg, start)
    cal = px["open"].index
    raw_o = px["open"] * fac
    raw_c = px["close"] * fac
    turn20 = (px["close"] * px["volume"]).rolling(20, min_periods=15).mean()
    days = cal[(cal >= lo) & (cal <= hi)]
    pick = days[np.linspace(0, len(days) - 1, min(n_dates, len(days))).astype(int)]
    rows = []
    t0 = time.time()
    for d in pick:
        st, data = jquants._get(cfg, "/equities/bars/daily", {"date": d.strftime("%Y%m%d")})
        if st != 200 or not data:
            print(f"OPEN {d.date()} 取得できず（{st}）")
            continue
        j = pd.DataFrame(data)
        j["code"] = j["Code"].astype(str).str[:4]
        for k in ("O", "C", "Vo", "UL", "LL"):
            j[k] = pd.to_numeric(j[k], errors="coerce")
        j = j.drop_duplicates("code").set_index("code")
        codes = [c for c in j.index if c in raw_o.columns]
        y_o, y_c = raw_o.loc[d, codes], raw_c.loc[d, codes]
        t = turn20.loc[d, codes]
        rows.append(pd.DataFrame({"date": d, "code": codes, "y_open": y_o.to_numpy(), "j_open": j.loc[codes, "O"].to_numpy(),
                                  "y_close": y_c.to_numpy(), "j_close": j.loc[codes, "C"].to_numpy(),
                                  "j_vol": j.loc[codes, "Vo"].to_numpy(), "ul": j.loc[codes, "UL"].to_numpy(),
                                  "ll": j.loc[codes, "LL"].to_numpy(), "turn20": t.to_numpy()}))
    print(f"OPEN 取得 {len(rows)}日・{time.time() - t0:.0f}s")
    if not rows:
        return
    x = pd.concat(rows, ignore_index=True)
    x = x[(x["j_open"] > 0) & (x["y_open"] > 0)]
    x["diff_open"] = x["y_open"] / x["j_open"] - 1
    x["diff_close"] = x["y_close"] / x["j_close"] - 1
    elig = x["turn20"] >= 1e8
    for name, m in (("全銘柄", x["code"].notna()), ("売買代金1億円以上（売買の対象）", elig),
                    ("売買代金1〜3億円", elig & (x["turn20"] < 3e8)), ("売買代金10億円以上", x["turn20"] >= 1e9)):
        g = x[m]
        a = g["diff_open"].abs()
        print(f"OPEN [{name}] {len(g):,}件: 始値が一致 {(a < 1e-6).mean() * 100:.2f}%・0.1%以内 {(a <= 0.001).mean() * 100:.2f}%・"
              f"1%超ずれ {(a > 0.01).mean() * 100:.3f}%・平均のずれ {g['diff_open'].mean() * 100:+.4f}%・平均の大きさ {a.mean() * 100:.4f}%・"
              f"99%点 {a.quantile(0.99) * 100:.3f}%・最大 {a.max() * 100:.1f}% | 終値が一致 {(g['diff_close'].abs() < 1e-6).mean() * 100:.2f}%")
    big = x[elig & (x["diff_open"].abs() > 0.01)]
    print(f"OPEN 売買の対象で始値が1%超ずれた {len(big)}件のうち、分割の前後と思われる（終値もずれている）もの "
          f"{(big['diff_close'].abs() > 0.01).sum()}件")
    print(f"OPEN 売買の対象で、その日にストップ高（UL=1）{(x.loc[elig, 'ul'] == 1).mean() * 100:.3f}%・"
          f"ストップ安（LL=1）{(x.loc[elig, 'll'] == 1).mean() * 100:.3f}%・出来高0 {(x.loc[elig, 'j_vol'] <= 0).mean() * 100:.3f}%")
    # 往復のコストへの影響の目安: 買いと売りは別の日なので、ずれの平均（向きつき）×2 と、大きさの平均
    g = x[elig]
    print(f"OPEN 往復への影響の目安（売買の対象）: 平均のずれ×2 = {g['diff_open'].mean() * 2 * 100:+.4f}%（Yahoo が高ければ検証は買いを不利に・売りを有利に見積もる。"
          f"向きは買いと売りで打ち消し合う）・ずれの大きさの平均 {g['diff_open'].abs().mean() * 100:.4f}%")
