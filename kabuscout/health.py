"""実際のペーパートレードが、過去ペーパートレードの想定どおりに動いているかの確認（実弾に進む前の安全装置）。

過去ペーパートレード（ウォークフォワード）でAIが選んだ取引を、決済順に並べた「k回連続の取引の損益の合計」の分布と、
直近 k 回（最大20回）の実際のペーパートレードの合計を比べる。
- 下位10%より悪い: やや下振れ（よくある不運の範囲だが注意）
- 下位2.5%より悪い: 想定外。相場の性質が変わったか、AIが壊れている可能性 → 実際の資金なら止めて見直す
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

MIN_TRADES = 10
WINDOW = 20


def expectation(trades: pd.DataFrame) -> dict:
    """過去ペーパートレードの取引（決済順の1取引あたり損益）を、判定用に保存する形にする。"""
    net = trades.sort_values("date")["net"].to_numpy()
    return dict(n=int(len(net)), mean=float(net.mean()), win=float((net > 0).mean()), nets=[round(float(x), 5) for x in net])


def _rolling_sums(nets: np.ndarray, k: int) -> np.ndarray:
    c = np.concatenate([[0.0], np.cumsum(nets)])
    return c[k:] - c[:-k]


def check(con, cfg: dict) -> dict:
    p = cfg["data"]["db_path"].parent / "backtest.json"
    exp = json.loads(p.read_text()).get("expect") if p.exists() else None
    pos = pd.read_sql("SELECT exit_date, entry, shares, pnl FROM paper_positions WHERE status='closed' ORDER BY exit_date, id", con)
    n = len(pos)
    if not exp:
        return dict(status="unknown", text="過去ペーパートレードの結果がまだありません", n=n)
    if n < MIN_TRADES:
        return dict(status="waiting", n=n, text=f"判定には決済{MIN_TRADES}回以上が必要（現在{n}回）")
    k = min(n, WINDOW)
    live = (pos["pnl"] / (pos["entry"] * pos["shares"])).to_numpy()[-k:]
    dist = _rolling_sums(np.asarray(exp["nets"]), k)
    pct = float((dist < live.sum()).mean())  # 過去の同じ回数の連続取引の中で、何%より良いか
    q10, q025, q50 = (float(np.quantile(dist, q)) for q in (0.10, 0.025, 0.5))
    if pct < 0.025:
        status, text = "alert", "想定外の不調（過去の下位2.5%より悪い）。実際の資金なら止めて見直す目安です"
    elif pct < 0.10:
        status, text = "caution", "やや下振れ（過去の下位10%）。よくある不運の範囲ですが注意"
    else:
        status, text = "ok", "想定どおりの範囲"
    return dict(status=status, text=text, n=n, k=k, live_sum=float(live.sum()), percentile=pct,
                typical=q50, low10=q10, low025=q025, live_win=float((live > 0).mean()), exp_win=exp["win"])
