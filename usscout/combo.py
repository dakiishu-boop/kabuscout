"""日本株のAI（kabuscout）と米国株（usscout）を組み合わせた口座のシミュレーション（円建て・税引き後）。

年利益を上げる仕組みとして根拠があるのは「売買の戦略」ではなく「資金の分け方」（docs/JOURNAL.md §61・§90）:
- 日本株のAI: 日本の小型株・下げ相場に強い（2008年 +20%、日経平均 −40%）。特定口座（毎年 20.315%）
- S&P500: 長期の成長。売買しないので NISA（非課税）に置ける
- 動きの似かたが小さい（相関 約0.4）ので、合わせると年率が上がり、最大下落はむしろ小さくなる

計算のしかた（事前に決めた設定。分け方は §61 と同じ 0・30・50・70・100%）:
- 日本株のAI: kabuscout の過去ペーパートレード（data/backtest.json の curve.ai、週ごと、円、コスト込み）
- 米国株: usscout の過去の口座（ドル、コスト込み）を毎日のドル円で円に（為替の手数料 1ドル25銭を最初と最後に）
- 毎年の最初の週に分け方を戻す。動かす金額に、日本株側 0.1%＋米国株側（手数料・気配差）＋為替（25銭 ÷ ドル円）のコスト
- 税金: 毎年末に、特定口座の側の利益に 20.315%（損失は3年繰越）。S&P500 を持ち続ける側は NISA で非課税
  （注意: NISA の今の制度は2024年から。2008〜2023年も今の制度だったと仮定している。年240万円の枠の範囲で、資金100万円なら足りる）
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics
from .config import ROOT

SHARES = (0.0, 0.3, 0.5, 0.7, 1.0)
US_PARTS = ("hold_spy", "trend_spy")


def load_jp_curve(path: Path, key: str = "ai") -> pd.Series:
    d = json.loads(Path(path).read_text())
    c = pd.Series({pd.Timestamp(a): float(b) for a, b in d["curve"][key]}).sort_index()
    return c


def combine(jp: pd.Series, us: pd.Series, share: float, us_taxed: bool, cfg: dict, fx: pd.Series) -> pd.Series:
    """週ごとの2つの口座（円）を、毎年初めに share（米国の割合）へ戻しながら合わせる。税引き後の評価額を返す。"""
    rate, carry_y = cfg["tax"]["rate"], cfg["tax"]["carry_years"]
    c = cfg["costs"]
    idx = jp.index
    rj, ru = jp.pct_change().fillna(0.0), us.reindex(idx).ffill().pct_change().fillna(0.0)
    f = fx.reindex(idx, method="ffill").bfill()
    move_cost_us = c.get("commission", 0) + c.get("spread", 0)
    total = float(jp.iloc[0])
    vj, vu = total * (1 - share), total * share
    base = {"jp": vj, "us": vu}
    losses = {"jp": [], "us": []}
    out, year = [], idx[0].year
    for d in idx:
        if d.year != year:  # 年が変わった最初の週: 前の年の税金 → 分け方を戻す
            for k, taxed in (("jp", True), ("us", us_taxed)):
                v = vj if k == "jp" else vu
                gain = v - base[k]
                losses[k] = [[y, l] for y, l in losses[k] if year - y <= carry_y and l > 0]
                tax = 0.0
                if taxed and gain > 0:
                    left = gain
                    for e in losses[k]:
                        t = min(e[1], left)
                        e[1] -= t
                        left -= t
                    tax = left * rate
                elif taxed and gain < 0:
                    losses[k].append([year, -gain])
                if k == "jp":
                    vj -= tax
                else:
                    vu -= tax
            tot = vj + vu
            move = share * tot - vu  # プラスなら日本 → 米国
            cost = abs(move) * (0.001 + move_cost_us + c.get("fx_spread_jpy", 0) / float(f.at[d])) if 0 < share < 1 else 0.0
            vu += move - (cost if move > 0 else 0)
            vj -= move + (cost if move < 0 else 0)
            base = {"jp": vj, "us": vu}
            year = d.year
        vj *= 1 + rj.at[d]
        vu *= 1 + ru.at[d]
        out.append(vj + vu)
    # 最後の年（途中まで）の税金も、最後の週に差し引く（metrics.after_tax と同じ扱い）
    for k, taxed in (("jp", True), ("us", us_taxed)):
        v = vj if k == "jp" else vu
        gain = v - base[k]
        if taxed and gain > 0:
            left = gain
            for e in [[y, l] for y, l in losses[k] if year - y <= carry_y and l > 0]:
                left -= min(e[1], left)
            out[-1] -= left * rate
    return pd.Series(out, index=idx)


def run(cfg: dict, jp_path: Path | None = None) -> dict:
    from . import research
    jp_path = Path(jp_path or ROOT / "data" / "backtest.json")
    jp = load_jp_curve(jp_path)
    lab = research.Lab(cfg)
    end = min(jp.index[-1], lab.p.dates[-1])
    jp = jp.loc[:end]
    fx = lab.p.close["JPY=X"]
    rep = {"jp_source": str(jp_path.name), "start": str(jp.index[0].date()), "end": str(end.date()), "rows": []}
    # 日本株のAIだけ（特定口座）
    jp_tax = metrics.after_tax(jp, cfg["tax"]["rate"], cfg["tax"]["carry_years"])
    rep["jp_alone"] = {"cagr": metrics.cagr(jp), "after_tax": metrics.cagr(jp_tax), "max_dd": metrics.max_dd(jp),
                       "worst_year": float(metrics.yearly(jp).min())}
    for us_name in US_PARTS:
        eq = lab.run(us_name, "net_mmf", start=str(jp.index[0].date()), end=str(end.date())).equity
        us_jpy = metrics.to_jpy(eq, fx, cfg["costs"].get("fx_spread_jpy", 0)) * float(jp.iloc[0])
        taxed = not us_name.startswith("hold_")
        for sh in SHARES:
            pre = combine(jp, us_jpy, sh, us_taxed=taxed, cfg=dict(cfg, tax=dict(cfg["tax"], rate=0.0)), fx=fx)
            post = combine(jp, us_jpy, sh, us_taxed=taxed, cfg=cfg, fx=fx)
            y = metrics.yearly(post)
            rep["rows"].append({"us": us_name, "us_share": sh, "us_account": "特定口座" if taxed else "NISA",
                                "cagr": metrics.cagr(pre), "after_tax": metrics.cagr(post), "max_dd": metrics.max_dd(post),
                                "worst_year": float(y.min()), "pos_years": float((y > 0).mean())})
    root = cfg["data"]["dir"]
    (root / "combo.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    (root / "combo.md").write_text(markdown(rep))
    return rep


def markdown(rep: dict) -> str:
    p = lambda v: f"{v:+.1%}"  # noqa: E731
    j = rep["jp_alone"]
    out = [f"# 日本株のAI＋米国株の組み合わせ（円建て・{rep['start']}〜{rep['end']}・毎年初めに分け方を戻す）", "",
           f"日本株のAIだけ（特定口座）: 年率 {p(j['cagr'])}・税引き後 {p(j['after_tax'])}・最大下落 {p(j['max_dd'])}・最悪の年 {p(j['worst_year'])}", "",
           "| 米国株の側 | 米国の割合 | 年率（税引き前） | 税引き後 | 最大下落 | 最悪の年 | プラスの年 |", "|---|---|---|---|---|---|---|"]
    name = {"hold_spy": "S&P500を持ち続ける（NISA）", "trend_spy": "SPY＋10か月平均の逃避（特定口座）"}
    for r in rep["rows"]:
        out.append(f"| {name[r['us']]} | {r['us_share']:.0%} | {p(r['cagr'])} | {p(r['after_tax'])} | {p(r['max_dd'])} | {p(r['worst_year'])} | {r['pos_years']:.0%} |")
    out += ["", "米国の割合 0% は日本株のAIだけ、100% は米国株だけ。日本株のAIの成績は kabuscout の過去ペーパートレード（ウォークフォワード）。",
            "NISA は今の制度が2008年からあったと仮定。税金は毎年末に利益の20.315%（損失3年繰越・含み益にも課税＝厳しめ）。"]
    return "\n".join(out)


def allocation(jp_value_jpy: float, us_value_usd: float, usdjpy: float, share: float) -> dict:
    """今の2つの口座の額から、分け方を戻すために動かす金額（年に1回）。プラスなら日本株の口座 → 米国株（NISA）へ。"""
    us_jpy = us_value_usd * usdjpy
    total = jp_value_jpy + us_jpy
    move = share * total - us_jpy
    return {"total_jpy": round(total), "us_target_jpy": round(share * total), "move_jpy": round(move),
            "move_usd": round(move / usdjpy, 2), "direction": "日本株 → 米国株" if move > 0 else "米国株 → 日本株"}
