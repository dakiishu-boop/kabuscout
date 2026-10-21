"""日次レポート。1つの report 辞書から、Markdown・Discord カード・ntfy 文面・Webページ用データを作る。"""
from __future__ import annotations

import pandas as pd

from .strategies import STRATEGY_NAMES

WEEKDAYS = "月火水木金土日"
MOOD_COLOR = {"強気": 0xD9534F, "中立": 0xF0AD4E, "弱気": 0x337AB7}  # 日本式: 上げ=赤
MEDALS = ["🥇", "🥈", "🥉"]


def market_stats(ind: dict) -> dict:
    r = {k: float(ind[k].iloc[-1].iloc[0]) for k in ("mkt_dev75", "mkt_ret20", "breadth25")}
    r["mood"] = "強気" if r["mkt_dev75"] > 0 and r["breadth25"] > 0.5 else "弱気" if r["mkt_dev75"] < 0 else "中立"
    return r


def edge_label(m: dict) -> str:
    lift = m["daily_top_ret"] - m["base_ret"]
    return "優位性あり" if lift >= 0.005 else "わずかに優位" if lift > 0 else "まだ優位性なし（ルール中心で順位付け）"


def build(cfg, con, ind, top, n_signals, n_scanned, bundle, weights, pt, data_date) -> dict:
    d = pd.Timestamp(data_date)
    picks = []
    for _, r in top.iterrows():
        picks.append(dict(
            rank=int(r["rank"]), code=r["code"], name=r["name"],
            strategies=[STRATEGY_NAMES.get(k, k) for k in r["strategies"].split(",")],
            prob=None if pd.isna(r["prob"]) else float(r["prob"]),
            close=float(r["close"]), tp=float(r["tp"]), sl=float(r["sl"]),
            earnings_days=None if pd.isna(r["earnings_days"]) else int(r["earnings_days"]),
            reasons=r["reasons"],
        ))
    live = pd.read_sql("SELECT label, ret FROM signals WHERE labeled=1 AND rank<=5", con)
    model = None
    if bundle is not None:
        m = bundle["metrics"]
        model = dict(version=bundle["version"], edge=edge_label(m), top_ret=m["daily_top_ret"],
                     top_win=m["daily_top_win"], base_ret=m["base_ret"], base_win=m["base_win"])
    return dict(
        date=data_date, date_label=f"{d.month}/{d.day}({WEEKDAYS[d.weekday()]})",
        site_url=cfg.get("site", {}).get("url", ""),
        market=market_stats(ind), n_scanned=n_scanned, n_signals=n_signals, picks=picks, model=model,
        live=dict(n=len(live), win=float(live["label"].mean()), ret=float(live["ret"].mean())) if len(live) >= 5 else None,
        hot=[(STRATEGY_NAMES.get(k, k), w) for k, w in sorted(weights.items(), key=lambda t: -t[1])[:3]],
        paper=pt,
    )


def _pct(a, b):
    return a / b - 1


def _earn(p, limit=14):
    e = p["earnings_days"]
    return f"⚠️決算まで{e}日" if e is not None and e <= limit else ""


# ---------------------------------------------------------------- Markdown（ログ・レポートファイル用）
def to_markdown(rep: dict) -> str:
    m = rep["market"]
    L = [f"# KabuScout {rep['date_label']} 引け後スクリーニング", "",
         f"地合い **{m['mood']}**（TOPIX 75日線乖離 {m['mkt_dev75']:+.1%} / 20日 {m['mkt_ret20']:+.1%} / "
         f"25日線より上の銘柄 {m['breadth25']:.0%}）", f"{rep['n_scanned']:,}銘柄を走査 → シグナル{rep['n_signals']}件", ""]
    for p in rep["picks"]:
        prob = f" AI {p['prob']:.0%}" if p["prob"] is not None else ""
        L.append(f"{p['rank']}. **{p['code']} {p['name']}**【{'・'.join(p['strategies'])}】{prob} {_earn(p)}")
        L.append(f"   終値 {p['close']:,.0f} → 利確 {p['tp']:,.0f} ({_pct(p['tp'], p['close']):+.1%}) / "
                 f"損切 {p['sl']:,.0f} ({_pct(p['sl'], p['close']):+.1%})")
        L.append(f"   {p['reasons']}")
    if rep["model"]:
        mo = rep["model"]
        L += ["", f"AI v{mo['version']}（{mo['edge']}）: 上位推奨の期待値 {mo['top_ret']:+.2%} / 勝率 {mo['top_win']:.0%}"
                  f"（全シグナル平均 {mo['base_ret']:+.2%} / {mo['base_win']:.0%}）"]
    if rep["paper"]:
        pt = rep["paper"]
        L.append(f"ペーパートレード: 評価額 {pt['equity']:,.0f}円（実現 {pt['realized']:+,.0f} / 含み {pt['unrealized']:+,.0f}）")
        L += [f"  ・{e}" for e in pt["events"]]
    return "\n".join(L)


# ---------------------------------------------------------------- Discord（カード表示）
def discord_payload(rep: dict, max_picks: int = 10) -> dict:
    m, url = rep["market"], rep["site_url"] or None
    head = dict(
        title=f"📈 KabuScout {rep['date_label']} 引け後スクリーニング",
        url=url, color=MOOD_COLOR[m["mood"]],
        description=f"**{rep['n_scanned']:,}** 銘柄を走査 → シグナル **{rep['n_signals']}** 件"
                    + (f"\n🌐 [Webで詳しく見る（チャート・過去の結果）]({url})" if url else ""),
        fields=[
            dict(name="地合い", value=f"**{m['mood']}**", inline=True),
            dict(name="TOPIX 75日線", value=f"{m['mkt_dev75']:+.1%}", inline=True),
            dict(name="25日線より上", value=f"{m['breadth25']:.0%}", inline=True),
        ],
    )
    fields = []
    for p in rep["picks"][:max_picks]:
        mark = MEDALS[p["rank"] - 1] if p["rank"] <= 3 else f"`{p['rank']:>2}`"
        prob = f"AI勝率 **{p['prob']:.0%}**　" if p["prob"] is not None else ""
        yahoo = f"https://finance.yahoo.co.jp/quote/{p['code']}.T"
        fields.append(dict(
            name=f"{mark} {p['code']} {p['name']}　{_earn(p)}",
            value=(f"🏷️ {' / '.join(p['strategies'])}\n"
                   f"```\n終値 {p['close']:>8,.0f}\n利確 {p['tp']:>8,.0f} ({_pct(p['tp'], p['close']):+.1%})\n"
                   f"損切 {p['sl']:>8,.0f} ({_pct(p['sl'], p['close']):+.1%})```"
                   f"{prob}[Yahoo]({yahoo})\n*{p['reasons']}*")[:1024],
            inline=False))
    picks = dict(title=f"🎯 注目銘柄 TOP{len(fields)}", color=0xE377C2, fields=fields)

    embeds = [head, picks]
    pt = rep["paper"]
    if pt:
        wr = f"{pt['win_rate']:.0%}" if pt["win_rate"] is not None else "-"
        embeds.append(dict(
            title="💼 ペーパートレード（仮想300万円）", color=0x2CA02C,
            description="\n".join(f"• {e}" for e in pt["events"]) or "本日の売買なし",
            fields=[
                dict(name="評価額", value=f"{pt['equity']:,.0f}円", inline=True),
                dict(name="実現損益", value=f"{pt['realized']:+,.0f}円", inline=True),
                dict(name="含み損益", value=f"{pt['unrealized']:+,.0f}円", inline=True),
                dict(name="保有 / 注文", value=f"{pt['open']} / {pt['pending']}", inline=True),
                dict(name="決済", value=f"{pt['closed']}回", inline=True),
                dict(name="勝率", value=wr, inline=True),
            ]))
    lines = []
    if rep["model"]:
        mo = rep["model"]
        lines.append(f"**{mo['edge']}**（未知期間で上位推奨の期待値 {mo['top_ret']:+.2%} / 全体平均 {mo['base_ret']:+.2%}）")
    if rep["live"]:
        lv = rep["live"]
        lines.append(f"通知した上位5件の答え合わせ: {lv['n']}件 勝率 {lv['win']:.0%} / 平均 {lv['ret']:+.2%}")
    if rep["hot"]:
        lines.append("🔥 最近効いている戦略: " + " / ".join(f"{n}" for n, _ in rep["hot"]))
    if lines:
        embeds.append(dict(title="🧠 AIの状態", color=0x9467BD, description="\n".join(lines),
                           footer=dict(text="※売買の判断はご自身で。過去の傾向は将来を保証しません。")))
    return {"username": "KabuScout", "embeds": embeds}


# ---------------------------------------------------------------- ntfy（短く）
def ntfy_text(rep: dict, n: int = 5) -> str:
    m = rep["market"]
    L = [f"地合い {m['mood']} / シグナル{rep['n_signals']}件"]
    for p in rep["picks"][:n]:
        prob = f" AI{p['prob']:.0%}" if p["prob"] is not None else ""
        L.append(f"{p['rank']}. {p['code']} {p['name']}｜{p['strategies'][0]}{prob}")
        L.append(f"   {p['close']:,.0f} → 利確{p['tp']:,.0f} / 損切{p['sl']:,.0f} {_earn(p)}")
    if rep["paper"]:
        L.append(f"💼 評価額 {rep['paper']['equity']:,.0f}円")
    return "\n".join(L)
