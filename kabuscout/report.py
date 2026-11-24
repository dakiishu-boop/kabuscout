"""日次レポート。1つの report 辞書から、Markdown・Discord カード・ntfy 文面・Webページ用データを作る。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .strategies import STRATEGY_NAMES, plan_for

WEEKDAYS = "月火水木金土日"
MOOD_COLOR = {"強気": 0xD9534F, "中立": 0xF0AD4E, "弱気": 0x337AB7}  # 日本式: 上げ=赤
MEDALS = ["🥇", "🥈", "🥉"]


# 含み損込みの評価額が最高値から config の stop_review_dd 以上減ったときの表示（過去の最悪を大きく超える＝想定外）
STOP_REVIEW = "最高値からの下落が、過去ペーパートレードの最悪を大きく超えました。実際の資金なら売買を止めて見直す目安です"

HEALTH_ICON = {"ok": "✅", "caution": "🟡", "alert": "🔴", "waiting": "⏳", "unknown": "・"}


def _health_text(h: dict) -> str:
    t = f"{HEALTH_ICON.get(h['status'], '')} {h['text']}"
    if h.get("k"):
        t += (f"（直近{h['k']}回の合計 {h['live_sum']:+.1%}：過去の同じ回数の中で上位{1 - h['percentile']:.0%}・"
              f"ふつうは {h['typical']:+.1%}、下位10%は {h['low10']:+.1%}）")
    return t

# ---------------------------------------------------------------- 表示名と答え合わせ（ジャーナル §89）
MIN_N = 30  # これより少ない戦略の実績は「参考値」
PROB_BANDS = [0.0, 0.2, 0.3, 0.4, 0.5, 1.01]


def prob_name(cfg: dict) -> str:
    """「AI勝率」の正確な名前: 共通ルールで +tp ATR が −sl ATR より先に来る確率。"""
    lab = cfg["labeling"]
    return f"+{lab['tp_atr']:g}ATR先着率({lab['horizon_days']}日)"


def prob_note(cfg: dict) -> str:
    lab = cfg["labeling"]
    return (f"※先着率＝共通ルール（{lab['horizon_days']}営業日以内に +{lab['tp_atr']:g}ATR が −{lab['sl_atr']:g}ATR より先に来る）の確率。"
            f"並んでいる利確・損切は戦略ごとの別ルール。予想リターンはその利確・損切のルールでの損益（コスト{cfg['research']['cost']:.1%}引き）の予測。"
            f"並び順は3つのモデルの順位の平均で、先着率の順ではない")


def _rule_text(plan: dict) -> str:
    return f"+{plan['tp_atr']:g}ATR/−{plan['sl_atr']:g}ATR・{plan['horizon_days']}日"


def answer_stats(con, cfg: dict) -> dict:
    """通知・Webに出した候補の答え合わせ。
    strategies: 実際の売買ルール（戦略ごと）での実績。勝ち＝利確が先、平均は期限切れ込みの損益 − コスト（予想リターンと同じ基準）。未決着は集計に入れない。
    calib: 予測の先着率の帯ごとの、共通ルールでの実際の先着率（先着率の当たり具合）。
    exp: 予想リターンと実績（同じ候補・同じ基準）。"""
    cost = float(cfg["research"]["cost"])
    try:
        sig = pd.read_sql("SELECT strategies, prob, exp_ret, labeled, label, labeled_plan, label_plan, ret_plan FROM signals", con)
    except Exception:
        return dict(cost=cost, min_n=MIN_N, strategies=[], calib=[], exp=None)
    if sig.empty:
        return dict(cost=cost, min_n=MIN_N, strategies=[], calib=[], exp=None)
    plans = [plan_for(str(k).split(","), cfg["labeling"]) for k in sig["strategies"]]
    sig["group"] = [p["source"] or str(k).split(",")[0] for p, k in zip(plans, sig["strategies"])]
    sig["rule"] = [_rule_text(p) for p in plans]
    sig["done"] = sig["labeled_plan"].fillna(0).astype(int) == 1
    sig["net"] = sig["ret_plan"] - cost
    rows = []
    for g, x in sig.groupby("group"):
        d = x[x["done"]]
        rows.append(dict(key=g, name=STRATEGY_NAMES.get(g, g), rule=x["rule"].iloc[0], n=int(len(d)), pending=int((~x["done"]).sum()),
                         win=float(d["label_plan"].mean()) if len(d) else None, avg=float(d["net"].mean()) if len(d) else None,
                         ref=bool(len(d) < MIN_N)))
    rows.sort(key=lambda r: (-r["n"], -r["pending"]))
    c = sig[(sig["labeled"] == 1) & sig["prob"].notna()]
    calib = []
    for lo, hi in zip(PROB_BANDS[:-1], PROB_BANDS[1:]):
        b = c[(c["prob"] >= lo) & (c["prob"] < hi)]
        if len(b):
            calib.append(dict(lo=lo, hi=min(hi, 1.0), n=int(len(b)), pred=float(b["prob"].mean()), actual=float(b["label"].mean())))
    e = sig[sig["done"] & sig["exp_ret"].notna()]
    exp = dict(n=int(len(e)), pred=float(e["exp_ret"].mean()), actual=float(e["net"].mean())) if len(e) else None
    return dict(cost=cost, min_n=MIN_N, strategies=rows, calib=calib, exp=exp)


def answer_lines(a: dict | None, max_rows: int = 8) -> list[str]:
    """答え合わせの行（Discord・Markdown 共通）。"""
    if not a or not (a.get("strategies") or a.get("calib")):
        return []
    L = [f"実際の売買ルールでの答え合わせ（勝ち＝利確が先・平均は期限切れ込み・コスト{a['cost']:.1%}引き）:"]
    for r in a["strategies"][:max_rows]:
        res = f"{r['n']}件 勝率 {r['win']:.0%} / 平均 {r['avg']:+.2%}" if r["n"] else "決着0件"
        L.append(f"・{r['name']}（{r['rule']}）: {res}・未決着{r['pending']}件" + ("（参考値）" if r["n"] and r["ref"] else ""))
    if a.get("exp"):
        x = a["exp"]
        L.append(f"予想リターンと実績（{x['n']}件）: 予想 {x['pred']:+.2%} / 実績 {x['actual']:+.2%}" + ("（参考値）" if x["n"] < a["min_n"] else ""))
    if a.get("calib"):
        L.append("先着率の当たり具合（予測の帯: 実際の先着率・件数）: " + " / ".join(
            f"{c['lo']:.0%}〜{c['hi']:.0%}: {c['actual']:.0%}・{c['n']}件" for c in a["calib"]))
    return L


def market_stats(ind: dict) -> dict:
    r = {k: float(ind[k].iloc[-1].iloc[0]) for k in ("mkt_dev75", "mkt_ret20", "breadth25")}
    r["mood"] = "強気" if r["mkt_dev75"] > 0 and r["breadth25"] > 0.5 else "弱気" if r["mkt_dev75"] < 0 else "中立"
    return r


def edge_label(m: dict) -> str:
    lift = m["daily_top_ret"] - m["base_ret"]
    return "優位性あり" if lift >= 0.005 else "わずかに優位" if lift > 0 else "まだ優位性なし（ルール中心で順位付け）"


def build(cfg, con, ind, top, n_signals, n_scanned, bundle, weights, pt, data_date, regime=None) -> dict:
    d = pd.Timestamp(data_date)
    picks = []
    for _, r in top.iterrows():
        picks.append(dict(
            rank=int(r["rank"]), code=r["code"], name=r["name"],
            strategies=[STRATEGY_NAMES.get(k, k) for k in r["strategies"].split(",")],
            prob=None if pd.isna(r["prob"]) else float(r["prob"]),
            exp_ret=None if pd.isna(r.get("exp_ret", np.nan)) else float(r["exp_ret"]),
            close=float(r["close"]), tp=float(r["tp"]), sl=float(r["sl"]),
            horizon=int(r["horizon"]) if "horizon" in r and r["horizon"] == r["horizon"] else cfg["labeling"]["horizon_days"],
            earnings_days=None if pd.isna(r["earnings_days"]) else int(r["earnings_days"]),
            reasons=r["reasons"], flows=_flows_text(r),
        ))
    live = pd.read_sql("SELECT label, ret FROM signals WHERE labeled=1 AND rank<=5", con)
    model = None
    if bundle is not None:
        m = bundle["metrics"]
        model = dict(version=bundle["version"], edge=edge_label(m), top_ret=m["daily_top_ret"],
                     top_win=m["daily_top_win"], base_ret=m["base_ret"], base_win=m["base_win"],
                     crisis=m.get("crisis"))
    return dict(
        date=data_date, date_label=f"{d.month}/{d.day}({WEEKDAYS[d.weekday()]})",
        site_url=cfg.get("site", {}).get("url", ""),
        market=market_stats(ind), n_scanned=n_scanned, n_signals=n_signals, picks=picks, model=model,
        live=dict(n=len(live), win=float(live["label"].mean()), ret=float(live["ret"].mean())) if len(live) >= 5 else None,
        hot=[(STRATEGY_NAMES.get(k, k), w) for k, w in sorted(weights.items(), key=lambda t: -t[1])[:3]],
        paper=pt, regime=regime or dict(crisis=False, event=""),
        prob_name=prob_name(cfg), prob_note=prob_note(cfg), answers=answer_stats(con, cfg),
    )


def crisis_line(rep: dict) -> str:
    """特例（危機）局面のときの注意書き。"""
    rg = rep.get("regime") or {}
    if not rg.get("crisis"):
        return ""
    s = f"⚠️ 平常時と違う相場（特例モード）: {rg['event']}。過去の危機局面では値動きの法則が崩れやすく、AIの判断も平常時ほど当てになりません。"
    c = (rep.get("model") or {}).get("crisis") or {}
    if c.get("daily_top_ret") is not None:
        s += f" 過去の危機局面での上位推奨の期待値は {c['daily_top_ret']:+.2%}。"
    return s + " ペーパートレードの新規買いは1銘柄に絞っています。"


def _flows_text(r) -> str:
    """需給の参考情報（JPXの日々の信用残・大口空売り残高）。"""
    parts = []
    g = lambda k: r.get(k) if k in r and r.get(k) == r.get(k) else None
    if g("margin_ratio") is not None:
        parts.append(f"信用倍率{g('margin_ratio'):.1f}倍")
    if g("buy_pct") is not None:
        chg = g("buy_pct_chg5")
        parts.append(f"買残 上場比{g('buy_pct'):.1%}" + (f"（1週間で{chg:+.1%}）" if chg else ""))
    if g("short_total") is not None:
        parts.append(f"大口空売り{g('short_total'):.1%}")
    if g("lending_fee") is not None and g("lending_fee") > 0:
        parts.append(f"逆日歩{g('lending_fee'):g}円（売り方の踏み上げに注意）")
    return "・".join(parts)


def _pct(a, b):
    return a / b - 1


def _earn(p, limit=14):
    e = p["earnings_days"]
    return f"⚠️決算まで{e}日" if e is not None and e <= limit else ""


def _swap_text(pt: dict) -> str:
    """比較用の口座（決算好反応の入れ替え、ジャーナル 64）の一行。本番の口座との差はルールの違いの分。"""
    s = pt["swap"]
    return (f"決算好反応の入れ替えの口座: 評価額 {s['equity']:,.0f}円（本番との差 {s['equity'] - pt['equity']:+,.0f}円・"
            f"入れ替え {s.get('swaps', 0)}回・最高値から {s.get('drawdown', 0):+.1%}）")


def _conf_text(pt: dict) -> str:
    """比較用の口座その2（3つのモデルの合意が高い候補を多めに買う、ジャーナル 82・83）の一行。"""
    s = pt["conf"]
    return (f"自信で買う量を変える口座: 評価額 {s['equity']:,.0f}円（本番との差 {s['equity'] - pt['equity']:+,.0f}円・"
            f"多めに買った注文 {s.get('sized', 0)}回・最高値から {s.get('drawdown', 0):+.1%}）")


def _rev_text(pt: dict) -> str:
    """比較用の口座その3（修正好反応の網と印を足したAI、ジャーナル 84・85）の一行。"""
    s = pt["rev"]
    return (f"修正好反応を足した口座: 評価額 {s['equity']:,.0f}円（本番との差 {s['equity'] - pt['equity']:+,.0f}円・"
            f"本番と違う注文 {s.get('diff_orders', 0)}回・最高値から {s.get('drawdown', 0):+.1%}）")


def _side_lines(pt: dict) -> list[str]:
    """比較用の口座（本番の判断には使わない）の行。口座ごとに、まとめの1行とその日の出来事。"""
    L = []
    for key, text in (("swap", _swap_text), ("conf", _conf_text), ("rev", _rev_text)):
        if pt.get(key):
            L.append(text(pt))
            L += [f"• {e}" for e in pt[key]["events"]]
    if pt.get("rev_wait"):
        L.append(pt["rev_wait"])
    return L


# ---------------------------------------------------------------- Markdown（ログ・レポートファイル用）
def to_markdown(rep: dict) -> str:
    m = rep["market"]
    L = [f"# KabuScout {rep['date_label']} 引け後スクリーニング", "",
         *([rep["prob_note"], ""] if rep.get("prob_note") else []),
         f"地合い **{m['mood']}**（日経平均 75日線乖離 {m['mkt_dev75']:+.1%} / 20日 {m['mkt_ret20']:+.1%} / "
         f"25日線より上の銘柄 {m['breadth25']:.0%}）", f"{rep['n_scanned']:,}銘柄を走査 → シグナル{rep['n_signals']}件", ""]
    if crisis_line(rep):
        L[2:2] = [crisis_line(rep)]
    if rep.get("stale"):
        L[2:2] = [f"⚠️ データの遅れ: {s}" for s in rep["stale"]] + [""]
    for p in rep["picks"]:
        L.append(f"{p['rank']}. **{p['code']} {p['name']}**【{'・'.join(p['strategies'])}】{_prob_text(rep, p)} {_earn(p)}")
        L.append(f"   終値 {p['close']:,.0f} → 利確 {p['tp']:,.0f} ({_pct(p['tp'], p['close']):+.1%}) / "
                 f"損切 {p['sl']:,.0f} ({_pct(p['sl'], p['close']):+.1%}) / 最長{p.get('horizon', '')}営業日")
        L.append(f"   {p['reasons']}")
        if p.get("flows"):
            L.append(f"   需給: {p['flows']}")
    if rep["model"]:
        mo = rep["model"]
        L += ["", f"AI v{mo['version']}（{mo['edge']}）: 上位推奨の期待値 {mo['top_ret']:+.2%} / 先着率 {mo['top_win']:.0%}"
                  f"（全シグナル平均 {mo['base_ret']:+.2%} / {mo['base_win']:.0%}）"]
    if rep["paper"]:
        pt = rep["paper"]
        L.append(f"ペーパートレード: 評価額 {pt['equity']:,.0f}円（実現 {pt['realized']:+,.0f} / 含み {pt['unrealized']:+,.0f} / "
                 f"最高値から {pt.get('drawdown', 0):+.1%}）")
        if pt.get("stop_review"):
            L.append(f"  ⚠️ {STOP_REVIEW}")
        if pt.get("health"):
            L.append(f"  健康診断: {_health_text(pt['health'])}")
        if pt.get("core"):
            c = pt["core"]
            L.append(f"  指数（{c['name']}）: {c['value']:,.0f}円（{c['ret']:+.1%}） → 合計 {pt['equity'] + c['value']:,.0f}円")
        L += [f"  ・{e}" for e in pt["events"]]
        L += [("    ・" + x[2:]) if x.startswith("• ") else f"  {x}" for x in _side_lines(pt)]
    L += answer_lines(rep.get("answers"))
    if rep.get("forward"):
        from .forward import text as fw_text
        L.append(f"全候補の答え合わせ: {fw_text(rep['forward'])}")
    return "\n".join(L)


def _prob_text(rep: dict, p: dict, bold: bool = False) -> str:
    """銘柄ごとの「+2ATR先着率 33%・予想リターン +1.2%」。"""
    b = "**" if bold else ""
    out = []
    if p.get("prob") is not None:
        out.append(f"{rep.get('prob_name', '先着率')} {b}{p['prob']:.0%}{b}")
    if p.get("exp_ret") is not None:
        out.append(f"予想リターン {b}{p['exp_ret']:+.1%}{b}")
    return "・".join(out)


# ---------------------------------------------------------------- Discord（カード表示）
def discord_payload(rep: dict, max_picks: int = 10) -> dict:
    m, url = rep["market"], rep["site_url"] or None
    head = dict(
        title=f"📈 KabuScout {rep['date_label']} 引け後スクリーニング",
        url=url, color=0x8B0000 if crisis_line(rep) else MOOD_COLOR[m["mood"]],
        description=(f"**{crisis_line(rep)}**\n\n" if crisis_line(rep) else "")
                    + "".join(f"⚠️ **データの遅れ**: {s}\n" for s in rep.get("stale") or [])
                    + f"**{rep['n_scanned']:,}** 銘柄を走査 → シグナル **{rep['n_signals']}** 件"
                    + (f"\n{rep['prob_note']}" if rep.get("prob_note") else "")
                    + (f"\n🌐 [Webで詳しく見る（チャート・過去の結果）]({url})" if url else ""),
        fields=[
            dict(name="地合い", value=f"**{m['mood']}**", inline=True),
            dict(name="日経平均 75日線", value=f"{m['mkt_dev75']:+.1%}", inline=True),
            dict(name="25日線より上", value=f"{m['breadth25']:.0%}", inline=True),
        ],
    )
    fields = []
    for p in rep["picks"][:max_picks]:
        mark = MEDALS[p["rank"] - 1] if p["rank"] <= 3 else f"`{p['rank']:>2}`"
        prob = (_prob_text(rep, p, bold=True) + "　") if p.get("prob") is not None or p.get("exp_ret") is not None else ""
        yahoo = f"https://finance.yahoo.co.jp/quote/{p['code']}.T"
        fields.append(dict(
            name=f"{mark} {p['code']} {p['name']}　{_earn(p)}",
            value=(f"🏷️ {' / '.join(p['strategies'])}\n"
                   f"```\n終値 {p['close']:>8,.0f}\n利確 {p['tp']:>8,.0f} ({_pct(p['tp'], p['close']):+.1%})\n"
                   f"損切 {p['sl']:>8,.0f} ({_pct(p['sl'], p['close']):+.1%})\n保有 最長{p.get('horizon', '')}営業日```"
                   f"{prob}[Yahoo]({yahoo})\n*{p['reasons']}*"
                   + (f"\n📊 {p['flows']}" if p.get("flows") else ""))[:1024],
            inline=False))
    picks = dict(title=f"🎯 注目銘柄 TOP{len(fields)}", color=0xE377C2, fields=fields)

    embeds = [head, picks]
    pt = rep["paper"]
    if pt:
        wr = f"{pt['win_rate']:.0%}" if pt["win_rate"] is not None else "-"
        embeds.append(dict(
            title=f"💼 ペーパートレード（仮想{pt.get('capital', 0) / 10000:,.0f}万円）",
            color=0xD62728 if pt.get("stop_review") else 0x2CA02C,
            description=((f"⚠️ **{STOP_REVIEW}**\n" if pt.get("stop_review") else "")
                         + ("\n".join(f"• {e}" for e in pt["events"]) or "本日の売買なし")),
            fields=[
                dict(name="評価額", value=f"{pt['equity']:,.0f}円", inline=True),
                dict(name="実現損益", value=f"{pt['realized']:+,.0f}円", inline=True),
                dict(name="含み損益", value=f"{pt['unrealized']:+,.0f}円", inline=True),
                dict(name="保有 / 注文", value=f"{pt['open']} / {pt['pending']}", inline=True),
                dict(name="決済", value=f"{pt['closed']}回", inline=True),
                dict(name="勝率", value=wr, inline=True),
                dict(name="最高値からの下落", value=f"{pt.get('drawdown', 0):+.1%}", inline=True),
            ] + ([dict(name="健康診断（過去の想定との比較）", value=_health_text(pt["health"]), inline=False)] if pt.get("health") else [])
              + ([dict(name=f"指数の部分（{pt['core']['name']}）", value=f"{pt['core']['value']:,.0f}円（{pt['core']['ret']:+.1%}）／合計 {pt['equity'] + pt['core']['value']:,.0f}円", inline=False)] if pt.get("core") else [])
              + ([dict(name="比較用の口座（本番の判断には使わない）", value="\n".join(_side_lines(pt))[:1024], inline=False)] if _side_lines(pt) else [])))
    lines = []
    if rep["model"]:
        mo = rep["model"]
        lines.append(f"**{mo['edge']}**（未知期間で上位推奨の期待値 {mo['top_ret']:+.2%} / 全体平均 {mo['base_ret']:+.2%}・コスト控除前）")
    if rep["live"]:
        lv = rep["live"]
        lines.append(f"通知した上位5件の答え合わせ（共通ルール・コスト前）: {lv['n']}件 先着率 {lv['win']:.0%} / 平均 {lv['ret']:+.2%}")
    lines += answer_lines(rep.get("answers"))[:12]
    if rep["hot"]:
        lines.append("🔥 最近効いている戦略: " + " / ".join(f"{n}" for n, _ in rep["hot"]))
    if rep.get("forward"):
        from .forward import text as fw_text
        lines.append("全候補の答え合わせ: " + fw_text(rep["forward"]))
        lines += [f"影の判定: {x}" for x in rep["forward"].get("settled") or []]
    if lines:
        embeds.append(dict(title="🧠 AIの状態", color=0x9467BD, description="\n".join(lines),
                           footer=dict(text="※売買の判断はご自身で。過去の傾向は将来を保証しません。")))
    return {"username": "KabuScout", "embeds": embeds}


# ---------------------------------------------------------------- ntfy（短く）
def ntfy_text(rep: dict, n: int = 5) -> str:
    m = rep["market"]
    L = [f"地合い {m['mood']} / シグナル{rep['n_signals']}件"] + ([crisis_line(rep)] if crisis_line(rep) else []) \
        + [f"⚠️ データの遅れ: {s}" for s in rep.get("stale") or []]
    for p in rep["picks"][:n]:
        prob = f" 先着{p['prob']:.0%}" if p["prob"] is not None else ""
        prob += f" 予想{p['exp_ret']:+.1%}" if p.get("exp_ret") is not None else ""
        L.append(f"{p['rank']}. {p['code']} {p['name']}｜{p['strategies'][0]}{prob}")
        L.append(f"   {p['close']:,.0f} → 利確{p['tp']:,.0f} / 損切{p['sl']:,.0f} {_earn(p)}")
    if rep["paper"]:
        L.append(f"💼 評価額 {rep['paper']['equity']:,.0f}円")
    return "\n".join(L)


# ---------------------------------------------------------------- 週末の研究レポート
def research_payload(summary: dict | None, bt: dict, url: str = "") -> dict:
    embeds = []
    if summary:
        adopted = [r for r in summary["results"] if r["adopted"]]
        lines = [f"探索期間 {summary['search_from']}〜{summary['search_to']} ／ 金庫 {summary['vault_from']}〜（研究中は未使用）",
                 f"取引コスト {summary['cost']:.1%}/往復 を差し引いて評価", ""]
        for r in summary["results"]:
            mark = "✅" if r["adopted"] else "・"
            lines.append(f"{mark} **{r['name']}**（{r['trials']}通り／累計{r['cum_trials']}）: {r['reason']}")
        P = summary.get("plan")
        if P:
            fmt = lambda p: f"利確+{p['tp_atr']}ATR／損切−{p['sl_atr']}ATR／最長{p['horizon_days']}日"
            lines += ["", f"{'✅' if P['adopted'] else '・'} **売買ルール**（{P['trials']}通り／累計{P['cum_trials']}）: {P['reason']}",
                      f"　現行 {fmt(P['incumbent'])}：1枠あたり年率 {P['inc_score']:+.1%}"]
            if P.get("candidate"):
                lines.append(f"　最有力 {fmt(P['candidate'])}：年率 {P['cand_score']:+.1%}")
            adopted = adopted + ([P] if P["adopted"] else [])
            per = P.get("per_strategy") or []
            if per:
                lines.append("")
                lines.append("**戦略ごとの売買ルール**（現行の年率換算 → 最有力候補）")
                for r in per:
                    mark = "✅" if r["adopted"] else "・"
                    cand = f" → {fmt(r['candidate'])} {r['cand_score']:+.1%}" if r.get("candidate") else ""
                    lines.append(f"{mark} {r['name']}: {r['inc_score']:+.1%}{cand}（{r['reason']}）" if r.get("inc_score") is not None
                                 else f"{mark} {r['name']}: {r['reason']}")
                adopted = adopted + [r for r in per if r["adopted"]]
        embeds.append(dict(
            title="🔬 研究ループ: " + (f"{len(adopted)}つの戦略の条件を更新" if adopted else "条件の更新なし（現行が最善 or 改善が偶然の範囲）"),
            url=url or None, color=0x2CA02C if adopted else 0x808080, description="\n".join(lines)[:4000]))
    s = bt["stats"]
    M = (summary or {}).get("model")
    if M and M.get("trials"):
        embeds[0]["description"] = (embeds[0]["description"] + f"\n{'✅' if M['adopted'] else '👥' if M.get('shadowed') else '・'} **AIのモデル設定**（{M['trials']}通り／累計{M['cum_trials']}）: {M['reason']}")[:4000]
    F = (summary or {}).get("features")
    if F and F.get("trials"):
        embeds[0]["description"] = (embeds[0]["description"] + f"\n{'✅' if F['adopted'] else '👥' if F.get('shadowed') else '・'} **AIの特徴量**（{F['trials']}通り／累計{F['cum_trials']}）: {F['reason']}")[:4000]
    P = (summary or {}).get("specialists")
    if P and P.get("trials"):
        embeds[0]["description"] = (embeds[0]["description"] + f"\n{'✅' if P['adopted'] else '👥' if P.get('shadowed') else '・'} **専門モデル**（{P['trials']}通り／累計{P['cum_trials']}）: {P['reason']}")[:4000]
    W = (summary or {}).get("weak_consensus")
    if W and W.get("trials"):
        embeds[0]["description"] = (embeds[0]["description"] + f"\n{'✅' if W['adopted'] else '・'} **弱い相場の合意モード**（{W['trials']}通り／累計{W['cum_trials']}）: {W['reason']}")[:4000]
    G = (summary or {}).get("gate")
    if G:
        embeds[0]["description"] = (embeds[0]["description"] + f"\n\n{'✅' if G['adopted'] else '・'} **地合いによる買い控え**（{G['trials']}通り／累計{G['cum_trials']}）: {G['reason']}")[:4000]
    embeds.append(dict(
        title=f"📜 過去でのペーパートレード（{bt['start'][:4]}〜{bt['end'][:4]}・ウォークフォワード）",
        url=(url + "#paper") if url else None, color=0x9467BD,
        description=f"毎年「その年より前のデータだけ」で学習したAIで、その年を売買した結果（資金{bt.get('account', {}).get('capital', 0) / 10000:,.0f}万円・100株単位・含み損込み）",
        fields=[
            dict(name="AI選定", value=f"年率 **{s['ai']['cagr']:+.1%}**（税引後 {s['ai'].get('after_tax', float('nan')):+.1%}）\n"
                                     f"最大下落 {s['ai']['max_dd']:.0%}・最悪の年 {s['ai'].get('worst_year', float('nan')):+.0%}", inline=True),
            dict(name="ランダム選定", value=f"年率 {s['random']['cagr']:+.1%}\n最大下落 {s['random']['max_dd']:.0%}", inline=True),
            dict(name="日経平均を保有", value=f"年率 {s['index']['cagr']:+.1%}\n最大下落 {s['index']['max_dd']:.0%}", inline=True),
        ] + _baseline_fields(bt),
        footer=dict(text="戦略条件は過去で調整済み・現存銘柄のみ（生存者バイアス）・2018年以前は1,000株単位の銘柄を100株で買えたものとしているため、実際より良く見えやすい点に注意")))
    return {"username": "KabuScout", "embeds": embeds}


def _baseline_fields(bt: dict) -> list[dict]:
    """比べる基準の期間（2019年〜）と、取引コストの見積もりを変えた場合（ジャーナル 77・78）。"""
    out = []
    S = bt.get("since")
    if S:
        a = S["ai"]
        i = S["index"]  # Sharpe は AI と日経平均のどちらの数字かが分かるように、それぞれの行に書く
        out.append(dict(name=f"{S['start'][:4]}年以降（比べる基準）",
                        value=f"AI 年率 **{a['cagr']:+.1%}**・最大下落 {a['max_dd']:.0%}・Sharpe {a.get('sharpe') or 0:.2f}\n"
                              f"日経平均（配当なし） 年率 {i['cagr']:+.1%}・最大下落 {i['max_dd']:.0%}・Sharpe {i.get('sharpe') or 0:.2f}", inline=False))
    C = bt.get("costs")
    if C:
        out.append(dict(name="往復コストを変えた場合（年率・全期間 ／ 基準の期間）",
                        value="・".join(f"{c['cost']:.1%}: {c['all']['cagr']:+.1%} ／ {c['since']['cagr']:+.1%}" for c in C), inline=False))
    return out


def research_text(summary: dict | None, bt: dict) -> str:
    s = bt["stats"]
    head = ""
    if summary:
        n = sum(r["adopted"] for r in summary["results"])
        head = f"研究ループ: {n}戦略を更新\n" if n else "研究ループ: 更新なし\n"
    return head + f"過去ペーパートレード 年率 AI {s['ai']['cagr']:+.1%} / ランダム {s['random']['cagr']:+.1%} / 日経平均 {s['index']['cagr']:+.1%}"
