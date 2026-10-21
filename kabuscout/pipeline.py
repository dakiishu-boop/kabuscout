"""日次パイプライン: データ更新 → 指標 → シグナル → 学習 → スコア → ペーパートレード → 通知。"""
from __future__ import annotations

import logging
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

from . import data, learner, notify, paper
from .db import connect
from .indicators import compute
from .strategies import STRATEGY_NAMES, detect_all, reasons

log = logging.getLogger(__name__)
JST = ZoneInfo("Asia/Tokyo")


def _earnings_days(code: str, asof: pd.Timestamp) -> float:
    """次の決算発表までの日数（不明なら NaN）。"""
    try:
        cal = yf.Ticker(f"{code}.T").calendar or {}
        ds = cal.get("Earnings Date") or []
        fut = [pd.Timestamp(d) for d in ds if pd.Timestamp(d) >= asof.normalize()]
        return float((min(fut) - asof.normalize()).days) if fut else np.nan
    except Exception:
        return np.nan


def _cutoff() -> pd.Timestamp:
    """使ってよい最新日。JST 15:40 前は当日の途中経過の足を使わない。"""
    now = datetime.now(JST)
    today = pd.Timestamp(now.date())
    return today if now.time() >= dtime(15, 40) else today - pd.Timedelta(days=1)


def build(con, cfg: dict, uni: pd.DataFrame):
    """全期間の指標・シグナル・ラベル付きデータセットを作る。"""
    proxy = cfg["universe"]["market_proxy"]
    codes = uni["code"].tolist()
    panel = data.load_panel(con, codes=codes + [proxy])
    cutoff = _cutoff()
    panel = {k: v[v.index <= cutoff] for k, v in panel.items()}
    # メモリ節約: 一度も流動性基準を満たしたことのない銘柄は除外
    tv = (panel["close"] * panel["volume"]).rolling(20, min_periods=10).mean().max()
    keep = [c for c in panel["close"].columns if tv.get(c, 0) >= cfg["universe"]["min_turnover_jpy"] or c == proxy]
    panel = {k: v[keep].astype("float32") for k, v in panel.items()}

    sectors = uni.set_index("code")["sector33"]
    ind = compute(panel, proxy, sectors)
    ind = {k: v.astype("float32") for k, v in ind.items()}
    in_uni = pd.Series([c in set(codes) for c in ind["close"].columns], index=ind["close"].columns)
    mask = ((ind["turnover"] >= cfg["universe"]["min_turnover_jpy"])
            & (ind["close"] >= cfg["universe"]["min_price"])) & in_uni
    sigs = detect_all(ind, mask)
    ds = learner.label(ind, learner.extract(ind, sigs), cfg)
    return ind, sigs, ds, int(mask.iloc[-1].sum())


def _market_line(ind: dict) -> str:
    r = {k: float(ind[k].iloc[-1].iloc[0]) for k in ("mkt_dev75", "mkt_ret20", "breadth25")}
    mood = "強気" if r["mkt_dev75"] > 0 and r["breadth25"] > 0.5 else "弱気" if r["mkt_dev75"] < 0 else "中立"
    return (f"地合い: {mood}（TOPIX 75日線乖離 {r['mkt_dev75']:+.1%} / 20日 {r['mkt_ret20']:+.1%} / "
            f"25日線より上の銘柄 {r['breadth25']:.0%}）")


def run_daily(cfg: dict, force: bool = False, retrain: bool = False, notify_on: bool = True) -> dict | None:
    con = connect(cfg["data"]["db_path"])
    uni = data.refresh_universe(con, cfg)
    names = dict(zip(uni["code"], uni["name"]))
    data.update_prices(con, uni["code"].tolist() + [cfg["universe"]["market_proxy"]], cfg)

    data_date = con.execute("SELECT MAX(date) FROM prices WHERE code=? AND date<=?",
                            (cfg["universe"]["market_proxy"], _cutoff().strftime("%Y-%m-%d"))).fetchone()[0]
    done = con.execute("SELECT status FROM runs WHERE data_date=?", (data_date,)).fetchone()
    if done and done[0] == "ok" and not force:
        log.info("%s は処理済み（休場日か二重実行）。--force で再実行できます", data_date)
        return None
    log.info("指標とシグナルを計算中…（データ日付 %s）", data_date)
    ind, sigs, ds, n_scanned = build(con, cfg, uni)
    last = ind["close"].index[-1]
    data_date = last.strftime("%Y-%m-%d")
    con.execute("INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?)",
                (data_date, datetime.now().isoformat(timespec="seconds"), None, "running", None))
    con.commit()

    # ---- 学習（一定営業日ごと、またはモデル未作成時）
    bundle = learner.load_active(con, cfg)
    since = int((ind["close"].index > pd.Timestamp(bundle["data_end"])).sum()) if bundle else 0
    if retrain or bundle is None or since >= cfg["learning"]["retrain_every_days"]:
        log.info("再学習中…（学習データ %d 件）", int(ds["complete"].sum()))
        bundle = learner.train(con, cfg, ds, data_date) or bundle
    weights = learner.load_strategy_weights(con)

    # ---- 今日の候補
    today = ds[ds["date"] == last].copy()
    scored = learner.score(today, bundle, weights)
    top = scored.head(cfg["screening"]["top_n"]).copy()
    top["name"] = top["code"].map(names).fillna("")
    top["reasons"] = [reasons(r, r["strategies"].split(",")) for _, r in top.iterrows()]
    H = cfg["labeling"]["horizon_days"]
    k_tp, k_sl = cfg["labeling"]["tp_atr"], cfg["labeling"]["sl_atr"]
    top["tp"] = top["close"] + k_tp * top["atr"]
    top["sl"] = top["close"] - k_sl * top["atr"]
    n_e = cfg["screening"]["earnings_check_n"]
    top["earnings_days"] = [(_earnings_days(c, last) if i < n_e else np.nan) for i, c in enumerate(top["code"])]
    top["rank"] = range(1, len(top) + 1)
    _store_live(con, top, data_date)
    _label_live(con, ds)

    pt = paper.step(con, cfg, ind, top, last) if cfg["paper_trading"]["enabled"] else None
    title, full, short = _message(cfg, ind, top, len(today), n_scanned, bundle, weights, pt, con, data_date)
    (cfg["data"]["report_dir"] / f"{data_date}.md").write_text(f"# {title}\n\n{full}\n")
    sent = notify.send(cfg, title, full, short) if notify_on else []
    con.execute("UPDATE runs SET finished=?, status='ok', message=? WHERE data_date=?",
                (datetime.now().isoformat(timespec="seconds"), f"{len(today)} signals; sent={sent}", data_date))
    con.commit()
    log.info("完了: %s 候補 %d 件（通知: %s）", data_date, len(today), ",".join(sent) or "なし")
    print(f"\n{title}\n{full}")
    return dict(date=data_date, top=top, n_signals=len(today))


def _store_live(con, top: pd.DataFrame, data_date: str) -> None:
    con.execute("DELETE FROM signals WHERE date=?", (data_date,))
    for _, r in top.iterrows():
        con.execute(
            "INSERT INTO signals(date,code,name,strategies,rule_score,prob,final_score,rank,model_version,"
            "close,atr,tp,sl,reasons,earnings) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (data_date, r["code"], r["name"], r["strategies"], float(r["rule_score"]),
             None if pd.isna(r["prob"]) else float(r["prob"]), float(r["final_score"]), int(r["rank"]),
             r["model_version"], float(r["close"]), float(r["atr"]), float(r["tp"]), float(r["sl"]),
             r["reasons"], None if pd.isna(r["earnings_days"]) else int(r["earnings_days"])))
    con.commit()


def _label_live(con, ds: pd.DataFrame) -> None:
    """過去に通知した候補の答え合わせ。"""
    pend = pd.read_sql("SELECT date, code FROM signals WHERE labeled=0", con)
    if pend.empty:
        return
    d = ds[ds["complete"]].assign(date=lambda x: x["date"].dt.strftime("%Y-%m-%d"))
    m = pend.merge(d[["date", "code", "label", "ret", "exit_days"]], on=["date", "code"])
    con.executemany("UPDATE signals SET label=?, ret=?, exit_days=?, labeled=1 WHERE date=? AND code=?",
                    [(int(r.label), float(r.ret), int(r.exit_days), r.date, r.code) for r in m.itertuples()])
    con.commit()


def _message(cfg, ind, top, n_sig, n_scanned, bundle, weights, pt, con, data_date):
    title = f"KabuScout {data_date} 引け後スクリーニング"
    L = [_market_line(ind), f"対象 {n_scanned:,} 銘柄を走査 → シグナル {n_sig} 件、上位 {len(top)} 件:", ""]
    for _, r in top.iterrows():
        strat = "・".join(STRATEGY_NAMES[k] for k in r["strategies"].split(","))
        prob = f" / AI勝率予測 {r['prob']:.0%}" if pd.notna(r["prob"]) else ""
        warn = f" ⚠決算まで{int(r['earnings_days'])}日" if pd.notna(r["earnings_days"]) and r["earnings_days"] <= 14 else ""
        L.append(f"**{r['rank']}. {r['code']} {r['name']}**【{strat}】{prob}{warn}")
        L.append(f"   終値 {r['close']:,.0f} / 利確目安 {r['tp']:,.0f} / 損切目安 {r['sl']:,.0f}")
        L.append(f"   {r['reasons']}")
    L.append("")
    if bundle is not None:
        m = bundle["metrics"]
        lift = m["daily_top_ret"] - m["base_ret"]
        edge = "優位性あり" if lift >= 0.005 else "わずかに優位" if lift > 0 else "まだ優位性なし（ルール中心で順位付け）"
        L.append(f"🧠 AIモデル v{bundle['version']}（{edge}）: 学習に使っていない期間で、上位推奨の期待値 "
                 f"{m['daily_top_ret']:+.2%} / 勝率 {m['daily_top_win']:.0%}（全シグナル平均 {m['base_ret']:+.2%} / {m['base_win']:.0%}）")
    live = pd.read_sql("SELECT label, ret FROM signals WHERE labeled=1 AND rank<=5", con)
    if len(live) >= 5:
        L.append(f"📊 実際に通知した上位5件の答え合わせ: {len(live)}件 勝率 {live['label'].mean():.0%} / 平均 {live['ret'].mean():+.2%}")
    if weights:
        best = sorted(weights.items(), key=lambda t: -t[1])[:3]
        L.append("🔥 最近効いている戦略: " + " / ".join(f"{STRATEGY_NAMES.get(k, k)}(×{w:.2f})" for k, w in best))
    if pt:
        wr = f" 勝率{pt['win_rate']:.0%}" if pt["win_rate"] is not None else ""
        L.append(f"💼 ペーパートレード: 保有{pt['open']} 注文{pt['pending']} 決済{pt['closed']}{wr} / "
                 f"評価額 {pt['equity']:,.0f}円（実現 {pt['realized']:+,.0f} / 含み {pt['unrealized']:+,.0f}）")
        for e in pt["events"]:
            L.append(f"   ・{e}")
    L.append("\n※売買の判断はご自身で。過去の傾向は将来を保証しません。")
    short = f"{len(top)}件: " + ", ".join(f"{r['code']} {r['name']}" for _, r in top.head(5).iterrows())
    return title, "\n".join(L), short
