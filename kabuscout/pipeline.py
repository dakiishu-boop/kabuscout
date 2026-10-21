"""日次パイプライン: データ更新 → 指標 → シグナル → 学習 → スコア → ペーパートレード → 通知。"""
from __future__ import annotations

import logging
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

import json
import time

import jpholiday

from . import data, learner, notify, paper, report, site
from .db import connect
from .indicators import compute
from .strategies import detect_all, reasons

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


def is_trading_day(d) -> bool:
    d = pd.Timestamp(d)
    return d.weekday() < 5 and not jpholiday.is_holiday(d.date()) and not (
        (d.month == 12 and d.day == 31) or (d.month == 1 and d.day <= 3))


def wait_for_close_data(cfg: dict) -> None:
    """引け直後の実行用。Yahoo の東証データは約20分遅れなので、15:55 まで待ち、
    さらに当日の足が取れるまで最大 cfg['schedule']['max_wait_min'] 分ほど待つ。"""
    sc = cfg.get("schedule", {})
    now = datetime.now(JST)
    if not sc.get("wait_for_data", True) or not is_trading_day(now) or not (dtime(15, 0) <= now.time() < dtime(18, 0)):
        return
    ready = now.replace(hour=15, minute=55, second=0, microsecond=0)
    if now < ready:
        log.info("引け後データの確定待ち: %s まで %d 秒", ready.strftime("%H:%M"), (ready - now).seconds)
        time.sleep((ready - now).seconds)
    deadline = time.time() + 60 * sc.get("max_wait_min", 45)
    proxy = cfg["universe"]["market_proxy"] + ".T"
    today = datetime.now(JST).strftime("%Y-%m-%d")
    while time.time() < deadline:
        try:
            df = yf.download(proxy, period="5d", progress=False, auto_adjust=False)
            if len(df) and df.index[-1].strftime("%Y-%m-%d") == today:
                return
        except Exception as e:
            log.warning("確認失敗: %s", e)
        log.info("当日の株価がまだ出ていないので5分待ちます")
        time.sleep(300)
    log.warning("当日の株価が確認できないまま実行します")


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


def run_daily(cfg: dict, force: bool = False, retrain: bool = False, notify_on: bool = True) -> dict | None:
    con = connect(cfg["data"]["db_path"])
    wait_for_close_data(cfg)
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
    rep = report.build(cfg, con, ind, top, len(today), n_scanned, bundle, weights, pt, data_date)
    con.execute("INSERT OR REPLACE INTO summaries VALUES (?,?)", (data_date, json.dumps(rep, ensure_ascii=False, default=float)))
    con.commit()
    md = report.to_markdown(rep)
    (cfg["data"]["report_dir"] / f"{data_date}.md").write_text(md + "\n")
    site.build(con, cfg)
    sent = notify.send(cfg, rep) if notify_on else []
    con.execute("UPDATE runs SET finished=?, status='ok', message=? WHERE data_date=?",
                (datetime.now().isoformat(timespec="seconds"), f"{len(today)} signals; sent={sent}", data_date))
    con.commit()
    log.info("完了: %s 候補 %d 件（通知: %s）", data_date, len(today), ",".join(sent) or "なし")
    print("\n" + md)
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
