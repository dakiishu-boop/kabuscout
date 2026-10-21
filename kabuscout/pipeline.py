"""日次パイプライン: データ更新 → 指標 → シグナル → 学習 → スコア → ペーパートレード → 通知。"""
from __future__ import annotations

import logging
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo

import json
import time

import jpholiday
import numpy as np
import pandas as pd
import yfinance as yf

from . import data, earnings, flows, learner, notify, paper, pool, regimes, report, site, tdnet
from .db import connect
from .strategies import apply_flags, load_params, plan_for, reasons

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
    # 指数ETFと主力株の両方で当日の足が出ているかを確認
    probe = [data.ticker(cfg["universe"]["market_proxy"]), "7203.T", "8306.T", "6758.T", "9984.T"]
    today = datetime.now(JST).strftime("%Y-%m-%d")
    while time.time() < deadline:
        try:
            df = yf.download(probe, period="5d", progress=False, auto_adjust=False, group_by="ticker")
            closes = df.xs("Close", axis=1, level=1)
            if len(closes) and closes.index[-1].strftime("%Y-%m-%d") == today and closes.iloc[-1].notna().all():
                return
        except Exception as e:
            log.warning("確認失敗: %s", e)
        log.info("当日の株価がまだ出ていないので5分待ちます")
        time.sleep(300)
    log.warning("当日の株価が確認できないまま実行します")


def _usable_date(store, cutoff: pd.Timestamp) -> pd.Timestamp:
    """全銘柄の8割以上の株価が揃っている最新日。Yahoo 側の更新途中や不具合で
    一部の銘柄しか当日の足が無いときに、古い値で判定してしまうのを防ぐ。"""
    c = store.load(start=(cutoff - pd.Timedelta(days=30)).strftime("%Y-%m-%d"),
                   end=cutoff.strftime("%Y-%m-%d"), fields=["close"])["close"]
    cnt = c.notna().sum(axis=1)
    ok = cnt[cnt >= 0.8 * cnt.max()]
    if ok.index[-1] != cnt.index[-1]:
        log.warning("%s は株価が揃っていない（%d/%d銘柄）ため、%s のデータで判定します",
                    cnt.index[-1].date(), cnt.iloc[-1], int(cnt.max()), ok.index[-1].date())
    return ok.index[-1]


def params_path(cfg: dict):
    return cfg["data"]["db_path"].parent / "strategy_params.json"


def dataset(cfg: dict, pl: pd.DataFrame) -> pd.DataFrame:
    """シグナル表に現在の戦略パラメータを当てはめた学習データ。
    AIの学習目標は全シグナル共通の短期ルール（共通の売買ルールでのラベル）のままにする。
    戦略ごとのルールで付け直すと、遠い利確ラインの戦略が「失敗」扱いになって順位付けが崩れた（ジャーナル ラウンド5）。"""
    return apply_flags(pl, load_params(params_path(cfg)))


def store_of(cfg: dict) -> data.PriceStore:
    return data.PriceStore(cfg["data"]["price_dir"])


def run_daily(cfg: dict, force: bool = False, retrain: bool = False, notify_on: bool = True,
              collect: bool = True) -> dict | None:
    con = connect(cfg["data"]["db_path"])
    wait_for_close_data(cfg)
    uni = data.refresh_universe(con, cfg)
    names = dict(zip(uni["code"], uni["name"]))
    store = store_of(cfg)
    proxy = cfg["universe"]["market_proxy"]
    data.update_prices(store, uni["code"].tolist() + [proxy], cfg)
    _drop_legacy_prices(con)

    cutoff = _usable_date(store, _cutoff())
    data_date = cutoff.strftime("%Y-%m-%d")
    done = con.execute("SELECT status FROM runs WHERE data_date=?", (data_date,)).fetchone()
    if done and done[0] == "ok" and not force:
        log.info("%s は処理済み（休場日か二重実行）。--force で再実行できます", data_date)
        return None
    log.info("指標とシグナルを計算中…（データ日付 %s）", data_date)
    ind, pl = pool.refresh(store, cfg, uni, cutoff)
    ds = dataset(cfg, pl)
    in_uni = ind["close"].columns.isin(uni["code"])
    n_scanned = int(((ind["turnover"].iloc[-1] >= cfg["universe"]["min_turnover_jpy"])
                     & (ind["close"].iloc[-1] >= cfg["universe"]["min_price"]) & in_uni).sum())
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
    plans = [plan_for(k.split(","), cfg["labeling"]) for k in top["strategies"]]
    top["tp_atr"] = [p["tp_atr"] for p in plans]
    top["sl_atr"] = [p["sl_atr"] for p in plans]
    top["horizon"] = [p["horizon_days"] for p in plans]
    top["tp"] = top["close"] + top["tp_atr"] * top["atr"]
    top["sl"] = top["close"] - top["sl_atr"] * top["atr"]
    n_e = cfg["screening"]["earnings_check_n"]
    top["earnings_days"] = [(_earnings_days(c, last) if i < n_e else np.nan) for i, c in enumerate(top["code"])]
    top["rank"] = range(1, len(top) + 1)
    snap = flows.snapshot(cfg)
    for col in ("margin_ratio", "buy_pct", "sell_pct", "buy_pct_chg5", "short_total"):
        top[col] = top["code"].map(snap[col]) if col in snap else np.nan
    _store_live(con, top, data_date)
    _label_live(con, ds)

    event = regimes.event_of([last])[0]
    crisis = bool(event) or float(ind["stress"].iloc[-1].iloc[0]) > 0.5
    regime = dict(crisis=crisis, event=event or ("市場の急変を自動検知" if crisis else ""))
    pt = paper.step(con, cfg, ind, top, last, crisis=crisis) if cfg["paper_trading"]["enabled"] else None
    rep = report.build(cfg, con, ind, top, len(today), n_scanned, bundle, weights, pt, data_date, regime)
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
    if collect:
        collect_extra(cfg, uni)
    return dict(date=data_date, top=top, n_signals=len(today))


def run_research(cfg: dict, research: bool = True, trials=None, budget_min=None, notify_on=True) -> None:
    """週末用: 研究ループ → （条件が変わったら）再学習 → 過去ペーパートレード → Webページ → 通知。"""
    from . import backtest, research as rs
    con = connect(cfg["data"]["db_path"])
    if research:  # 週末は時間に余裕があるので、追加データの初回の大量取得をここで進める
        uni = data.refresh_universe(con, cfg)
        data.update_prices(store_of(cfg), uni["code"].tolist() + [cfg["universe"]["market_proxy"]], cfg)
        for name, fn in (("決算情報", lambda: earnings.update(cfg, uni["code"].tolist(), max_minutes=60)),
                         ("適時開示", lambda: tdnet.update(cfg, max_minutes=120))):
            try:
                fn()
            except Exception as e:
                log.warning("%sの更新に失敗: %s", name, e)
        pool.refresh(store_of(cfg), cfg, uni, _usable_date(store_of(cfg), _cutoff()))
    pl = pool.load(cfg)
    if pl is None:
        raise SystemExit("シグナル表がありません。先に daily を実行してください")
    summary = rs.run(con, cfg, trials=trials, budget_min=budget_min) if research else None
    if summary:
        # 研究で採用された条件・売買ルールをこの後の再学習と過去ペーパートレードに反映するため、設定を読み直す
        from .config import load_config
        cfg = load_config()
    if summary and summary.get("plan_changed"):
        # 共通の売買ルールが変わったら、新しいルールで25年分のラベルを付け直す
        log.info("売買ルールを更新: %s → シグナル表を作り直します", summary["plan"]["candidate"])
        pl = pool.build_full(store_of(cfg), cfg, data.refresh_universe(con, cfg))
    ds = dataset(cfg, pl)
    if research:  # 週1回は必ず再学習（新旧モデルの対決は train 内で行う）
        log.info("再学習中…" + ("（戦略の条件が更新されたため）" if summary and summary["changed"] else ""))
        learner.train(con, cfg, ds, pl["date"].max().strftime("%Y-%m-%d"))
    bt = backtest.run(cfg, ds)
    if research:  # 地合いによる買い控え（過去ペーパートレードの予測をそのまま使って比較）
        g = rs.gate_search(con, cfg, bt["_preds"], bt["_cal"])
        pf = rs.portfolio_search(con, cfg, bt["_preds"], bt["_cal"])
        rj = cfg["data"]["db_path"].parent / "research.json"
        if rj.exists():
            j = json.loads(rj.read_text())
            j["gate"], j["portfolio"] = g, pf
            rj.write_text(json.dumps(j, ensure_ascii=False, default=float))
        if summary is not None:
            summary["gate"], summary["portfolio"] = g, pf
        if g["adopted"] or pf["adopted"]:
            from .config import load_config
            cfg = load_config()
            bt = backtest.run(cfg, ds, preds=bt["_preds"])
    site.build(con, cfg)
    if notify_on:
        notify.send_research(cfg, summary, bt)


def collect_extra(cfg: dict, uni: pd.DataFrame) -> None:
    """通知の後に、翌日以降に使う追加データを集める（初回の大量取得で通知が遅れないように）。"""
    lim = cfg.get("earnings", {}).get("max_minutes", 30)
    for name, fn in (("決算情報", lambda: earnings.update(cfg, uni["code"].tolist(), max_minutes=lim)),
                     ("需給データ", lambda: flows.collect(cfg)),
                     ("適時開示", lambda: tdnet.update(cfg, max_minutes=lim))):
        try:
            fn()
        except Exception as e:
            log.warning("%sの更新に失敗: %s", name, e)


def _drop_legacy_prices(con) -> None:
    """旧バージョン（SQLite に株価を保存していた頃）のテーブルを空にして容量を空ける。"""
    if con.execute("SELECT 1 FROM prices LIMIT 1").fetchone():
        con.execute("DELETE FROM prices")
        con.commit()
        con.execute("VACUUM")


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
