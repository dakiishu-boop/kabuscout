"""J-Quants API V2（JPX公式・環境変数 JQUANTS_API_KEY）。

Free プランは1分あたり5回まで・過去約2年・約12週遅れ（プラン内容は probe で実際に確かめる）。
取得は間隔を空けて1回ずつ行い、毎日・毎週の実行で少しずつ貯める。

- 決算短信のサマリ（/fins/summary）: 実績と会社の業績予想（今期・来期）、開示日つき
  → 予想PER（予想EPS÷株価）、業績予想の修正、通期予想に対する進捗率など
- 先読みを防ぐため、各開示は「開示日 + データの遅れ（Free は約12週）」から使えるものとして扱う。
  本番で手に入るのは遅れたデータなので、研究でも同じだけ遅らせないと本番と食い違う
"""
from __future__ import annotations

import json
import logging
import os
import re
import time

import pandas as pd
import requests

log = logging.getLogger(__name__)
API = "https://api.jquants.com/v2"
_last = [0.0]
_msg = [""]  # 直近のエラーメッセージ（契約プランの対象期間が書かれている）


def _key() -> str | None:
    return os.environ.get("JQUANTS_API_KEY") or None


def _dir(cfg):
    d = cfg["data"]["db_path"].parent / "jquants"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _get(cfg: dict, path: str, params: dict | None = None) -> tuple[int, list[dict]]:
    """1ページずつ間隔を空けて取得し、pagination_key があれば続きも取る。(HTTPステータス, 行) を返す。"""
    gap = 60.0 / cfg.get("jquants", {}).get("requests_per_min", 5) + 0.5
    rows: list[dict] = []
    q = dict(params or {})
    while True:
        for attempt in range(4):
            wait = gap - (time.time() - _last[0])
            if wait > 0:
                time.sleep(wait)
            _last[0] = time.time()
            try:
                r = requests.get(API + path, params=q, headers={"x-api-key": _key()}, timeout=60)
            except requests.RequestException as e:
                log.warning("J-Quants %s: 通信エラー（%s）", path, type(e).__name__)
                time.sleep(10 * (attempt + 1))
                continue
            if r.status_code == 429:  # 回数制限 → 少し待って再試行
                time.sleep(30 * (attempt + 1))
                continue
            break
        else:
            return 429, rows
        if r.status_code != 200:
            msg = ""
            try:
                msg = str(r.json().get("message", ""))[:200]
            except ValueError:
                pass
            _msg[0] = msg
            log.info("J-Quants %s %s → %d %s", path, q, r.status_code, msg)
            return r.status_code, rows
        js = r.json()
        rows += js.get("data", [])
        if not js.get("pagination_key"):
            return 200, rows
        q["pagination_key"] = js["pagination_key"]


def probe(cfg: dict) -> dict:
    """どのデータが、どの期間まで取れるかを確かめる（契約プランによって違うため）。"""
    if not _key():
        log.info("JQUANTS_API_KEY が未設定のため J-Quants を飛ばします")
        return {}
    today = pd.Timestamp.now(tz="Asia/Tokyo").tz_localize(None).normalize()
    d = lambda days: (today - pd.Timedelta(days=days)).strftime("%Y%m%d")
    tests = {
        "listed": ("/equities/master", {}),
        "fins_recent": ("/fins/summary", {"date": d(7)}),
        "fins_13w": ("/fins/summary", {"date": d(95)}),
        "fins_toyota": ("/fins/summary", {"code": "72030"}),
        "valuation_toyota": ("/equities/valuation", {"code": "72030"}),
        "earnings_calendar": ("/equities/earnings-calendar", {}),
        "investor_types": ("/equities/investor-types", {}),
        "bars_toyota": ("/equities/bars/daily", {"code": "72030"}),
        "margin_toyota": ("/markets/margin-interest", {"code": "72030"}),
        "breakdown_toyota": ("/markets/breakdown", {"code": "72030"}),
        "short_ratio": ("/markets/short-ratio", {"date": d(100)}),
        "dividend_toyota": ("/fins/dividend", {"code": "72030"}),
    }
    out = {}
    for name, (path, params) in tests.items():
        st, rows = _get(cfg, path, params)
        info = dict(status=st, rows=len(rows))
        if rows:
            df = pd.DataFrame(rows)
            info["columns"] = list(df.columns)[:40]
            for c in ("Date", "DiscDate", "PubDate", "EnDate"):
                if c in df:
                    info["from"], info["to"] = str(df[c].min()), str(df[c].max())
                    break
        out[name] = info
        log.info("J-Quants %-18s %s 件数=%s 期間=%s〜%s", name, st, info["rows"], info.get("from", "-"), info.get("to", "-"))
    (_dir(cfg) / "probe.json").write_text(json.dumps(dict(at=str(pd.Timestamp.now()), tests=out), ensure_ascii=False, indent=1))
    return out


def coverage(cfg: dict) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """契約プランで取れる期間（Free は約2年・約12週遅れ）。範囲外の日を頼むと、エラーに期間が書かれて返ってくる。"""
    today = pd.Timestamp.now(tz="Asia/Tokyo").tz_localize(None).normalize()
    st, _ = _get(cfg, "/fins/summary", {"date": today.strftime("%Y%m%d")})
    m = re.search(r"(\d{4}-\d{2}-\d{2})\s*~\s*(\d{4}-\d{2}-\d{2})", _msg[0]) if st == 400 else None
    if m:
        return pd.Timestamp(m.group(1)), pd.Timestamp(m.group(2))
    if st == 200:  # 遅れの無いプラン
        return today - pd.Timedelta(days=int(cfg.get("jquants", {}).get("history_days", 760))), today
    return None


def update_fins(cfg: dict, max_minutes: float = 20) -> dict:
    """決算短信サマリを開示日ごとに取得して貯める（新しい日から順に。済んだ日は記録して飛ばす）。
    対象期間の終わりは毎日1日ずつ進むので、まだ範囲に入っていない日は記録せず、範囲に入ってから取る。"""
    if not _key():
        return dict(skipped="no key")
    d = _dir(cfg)
    path, state_p = d / "fins_summary.parquet", d / "fins_state.json"
    state = json.loads(state_p.read_text()) if state_p.exists() else dict(done=[], empty=[])
    cov = coverage(cfg)
    if cov is None:
        log.warning("J-Quants 決算サマリの対象期間が分かりませんでした（キーやプランを確認）")
        return dict(error="coverage")
    lo, hi = (x.strftime("%Y%m%d") for x in cov)
    state["empty"] = [x for x in state.get("empty", []) if x <= hi]  # 以前に範囲外を「空」と記録してしまった日を戻す
    state["coverage"] = [lo, hi]
    done = set(state["done"]) | set(state["empty"])
    days = [x.strftime("%Y%m%d") for x in pd.bdate_range(cov[0], cov[1])][::-1]
    todo = [x for x in days if x not in done or x in set(days[:3])]
    t0, got, new = time.time(), [], 0
    for day in todo:
        if time.time() - t0 > max_minutes * 60:
            break
        st, rows = _get(cfg, "/fins/summary", {"date": day})
        if st != 200:
            log.warning("J-Quants 決算サマリ %s の取得に失敗（%d）→ 次回に再試行", day, st)
            break
        if rows:
            got.append(pd.DataFrame(rows))
            new += len(rows)
            state["done"].append(day)
        else:  # 休日など開示の無い日
            state["empty"].append(day)
        if len(got) >= 20:  # 途中で止まっても失わないようにこまめに保存
            _save(path, got)
            got = []
            state_p.write_text(json.dumps(_dedup(state)))
    if got:
        _save(path, got)
    state_p.write_text(json.dumps(_dedup(state)))
    n = len(pd.read_parquet(path)) if path.exists() else 0
    left = len([x for x in days if x not in set(state["done"]) | set(state["empty"])])
    log.info("J-Quants 決算サマリ（対象 %s〜%s）: 今回 %d件追加 → 合計 %d件（取得済み %d日・残り %d日）",
             lo, hi, new, n, len(set(state["done"])), left)
    return dict(new=new, total=n, days=len(set(state["done"])), left=left, coverage=[lo, hi])


def _dedup(state: dict) -> dict:
    return {k: sorted(set(v)) for k, v in state.items()}


def _save(path, frames: list[pd.DataFrame]) -> None:
    df = pd.concat(frames, ignore_index=True).astype(str)
    if path.exists():
        df = pd.concat([pd.read_parquet(path), df], ignore_index=True)
    keys = [c for c in ("DiscNo", "Code", "DiscDate") if c in df]
    df.drop_duplicates(keys or None, keep="last").to_parquet(path)


def load_fins(cfg: dict) -> pd.DataFrame | None:
    p = _dir(cfg) / "fins_summary.parquet"
    return pd.read_parquet(p) if p.exists() else None


# ---------------------------------------------------------------- 特徴量
# fwd_ey: 会社予想の1株利益 ÷ 株価（予想PERの逆数。高いほど割安）
# fc_rev: 同じ決算期の会社予想（1株利益）の、前回の開示からの修正率
# progress_gap: 通期予想の純利益に対する累計の進捗 − 経過した四半期の目安（1Q 25%・2Q 50%・3Q 75%）
# sales_yoy: 売上高の前年同期比
JQ_FEATURES = ["fwd_ey", "fc_rev", "progress_gap", "sales_yoy"]
_QTR = {"1Q": 0.25, "2Q": 0.5, "3Q": 0.75, "FY": 1.0}


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.replace({"": None, "None": None, "nan": None}), errors="coerce")


def events_table(fins: pd.DataFrame, lag_days: int) -> pd.DataFrame:
    """開示ごとの値（使える日 = 開示日 + データの遅れ）。"""
    f = fins.copy()
    f["code"] = f["Code"].str[:4]
    f["disc"] = pd.to_datetime(f["DiscDate"], errors="coerce")
    f["fy_end"] = pd.to_datetime(f["CurFYEn"], errors="coerce")
    f["per_end"] = pd.to_datetime(f["CurPerEn"], errors="coerce")
    for c in ("FEPS", "NP", "FNP", "Sales"):
        f[c] = _num(f[c]) if c in f else float("nan")
    f = f.dropna(subset=["disc"]).sort_values(["code", "disc", "DiscNo"])
    # 予想の修正率（同じ決算期の、前回の開示の予想と比べる）
    prev = f.groupby(["code", "fy_end"])["FEPS"].shift(1)
    f["fc_rev"] = ((f["FEPS"] - prev) / prev.abs()).where(prev.abs() > 0).clip(-1, 1)
    # 進捗率（決算短信のみ。業績予想の修正だけの開示は実績が無い）
    q = f["CurPerType"].map(_QTR)
    f["progress_gap"] = (f["NP"] / f["FNP"] - q).where((f["FNP"] > 0) & q.notna() & f["NP"].notna()).clip(-1, 1)
    # 前年同期比（同じ四半期の種類で、1年前の期末の開示と比べる）
    k = f.dropna(subset=["Sales", "per_end"]).drop_duplicates(["code", "CurPerType", "per_end"], keep="last")
    last_year = k.assign(per_end=k["per_end"] + pd.DateOffset(years=1))[["code", "CurPerType", "per_end", "Sales"]]
    f = f.merge(last_year.rename(columns={"Sales": "sales_ly"}), on=["code", "CurPerType", "per_end"], how="left")
    f["sales_yoy"] = (f["Sales"] / f["sales_ly"] - 1).where(f["sales_ly"] > 0).clip(-1, 3)
    f["known"] = f["disc"] + pd.Timedelta(days=lag_days)
    return f[["code", "known", "FEPS", "fc_rev", "progress_gap", "sales_yoy"]]


def wide_features(fins: pd.DataFrame | None, close: pd.DataFrame, lag_days: int = 84) -> dict[str, pd.DataFrame]:
    idx, cols = close.index, close.columns
    empty = lambda: pd.DataFrame(float("nan"), index=idx, columns=cols, dtype="float32")
    if fins is None or fins.empty:
        return {f: empty() for f in JQ_FEATURES}
    ev = events_table(fins, lag_days)
    ev = ev[ev["code"].isin(cols) & (ev["known"] <= idx[-1])]
    ev = ev.assign(pos=idx.searchsorted(ev["known"])).drop_duplicates(["code", "pos"], keep="last")
    out = {}
    for name in ("FEPS", "fc_rev", "progress_gap", "sales_yoy"):
        w = ev.dropna(subset=[name]).pivot(index="pos", columns="code", values=name)
        # 修正率は「修正があった直後」の情報なので60営業日で切る。それ以外は次の開示まで（最長300営業日）
        w = w.reindex(index=range(len(idx)), columns=cols).ffill(limit=60 if name == "fc_rev" else 300)
        w.index = idx
        out[name] = w
    res = {"fwd_ey": (out["FEPS"] / close).clip(-0.5, 0.5), "fc_rev": out["fc_rev"],
           "progress_gap": out["progress_gap"], "sales_yoy": out["sales_yoy"]}
    return {k: v.astype("float32") for k, v in res.items()}
