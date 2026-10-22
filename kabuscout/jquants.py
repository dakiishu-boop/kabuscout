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
import time

import pandas as pd
import requests

log = logging.getLogger(__name__)
API = "https://api.jquants.com/v2"
_last = [0.0]


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


def update_fins(cfg: dict, max_minutes: float = 20) -> dict:
    """決算短信サマリを開示日ごとに取得して貯める（新しい日から順に。済んだ日は記録して飛ばす）。"""
    if not _key():
        return dict(skipped="no key")
    d = _dir(cfg)
    path, state_p = d / "fins_summary.parquet", d / "fins_state.json"
    state = json.loads(state_p.read_text()) if state_p.exists() else dict(done=[], empty=[])
    done = set(state["done"]) | set(state.get("empty", []))
    jc = cfg.get("jquants", {})
    today = pd.Timestamp.now(tz="Asia/Tokyo").tz_localize(None).normalize()
    start = today - pd.Timedelta(days=int(jc.get("history_days", 760)))
    days = [x.strftime("%Y%m%d") for x in pd.bdate_range(start, today)][::-1]
    # 直近の数日は開示が後から増えることがあるので取り直す
    recent = set(days[:5])
    todo = [x for x in days if x not in done or x in recent]
    t0, got, new = time.time(), [], 0
    for day in todo:
        if time.time() - t0 > max_minutes * 60:
            break
        st, rows = _get(cfg, "/fins/summary", {"date": day})
        if st == 400:  # 契約プランの範囲外の日付（古すぎる・新しすぎる）
            state.setdefault("empty", []).append(day)
            continue
        if st in (401, 403):  # キーが無効・プランで使えない → 日付を「空」と記録せずに止める
            log.warning("J-Quants 決算サマリが使えません（%d）。キーやプランを確認してください", st)
            break
        if st != 200:
            log.warning("J-Quants 決算サマリ %s の取得に失敗（%d）→ 次回に再試行", day, st)
            break
        if rows:
            got.append(pd.DataFrame(rows))
            new += len(rows)
            state["done"].append(day)
        else:
            state.setdefault("empty", []).append(day)
        if len(got) >= 20:  # 途中で止まっても失わないようにこまめに保存
            _save(path, got)
            got = []
            state_p.write_text(json.dumps(_dedup(state)))
    if got:
        _save(path, got)
    state_p.write_text(json.dumps(_dedup(state)))
    n = len(pd.read_parquet(path)) if path.exists() else 0
    log.info("J-Quants 決算サマリ: 今回 %d件追加 → 合計 %d件（取得済み %d日）", new, n, len(set(state["done"])))
    return dict(new=new, total=n, days=len(set(state["done"])))


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
