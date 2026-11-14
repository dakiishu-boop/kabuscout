"""適時開示（TDnet）: 全上場企業の開示のタイトルと時刻（2010年頃〜）。

TDnet 本体は直近1ヶ月分しか公開していないため、過去分は有志の無料API（やのしん TDnet WEB-API）から取得する。
非公式サービスなので、1日分ずつ間隔を空けて取得し、手元に保存する。

タイトルから次の出来事を判定して使う:
- 決算短信（全銘柄の決算発表日。Yahoo の決算情報が無い銘柄も補える）
- 自己株式の取得（自社株買い）の決定 … 需給の改善・株主還元
- 業績予想の上方/下方修正、増配
- 株式分割、株主優待の新設
- 公募増資・新株発行（株式の希薄化）
"""
from __future__ import annotations

import json
import logging
import re
import time

import numpy as np
import pandas as pd
import requests

log = logging.getLogger(__name__)
API = "https://webapi.yanoshin.jp/webapi/tdnet/list/{d}.json?limit=5000"
UA = {"User-Agent": "KabuScout/1.0 (personal research; polite 1 req/s)"}

EVENT_RULES = [
    ("earnings", r"決算短信"),
    ("buyback", r"自己株式(の)?取得に係る事項の決定|自己株式の取得(及び|ならびに|並びに)?.*(決定|実施)|自己株式の公開買付け"),
    ("forecast_up", r"上方修正"),
    ("forecast_down", r"下方修正"),
    ("dividend_up", r"増配"),
    ("split", r"株式分割"),
    ("yutai_new", r"株主優待.*(新設|導入)"),
    ("offering", r"公募|新株式発行|第三者割当|新株予約権.*発行"),
]
EXCLUDE = re.compile(r"訂正|取得状況|取得終了|結果|中止|延期|お詫び")
# 自社の株が買収される（TOB・MBO・締め出し）発表。株価が買付価格に張り付くので、成立して上場廃止になるまで買っても値幅が無い。
# 過去の株価データには成立して上場廃止になった銘柄が無い（生存者バイアス）ので、過去ペーパートレードには出てこない＝本番だけの穴（§75）。
# Claude で分類した 2010〜2026年の題名1.6万件と比べて、案件単位で 2024年以降の 84% を拾い、当たりの 85% が本物
TOB_TARGET = re.compile(r"(当社|当社の)(普通)?(株式|株券)(等)?(に対する|の)公開買付|ＭＢＯ|MBO|マネジメント・バイアウト|株式等売渡請求|"
                        r"単元株式数の定めの廃止|当社株式の上場廃止")
TOB_NOT = re.compile(r"訂正|不成立|撤回|中止|自己株式|公開買付けへの応募|子会社(?!化)|名古屋|札幌|福岡|プロマーケット|PRO Market|上場廃止申請|反対")
TOB_END = re.compile(r"(当社|当社の)(普通)?(株式|株券).{0,30}公開買付.{0,40}(不成立|撤回|中止)")
# 業績予想の修正・予想と実績の差異（研究の「修正好反応」と同じ拾い方。訂正だけ除く）。将来の検証（前向きテスト）のために題名を残すだけで、
# events.EVENT_TYPES には入れないので AI の特徴量・戦略は変わらない（§84）
REVISION = re.compile(r"業績予想.*修正|上方修正|下方修正|予想.{0,8}実績.{0,8}差異")
# 分類のルールを足したとき、直近400日だけ一度取り直す（目印のファイル名。足すたびに1つ増やす）
RESCANS = ["tdnet_rescan_tob_v1"]   # 買収の発表（tob_target/tob_end）
# 業績予想の修正は、比較用のAIの学習に2010年からの過去分が要るので、全期間を一度だけ取り直す（backfill_revisions。直近400日の取り直しはしない）
BACKFILL = "tdnet_rev_backfill.json"   # {"until": 取り直す最後の日, "done": [取り直した日]}


def classify(title: str) -> list[str]:
    t = title or ""
    if TOB_END.search(t) and "訂正" not in t and "自己株式" not in t:
        return ["tob_end"]
    out = ["tob_target"] if TOB_TARGET.search(t) and not TOB_NOT.search(t) else []
    rev = ["revision"] if REVISION.search(t) and "訂正" not in t else []
    if EXCLUDE.search(t):
        return out + rev
    return out + [k for k, pat in EVENT_RULES if re.search(pat, t)] + rev


def pending_buyouts(td: pd.DataFrame | None, today: pd.Timestamp, days: int = 365) -> dict[str, str]:
    """買収（TOB・MBO・締め出し）の発表から days 日以内で、不成立・撤回の発表が無い銘柄 → {コード: 説明}。"""
    if td is None or td.empty:
        return {}
    since = today - pd.Timedelta(days=days)
    ev = td[(td["pubdate"] >= since) & (td["pubdate"] <= today + pd.Timedelta(days=1))]
    start = ev[ev["events"].str.contains("tob_target")].groupby("code")["pubdate"].max()
    end = ev[ev["events"].str.contains("tob_end")].groupby("code")["pubdate"].max()
    return {c: f"買収の発表あり（{d:%Y/%m/%d}）" for c, d in start.items() if not (c in end and end[c] > d)}


def _path(cfg):
    return cfg["data"]["db_path"].parent / "tdnet.parquet"


def load(cfg: dict) -> pd.DataFrame | None:
    p = _path(cfg)
    return pd.read_parquet(p) if p.exists() else None


def update(cfg: dict, start: str = "2010-01-01", max_minutes: float | None = None) -> pd.DataFrame:
    """未取得の日を1日ずつ取得（初回は約1時間。時間切れなら次回の続きから）。"""
    p = _path(cfg)
    old = pd.read_parquet(p) if p.exists() else pd.DataFrame(columns=["pubdate", "code", "title", "events"])
    meta_p = p.with_suffix(".json")
    raw = json.loads(meta_p.read_text()) if meta_p.exists() else []
    done = set(raw.values() if isinstance(raw, dict) else raw)
    today = pd.Timestamp.today().normalize()
    todo = [p.with_name(n) for n in RESCANS if not p.with_name(n).exists()]
    if todo and done:  # 分類のルールを足したので、直近400日だけ一度取り直す（前は保存しなかった題名を拾うため）
        done -= {d.strftime("%Y%m%d") for d in pd.date_range(today - pd.Timedelta(days=400), today)}
        meta_p.write_text(json.dumps(sorted(done)))
        for r in todo:
            r.touch()
    n = _reclassify(old, today - pd.Timedelta(days=400))
    if n:
        old.to_parquet(p)
        log.info("  適時開示: 手元の題名 %d件を今のルールで分類し直した", n)
    days = [d for d in pd.date_range(start, today) if d.strftime("%Y%m%d") not in done or d >= today - pd.Timedelta(days=3)]
    t0, rows = time.time(), []
    for i, d in enumerate(days):
        if max_minutes and time.time() - t0 > max_minutes * 60:
            log.info("  適時開示: 時間切れ（%d/%d日、続きは次回）", i, len(days))
            break
        key = d.strftime("%Y%m%d")
        try:
            r = requests.get(API.format(d=key), headers=UA, timeout=30)
            r.raise_for_status()
            for it in r.json().get("items", []):
                t = it.get("Tdnet", it)  # 件数が多い日は {"Tdnet": {...}} で包まれずに返ってくる
                ev = classify(t["title"])
                if ev:
                    rows.append(dict(pubdate=pd.Timestamp(t["pubdate"]), code=str(t["company_code"])[:4],
                                     title=t["title"], events=",".join(ev)))
            if d < today - pd.Timedelta(days=3):
                done.add(key)
        except Exception as e:
            log.warning("  適時開示の取得失敗 %s: %s", key, e)
            time.sleep(10)
        if i % 250 == 0:
            log.info("  適時開示 %d / %d 日", i, len(days))
            if rows:  # 途中経過も保存（中断しても取得済みの分は残す）
                old = _save(p, meta_p, old, rows, done)
                rows = []
        time.sleep(1.0)
    return _save(p, meta_p, old, rows, done)


def _reclassify(old: pd.DataFrame, since: pd.Timestamp) -> int:
    """手元にある since 以降の題名を、今のルールで分類し直す（通信なし）。直した件数を返す。
    取り直しでは、すでに別の分類（例: offering）で保存してあった題名に tob_target が付かなかった（ジャーナル 79）。"""
    m = old["pubdate"] >= since
    if not m.any():
        return 0
    now = old.loc[m, "title"].map(lambda t: ",".join(classify(t)))
    ch = now[now.ne(old.loc[m, "events"]) & now.ne("")]
    old.loc[ch.index, "events"] = ch
    return len(ch)


def _save(p, meta_p, old, rows, done):
    if rows:  # 同じ題名を取り直したときは、新しい分類のほうを残す
        old = pd.concat([old, pd.DataFrame(rows)], ignore_index=True).drop_duplicates(["pubdate", "code", "title"], keep="last")
        old.to_parquet(p)
    meta_p.write_text(json.dumps(sorted(done)))
    return old


def _backfill_state(cfg: dict) -> tuple:
    p = _path(cfg).with_name(BACKFILL)
    return p, (json.loads(p.read_text()) if p.exists() else {})


def revisions_ready(cfg: dict, start: str = "2010-01-01") -> bool:
    """業績修正の題名の過去分がそろったか（backfill_revisions が最後の日まで取り終えた）。"""
    _, st = _backfill_state(cfg)
    if not st.get("until"):
        return False
    done = set(st.get("done") or [])
    return all(d.strftime("%Y%m%d") in done for d in pd.date_range(start, st["until"]))


def backfill_progress(cfg: dict, start: str = "2010-01-01") -> tuple[int, int]:
    """(取り直した日数, 全部の日数)。"""
    _, st = _backfill_state(cfg)
    days = pd.date_range(start, st.get("until") or pd.Timestamp.today().normalize())
    done = set(st.get("done") or [])
    return sum(d.strftime("%Y%m%d") in done for d in days), len(days)


def _merge_revisions(old: pd.DataFrame, rows: list[dict]) -> pd.DataFrame:
    """取り直した業績修正の題名を足す。保存済みの題名には revision の印を足すだけ、無かった題名は revision だけで足す
    （ほかの分類は変えない＝本番の特徴量・候補は変わらない）。"""
    if not rows:
        return old
    new = pd.DataFrame(rows).drop_duplicates(["pubdate", "code", "title"])
    k = ["pubdate", "code", "title"]
    k_old, k_new = pd.MultiIndex.from_frame(old[k]), pd.MultiIndex.from_frame(new[k])
    m = k_old.isin(k_new) & ~old["events"].fillna("").str.contains("revision")
    ev = old.loc[m, "events"].fillna("")
    old.loc[m, "events"] = np.where(ev == "", "revision", ev + ",revision")
    return pd.concat([old, new[~k_new.isin(k_old)].assign(events="revision")], ignore_index=True)


def backfill_revisions(cfg: dict, start: str = "2010-01-01", max_minutes: float | None = None,
                       today: pd.Timestamp | None = None) -> int:
    """業績修正の題名の過去分（2010年〜、初めて動いた日まで）を一度だけ取り直して足す。時間切れなら次回の続きから
    （1日1回の取得・1秒おき。全部で約6,200回）。それより後の日は、ふだんの update が今のルール（revision を含む）で取る。足した件数を返す。"""
    p, st = _backfill_state(cfg)
    until = pd.Timestamp(st.get("until") or (today or pd.Timestamp.today()).normalize())
    done = set(st.get("done") or [])
    days = [d for d in pd.date_range(start, until) if d.strftime("%Y%m%d") not in done]
    save_state = lambda: p.write_text(json.dumps(dict(until=until.strftime("%Y-%m-%d"), done=sorted(done))))
    if not days:
        save_state()
        return 0
    tp = _path(cfg)
    old = pd.read_parquet(tp) if tp.exists() else pd.DataFrame(columns=["pubdate", "code", "title", "events"])
    t0, rows, n = time.time(), [], 0

    def flush(old, rows):
        old = _merge_revisions(old, rows)
        old.to_parquet(tp)
        save_state()
        return old

    for i, d in enumerate(days):
        if max_minutes and time.time() - t0 > max_minutes * 60:
            log.info("  業績修正の過去分: 時間切れ（%d/%d日、続きは次回）", i, len(days))
            break
        key = d.strftime("%Y%m%d")
        try:
            r = requests.get(API.format(d=key), headers=UA, timeout=30)
            r.raise_for_status()
            for it in r.json().get("items", []):
                t = it.get("Tdnet", it)
                if "revision" in classify(t["title"]):
                    rows.append(dict(pubdate=pd.Timestamp(t["pubdate"]), code=str(t["company_code"])[:4], title=t["title"]))
            done.add(key)
        except Exception as e:
            log.warning("  業績修正の過去分の取得失敗 %s: %s", key, e)
            time.sleep(10)
        if i % 250 == 0 and rows:  # 途中経過も保存（中断しても取得済みの分は残す）
            n += len(rows)
            old, rows = flush(old, rows), []
            log.info("  業績修正の過去分 %d / %d 日", i, len(days))
        time.sleep(1.0)
    n += len(rows)
    flush(old, rows)
    return n
