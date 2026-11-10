"""決算短信の「定性的情報」（経営成績の概況・見通し）を Claude で読み、決算の中身の質を貯める（研究用・Mac で実行）。

決算好反応（PEAD）はいちばん稼いでいる戦略だが、使っているのは数字（XBRL・株価の反応）だけで、
「本業が強くて良かったのか、一度きりの利益なのか」「会社が見通しを保守的に置いているか」は分からない。
TDnet は書類を約1ヶ月で消すので、11月の決算シーズンの分は今集めないと二度と取れない。
Claude のモデルは 2026年6月以降のことを知らないので、2026年秋の決算での検証は「答えを覚えている」問題が無い（§69）。
検証は発表から20〜60営業日たってから（2026年12月〜2027年1月）。それまでは本番に使わない。

- 1つの短信の最初の PAGES ページ（サマリーと定性的情報）だけを送る
- 元の PDF は保存せず、判定だけを data/tdnet_quals/ に置く（公開しない）
- モデルは環境変数 KABUSCOUT_QUAL_MODEL（既定 claude-sonnet-5-5。Opus との一致率と費用で選んだ。ジャーナル §75）
"""
from __future__ import annotations

import io
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

from .tdnet import API, UA
from .tdnet_reasons import DRIVERS, _client, api_stop_reason, check_api, notify_stop

log = logging.getLogger(__name__)
TITLE = re.compile(r"決算短信")
SKIP = re.compile(r"訂正|REIT|リート|投資法人")
PAGES = 7
DEFAULT_MODEL = "claude-sonnet-5-5"
NUM = {"type": ["number", "null"]}
SCHEMA = {
    "type": "object",
    "properties": {
        "period": {"type": "string", "enum": ["Q1", "Q2", "Q3", "FY", "other"]},
        "guidance_change": {"type": "string", "enum": ["raised", "lowered", "mixed", "unchanged", "withdrawn", "new", "none"]},
        "progress": {"type": "string", "enum": ["ahead", "on_track", "behind", "not_mentioned"]},
        "main_driver": {"type": "string", "enum": DRIVERS},
        "second_driver": {"type": "string", "enum": DRIVERS},
        "persistence": {"type": "string", "enum": ["recurring", "one_off", "mixed", "unclear"]},
        "one_off_in_profit": {"type": "boolean"},
        "demand_trend": {"type": "string", "enum": ["improving", "stable", "deteriorating", "unclear"]},
        "outlook_tone": {"type": "string", "enum": ["positive", "neutral", "negative"]},
        "conservative_guidance": {"type": "boolean"},
        "orders_backlog_up": {"type": ["boolean", "null"]},
        "new_return": {"type": "boolean"},
        "summary_ja": {"type": "string"},
    },
    "required": ["period", "guidance_change", "progress", "main_driver", "second_driver", "persistence", "one_off_in_profit",
                 "demand_trend", "outlook_tone", "conservative_guidance", "orders_backlog_up", "new_return", "summary_ja"],
    "additionalProperties": False,
}
PROMPT = """添付は日本の上場企業の決算短信の最初の数ページ（サマリーと経営成績の概況・見通し）です。書いてある内容だけから判定してください。株価や将来の業績を推測で予想しないこと。

- period: 第1四半期=Q1、第2四半期（中間期）=Q2、第3四半期=Q3、通期（本決算）=FY
- guidance_change: この短信で通期の業績予想（本決算なら来期予想の発表は new）を 上方修正=raised / 下方修正=lowered / 項目で上下が分かれる=mixed / 据え置き=unchanged / 取り下げ・未定に変更=withdrawn / 予想を出していない=none
- progress: 通期予想に対する進捗が「計画を上回る」=ahead、「おおむね計画どおり」=on_track、「計画を下回る」=behind、触れていない=not_mentioned
- main_driver / second_driver: 前年同期と比べた利益の増減の主な理由と2番目の理由（2番目が無ければ none）:
  demand=本業の需要・受注・販売数量 / pricing=販売価格・市況 / cost=原価・費用 / fx=為替 / one_off_gain=資産売却益など一度きりの利益 / one_off_loss=減損など一度きりの損失 / tax=税金 / non_operating=営業外 / m_and_a=買収・連結範囲 / timing=期ずれ / incident=災害・事故 / other=その他
- persistence: その理由が来期以降も続きそうな性質か（recurring / one_off / mixed / unclear）
- one_off_in_profit: 特別損益など一度きりの項目が純利益を大きく動かしていれば true
- demand_trend: 主力事業の需要・受注の流れ（改善=improving / 横ばい=stable / 悪化=deteriorating / 書かれていない=unclear）
- outlook_tone: 会社自身の先行きの書きぶり（positive / neutral / negative）
- conservative_guidance: 上振れを通期予想に織り込んでいない、見通しを保守的に置いていると読めれば true
- orders_backlog_up: 受注高・受注残が前年より増えていると書かれていれば true、減っていれば false、触れていなければ null
- new_return: この短信（同時発表を含む）で増配・自社株買いなど新しい株主還元を示していれば true
- summary_ja: 決算の中身の要約（日本語、70字以内）"""


def _dir(cfg):
    d = cfg["data"]["db_path"].parent / "tdnet_quals"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load(cfg: dict) -> pd.DataFrame | None:
    parts = sorted(_dir(cfg).glob("quals_*.parquet"))
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True).drop_duplicates("id", keep="last") if parts else None


def is_target(title: str) -> bool:
    return bool(TITLE.search(title or "")) and not SKIP.search(title or "")


def first_pages(pdf: bytes, n: int = PAGES) -> bytes:
    from pypdf import PdfReader, PdfWriter
    r = PdfReader(io.BytesIO(pdf))
    if len(r.pages) <= n:
        return pdf
    w = PdfWriter()
    for p in r.pages[:n]:
        w.add_page(p)
    out = io.BytesIO()
    w.write(out)
    return out.getvalue()


def classify_pdf(client, pdf: bytes, model: str) -> tuple[dict, dict]:
    import base64
    content = [{"type": "document", "source": {"type": "base64", "media_type": "application/pdf",
                                               "data": base64.standard_b64encode(pdf).decode()}},
               {"type": "text", "text": PROMPT}]
    r = client.messages.create(model=model, max_tokens=4000, messages=[{"role": "user", "content": content}],
                               output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}})
    if r.stop_reason in ("refusal", "max_tokens"):
        raise RuntimeError(f"stop_reason={r.stop_reason}")
    return json.loads(next(b.text for b in r.content if b.type == "text")), dict(in_tok=r.usage.input_tokens, out_tok=r.usage.output_tokens)


def update(cfg: dict, days: int = 30, max_minutes: float = 120, limit: int | None = None, workers: int = 6) -> dict:
    d = _dir(cfg)
    done_p = d / "done.parquet"
    done = set(pd.read_parquet(done_p)["id"]) if done_p.exists() else set()
    model = os.environ.get("KABUSCOUT_QUAL_MODEL", DEFAULT_MODEL)
    client = _client()
    stop = check_api(client)
    if stop:
        log.error("決算短信の定性情報を中止: %s", stop)
        return dict(targets=0, failed=0, total_docs=len(done), minutes=0.0, model=model, stopped=stop)
    today = pd.Timestamp.today().normalize()
    targets = []
    for k in range(days):
        day = today - pd.Timedelta(days=k)
        try:
            items = [it.get("Tdnet", it) for it in requests.get(API.format(d=day.strftime("%Y%m%d")), headers=UA, timeout=60).json().get("items", [])]
        except Exception as e:
            log.warning("適時開示の一覧 %s: %s", day.date(), e)
            continue
        time.sleep(1.0)
        targets += [t for t in items if t.get("document_url") and is_target(t.get("title")) and str(t.get("id")) not in done]
    targets = targets[: limit or None]
    log.info("決算短信の定性情報: 未処理 %d件（モデル %s）", len(targets), model)
    t0, rows, new_ids, fails, futs = time.time(), [], [], 0, []

    def flush():
        nonlocal rows, new_ids
        if rows:
            pd.DataFrame(rows).to_parquet(d / f"quals_{pd.Timestamp.now():%Y%m%d_%H%M%S_%f}.parquet")
        if new_ids:
            done.update(new_ids)
            pd.DataFrame({"id": sorted(done)}).to_parquet(done_p)
        rows, new_ids = [], []

    def work(t, pdf):
        res, use = classify_pdf(client, first_pages(pdf), model)
        return dict(id=str(t["id"]), code=str(t.get("company_code") or "")[:4], pubdate=pd.Timestamp(t["pubdate"]),
                    title=t["title"], model=model, **use, **res)

    def collect(f, t):
        nonlocal fails, stop
        try:
            rows.append(f.result())
            new_ids.append(str(t["id"]))
        except Exception as e:
            log.warning("決算短信 %s: %s", t.get("id"), e)
            fails += 1
            stop = stop or api_stop_reason(e)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for t in targets:
            if time.time() - t0 > max_minutes * 60:
                break
            if stop:  # キーが無効・残高切れなら、残りの PDF は取りに行かない
                log.error("決算短信の定性情報を中止: %s", stop)
                break
            try:
                resp = requests.get(t["document_url"].split("rd.php?")[-1], headers=UA, timeout=60)
                ok = resp.status_code == 200 and resp.content[:4] == b"%PDF"
            except Exception as e:
                log.warning("PDF %s: %s", t.get("id"), e)
                ok = False
            finally:
                time.sleep(0.7)
            if not ok:
                fails += 1
                continue
            futs.append((ex.submit(work, t, resp.content), t))
            while len(futs) > workers * 2:
                collect(*futs.pop(0))
            if len(rows) >= 50:
                flush()
        for f, t in futs:
            collect(f, t)
    flush()
    total = len(set(pd.read_parquet(done_p)["id"])) if done_p.exists() else 0
    info = dict(targets=len(targets), failed=fails, total_docs=total, minutes=round((time.time() - t0) / 60, 1), model=model,
                **({"stopped": stop} if stop else {}))
    log.info("決算短信の定性情報: 対象 %d件（失敗 %d件）／累計 %d件", len(targets), fails, total)
    return info


if __name__ == "__main__":  # python -m kabuscout.tdnet_quals [分]
    import sys
    from .config import load_config
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    info = update(load_config(), days=30, max_minutes=float(sys.argv[1]) if len(sys.argv) > 1 else 120)
    print(pd.Timestamp.now(), info)
    notify_stop("決算短信の読み取り", info)
