"""業績予想の修正・予想と実績の差異の「理由」を、Claude に PDF を読ませて分類し、毎日貯める。

XBRL（tdnet_xbrl.py）で取れるのは数字だけで、「なぜ修正したか」は入っていない。
本業の需要が強くて上方修正（続きやすい）なのか、土地の売却益など一度きり（続かない）なのかを分けたい。
TDnet は書類を約1ヶ月で消すので過去分は作れない → 直近30日分の未処理を毎日集めて貯める（効くかの検証は数年後）。

- Claude には「書いてあることの分類」だけをさせ、株価の予想はさせない（モデルが過去を覚えている問題を避ける）
- 元の PDF は保存せず、分類結果だけを data/tdnet_reasons/ に置く（公開しない）
- API キーは .env の ANTHROPIC_API_KEY（クラウドには置かない。Mac で動かす: scripts/install_reasons_launchd.sh）
- モデルは環境変数 KABUSCOUT_REASON_MODEL で変えられる（既定 claude-opus-5-5）
"""
from __future__ import annotations

import base64
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

from .tdnet import API, UA

log = logging.getLogger(__name__)
TITLE = re.compile(r"業績予想|業績修正|予想.{0,12}差異|実績.{0,12}差異|差異.{0,6}実績")
SKIP = re.compile(r"訂正|決算短信")
MAX_PDF = 8_000_000
DEFAULT_MODEL = "claude-opus-5-5"

DRIVERS = ["demand", "pricing", "cost", "fx", "one_off_gain", "one_off_loss", "tax", "non_operating",
           "m_and_a", "timing", "incident", "other", "none"]
DRIVER_JA = {"demand": "本業の需要・受注・販売数量", "pricing": "販売価格（値上げ・値下げ・市況）", "cost": "原価・費用（原材料・人件費・経費削減）",
             "fx": "為替", "one_off_gain": "一度きりの利益（資産売却益・補助金など特別利益）", "one_off_loss": "一度きりの損失（減損・リストラ・訴訟など特別損失）",
             "tax": "税金（繰延税金資産など）", "non_operating": "営業外（持分法・金融収益・投資損益）", "m_and_a": "買収・売却・連結範囲の変更",
             "timing": "案件・売上の期ずれ", "incident": "災害・事故・不祥事・システム障害", "other": "その他", "none": "なし"}
NUM = {"type": ["number", "null"]}
SCHEMA = {
    "type": "object",
    "properties": {
        "doc_kind": {"type": "string", "enum": ["forecast_revision", "actual_vs_forecast", "both", "other"]},
        "fy_direction": {"type": "string", "enum": ["up", "down", "mixed", "unchanged", "no_fy_revision"]},
        "sales_change_pct": NUM, "op_change_pct": NUM, "net_change_pct": NUM,
        "main_driver": {"type": "string", "enum": DRIVERS},
        "second_driver": {"type": "string", "enum": DRIVERS},
        "persistence": {"type": "string", "enum": ["recurring", "one_off", "mixed", "unclear"]},
        "core_business": {"type": "boolean"},
        "conservative_guidance": {"type": "boolean"},
        "dividend_change": {"type": "string", "enum": ["up", "down", "none"]},
        "summary_ja": {"type": "string"},
    },
    "required": ["doc_kind", "fy_direction", "sales_change_pct", "op_change_pct", "net_change_pct", "main_driver",
                 "second_driver", "persistence", "core_business", "conservative_guidance", "dividend_change", "summary_ja"],
    "additionalProperties": False,
}
PROMPT = """添付は日本の上場企業の適時開示（業績予想の修正、または業績予想と実績の差異のお知らせ）です。書いてある内容だけから次を判定してください。推測で株価や将来を予想しないこと。

- doc_kind: 通期などの予想を修正した=forecast_revision、期間の予想と実績の差を説明しただけ=actual_vs_forecast、両方=both、それ以外=other
- fy_direction: 今回の通期予想の修正の向き（営業利益を最優先、無ければ経常利益→純利益で判断）。上=up、下=down、項目で上下が分かれる=mixed、据え置きと明記=unchanged、通期の修正が無い=no_fy_revision
- sales_change_pct / op_change_pct / net_change_pct: 通期予想の 売上高 / 営業利益 / 親会社株主に帰属する当期純利益 の「前回予想→今回予想」の増減率（%、例 +12.5）。表に増減率があればそれを使う。通期の修正が無い・前回が未定・数値が無いときは null
- main_driver / second_driver: 修正（無ければ差異）の主な理由と2番目の理由。次から選ぶ（2番目が無ければ none）:
  demand=本業の需要・受注・販売数量 / pricing=販売価格・市況 / cost=原価・費用 / fx=為替 / one_off_gain=資産売却益など一度きりの利益 / one_off_loss=減損・リストラなど一度きりの損失 / tax=税金 / non_operating=持分法・金融収益など営業外 / m_and_a=買収・連結範囲 / timing=案件・売上の期ずれ / incident=災害・事故・不祥事 / other=その他
- persistence: その理由が来期以降も続きそうな性質か。本業の需要・価格・構造的なコスト=recurring、特別損益・期ずれ・一時要因=one_off、両方=mixed、書かれていない=unclear
- core_business: 営業利益（本業）が理由の中心なら true、純利益だけの話（特別損益・税金など）なら false
- conservative_guidance: 上期の上振れ分を通期に反映しない・下期を保守的に見ている、と読み取れる記述があれば true
- dividend_change: 同じ書類で配当予想を増やした=up、減らした=down、それ以外=none
- summary_ja: 理由の要約（日本語、60字以内）"""


def _dir(cfg):
    d = cfg["data"]["db_path"].parent / "tdnet_reasons"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load(cfg: dict) -> pd.DataFrame | None:
    parts = sorted(_dir(cfg).glob("reasons_*.parquet"))
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True).drop_duplicates("id", keep="last") if parts else None


def is_target(title: str) -> bool:
    return bool(TITLE.search(title or "")) and not SKIP.search(title or "")


def _client():
    import anthropic
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(".env に ANTHROPIC_API_KEY がありません")
    return anthropic.Anthropic(max_retries=4, timeout=180)


def classify_pdf(client, pdf: bytes, model: str) -> tuple[dict, dict]:
    """PDF を Claude に読ませて、SCHEMA の形の分類と使ったトークン数を返す。"""
    content = [{"type": "document", "source": {"type": "base64", "media_type": "application/pdf",
                                               "data": base64.standard_b64encode(pdf).decode()}},
               {"type": "text", "text": PROMPT}]
    r = client.messages.create(model=model, max_tokens=4000, messages=[{"role": "user", "content": content}],
                               output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}})
    if r.stop_reason in ("refusal", "max_tokens"):
        raise RuntimeError(f"stop_reason={r.stop_reason}")
    text = next(b.text for b in r.content if b.type == "text")
    return json.loads(text), dict(in_tok=r.usage.input_tokens, out_tok=r.usage.output_tokens)


def _list_targets(days: int, done: set) -> list[dict]:
    today = pd.Timestamp.today().normalize()
    out = []
    for k in range(days):
        day = today - pd.Timedelta(days=k)
        try:
            r = requests.get(API.format(d=day.strftime("%Y%m%d")), headers=UA, timeout=60)
            items = [it.get("Tdnet", it) for it in r.json().get("items", [])]
        except Exception as e:
            log.warning("適時開示の一覧 %s: %s", day.date(), e)
            continue
        time.sleep(1.0)
        out += [t for t in items if t.get("document_url") and is_target(t.get("title")) and str(t.get("id")) not in done]
    return out


def update(cfg: dict, days: int = 30, max_minutes: float = 60, limit: int | None = None, workers: int = 4) -> dict:
    """直近 days 日の業績予想の修正・差異のお知らせのうち、未処理のものを分類して貯める。"""
    d = _dir(cfg)
    done_p = d / "done.parquet"
    done = set(pd.read_parquet(done_p)["id"]) if done_p.exists() else set()
    model = os.environ.get("KABUSCOUT_REASON_MODEL", DEFAULT_MODEL)
    client = _client()
    targets = _list_targets(days, done)[: limit or None]
    log.info("理由の分類: 未処理 %d件（モデル %s）", len(targets), model)
    t0, rows, new_ids, fails, futs = time.time(), [], [], 0, []

    def flush():
        nonlocal rows, new_ids
        if rows:
            pd.DataFrame(rows).to_parquet(d / f"reasons_{pd.Timestamp.now():%Y%m%d_%H%M%S_%f}.parquet")
        if new_ids:
            done.update(new_ids)
            pd.DataFrame({"id": sorted(done)}).to_parquet(done_p)
        rows, new_ids = [], []

    def work(t, pdf):
        res, use = classify_pdf(client, pdf, model)
        return dict(id=str(t["id"]), code=str(t.get("company_code") or "")[:4], pubdate=pd.Timestamp(t["pubdate"]),
                    title=t["title"], model=model, **use, **res)

    def collect(f, t):
        nonlocal fails
        try:
            rows.append(f.result())
            new_ids.append(str(t["id"]))
        except Exception as e:
            log.warning("理由の分類 %s %s: %s", t.get("id"), (t.get("title") or "")[:30], e)
            fails += 1

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for t in targets:
            if time.time() - t0 > max_minutes * 60:
                break
            url = t["document_url"].split("rd.php?")[-1]
            try:
                resp = requests.get(url, headers=UA, timeout=60)
                ok = resp.status_code == 200 and resp.content[:4] == b"%PDF" and len(resp.content) <= MAX_PDF
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
    info = dict(targets=len(targets), failed=fails, total_docs=total, minutes=round((time.time() - t0) / 60, 1), model=model)
    log.info("理由の分類: 対象 %d件（失敗 %d件）／累計 %d件", len(targets), fails, total)
    return info


if __name__ == "__main__":  # launchd から: python -m kabuscout.tdnet_reasons
    import sys
    from .config import load_config
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    print(pd.Timestamp.now(), update(load_config(), days=30, max_minutes=float(sys.argv[1]) if len(sys.argv) > 1 else 90))
