"""財務データ: 金融庁 EDINET API v2（無料・要APIキー、環境変数 EDINET_API_KEY）。

有価証券報告書の「主要な経営指標等の推移」には直近5期分の売上高・純利益・1株純資産(BPS)・
1株利益(EPS)・ROE・自己資本比率が載っている。1社あたり数年おきに報告書を取れば長い履歴が揃うので、
全社10年分を1件ずつ取るより大幅に少ない取得回数で済む。

取得は1件ずつ間隔を空けて行い、毎週の研究ジョブの時間で少しずつ貯める（初回は数週間かけて揃う）。
先読みを防ぐため、各期の数字は「決算期末の95日後」から使えるものとして扱う（有報の提出は通常3ヶ月以内）。

EDINET は海外のサーバーからのアクセスを入口で拒否する（GitHub Actions からは HTTP 403 の HTMLページが返る）。
そのため取得は日本の PC（.env に EDINET_API_KEY）で行い、取り出した主要な経営指標だけを edinet_data/metrics.parquet
としてリポジトリに置く（公共データ利用規約に基づく二次利用。出典: EDINET（金融庁））。クラウドはそれを読み込む。
"""
from __future__ import annotations

import io
import json
import logging
import os
import re
import time
import zipfile

import numpy as np
import pandas as pd
import requests

log = logging.getLogger(__name__)
API = "https://api.edinet-fsa.go.jp/api/v2"
FIN_FEATURES = ["pbr", "per", "roe", "equity_ratio", "sales_growth", "profit_growth"]
YEARS_BACK = {"CurrentYear": 0, "Prior1Year": 1, "Prior2Year": 2, "Prior3Year": 3, "Prior4Year": 4}
# 要素IDの末尾（日本基準・IFRS・米国基準の表記ゆれを吸収）
METRICS = {
    "sales": [r"NetSalesSummaryOfBusinessResults$", r"RevenueIFRSSummaryOfBusinessResults$", r"OperatingRevenue1SummaryOfBusinessResults$",
              r"RevenuesUSGAAPSummaryOfBusinessResults$", r"OrdinaryIncomeSummaryOfBusinessResults$"],
    "profit": [r"ProfitLossAttributableToOwnersOfParentSummaryOfBusinessResults$", r"ProfitLossAttributableToOwnersOfParentIFRSSummaryOfBusinessResults$",
               r"NetIncomeLossSummaryOfBusinessResults$", r"NetIncomeLossAttributableToOwnersOfParentUSGAAPSummaryOfBusinessResults$"],
    "bps": [r"NetAssetsPerShareSummaryOfBusinessResults$", r"EquityAttributableToOwnersOfParentPerShareIFRSSummaryOfBusinessResults$",
            r"EquityAttributableToOwnersOfParentPerShareUSGAAPSummaryOfBusinessResults$"],
    "eps": [r"BasicEarningsLossPerShareSummaryOfBusinessResults$", r"BasicEarningsLossPerShareIFRSSummaryOfBusinessResults$",
            r"BasicEarningsLossPerShareUSGAAPSummaryOfBusinessResults$"],
    "roe": [r"RateOfReturnOnEquitySummaryOfBusinessResults$", r"RatioOfProfitToEquityAttributableToOwnersOfParentIFRSSummaryOfBusinessResults$",
            r"RateOfReturnOnEquityUSGAAPSummaryOfBusinessResults$"],
    "equity_ratio": [r"EquityToAssetRatioSummaryOfBusinessResults$", r"RatioOfOwnersEquityToGrossAssetsIFRSSummaryOfBusinessResults$",
                     r"RatioOfEquityAttributableToOwnersOfParentToTotalAssetsIFRSSummaryOfBusinessResults$",
                     r"EquityToAssetRatioUSGAAPSummaryOfBusinessResults$"],
}


def _repo_metrics():
    from .config import ROOT
    return ROOT / "edinet_data" / "metrics.parquet"


def _key() -> str | None:
    return os.environ.get("EDINET_API_KEY") or None


def _paths(cfg):
    d = cfg["data"]["db_path"].parent / "edinet"
    d.mkdir(parents=True, exist_ok=True)
    return d / "docs.parquet", d / "metrics.parquet", d / "state.json"


def _get(url: str, params: dict, tries: int = 3):
    for a in range(tries):
        try:
            r = requests.get(url, params={**params, "Subscription-Key": _key()}, timeout=60,
                             headers={"User-Agent": "Mozilla/5.0 (KabuScout; personal research)"})
            if r.status_code == 200:
                return r
            log.warning("EDINET %s: HTTP %s %s", url.rsplit("/", 1)[-1], r.status_code, r.text[:200].replace("\n", " "))
        except Exception as e:
            log.warning("EDINET 通信エラー: %s", str(e).replace(_key() or "\0", "***"))  # URL に含まれるキーを伏せる
        time.sleep(5 * (a + 1))
    return None


def list_docs(date: pd.Timestamp) -> pd.DataFrame:
    """その日に提出された書類の一覧（有価証券報告書のみ）。"""
    r = _get(f"{API}/documents.json", {"date": date.strftime("%Y-%m-%d"), "type": 2})
    if r is None:
        raise RuntimeError("書類一覧の取得に失敗")
    res = r.json().get("results") or []
    df = pd.DataFrame(res)
    if df.empty:
        return df
    df = df[(df["docTypeCode"] == "120") & df["secCode"].notna() & (df.get("csvFlag", "1") == "1")]
    return df.assign(code=df["secCode"].str[:4], list_date=date)[["docID", "code", "filerName", "periodEnd", "submitDateTime", "list_date"]]


def parse_doc(content: bytes) -> pd.DataFrame:
    """有報のCSV（XBRLを表形式にしたもの）から、主要な経営指標の5期分を取り出す。"""
    z = zipfile.ZipFile(io.BytesIO(content))
    name = next((n for n in z.namelist() if re.search(r"jpcrp.*\.csv$", n)), None)
    if not name:
        return pd.DataFrame()
    df = pd.read_csv(io.BytesIO(z.read(name)), sep="\t", encoding="utf-16", dtype=str)
    df.columns = ["element", "label", "context", "rel_year", "consol", "period", "unit_id", "unit", "value"][: len(df.columns)]
    rows = {}
    for metric, pats in METRICS.items():
        for pat in pats:
            hit = df[df["element"].str.contains(pat, regex=True, na=False)]
            if hit.empty:
                continue
            for _, r in hit.iterrows():
                m = re.match(r"(CurrentYear|Prior\dYear)(Duration|Instant)(_NonConsolidatedMember)?$", str(r["context"]))
                if not m:
                    continue
                if m.group(1) not in YEARS_BACK:  # 6期以上前の値が載っている書類もある（使わない）
                    continue
                k = YEARS_BACK[m.group(1)]
                nonc = bool(m.group(3))
                try:
                    v = float(str(r["value"]).replace(",", ""))
                except ValueError:
                    continue
                cur = rows.setdefault(k, {})
                # 連結を優先、無ければ個別
                if metric not in cur or (cur.get(f"_{metric}_nonc") and not nonc):
                    cur[metric], cur[f"_{metric}_nonc"] = v, nonc
            break
    out = pd.DataFrame([{"years_back": k, **{m: v for m, v in d.items() if not m.startswith("_")}} for k, d in rows.items()])
    return out


def update(cfg: dict, max_minutes: float = 30, start: str = "2014-06-01") -> dict:
    """書類一覧を日付順にたどり、各社の有報を「同じ会社は4年に1回」取得する（5期分の推移で間を埋める）。"""
    if not _key():
        log.info("EDINET_API_KEY が未設定のため財務データの取得をスキップ")
        return dict(skipped=True)
    docs_p, met_p, state_p = _paths(cfg)
    state = json.loads(state_p.read_text()) if state_p.exists() else {"listed_until": start}
    docs = pd.read_parquet(docs_p) if docs_p.exists() else pd.DataFrame()
    mets = pd.read_parquet(met_p) if met_p.exists() else pd.DataFrame()
    if mets.empty and _repo_metrics().exists():
        mets = pd.read_parquet(_repo_metrics())
    t0 = time.time()
    budget = max_minutes * 60
    # 1) 書類一覧（全期間の半分の時間まで）
    d = pd.Timestamp(state["listed_until"])
    today = pd.Timestamp.today().normalize()
    new_lists = []
    while d <= today and time.time() - t0 < budget * 0.4:
        if d.weekday() >= 5:  # 土日は提出が無い
            d += pd.Timedelta(days=1)
            continue
        try:
            new_lists.append(list_docs(d))
            state["listed_until"] = (d + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        except Exception as e:
            log.warning("EDINET 書類一覧 %s: %s", d.date(), e)
            break
        d += pd.Timedelta(days=1)
        time.sleep(1.0)
    if new_lists:
        docs = pd.concat([docs] + [x for x in new_lists if len(x)], ignore_index=True).drop_duplicates("docID")
    # 2) 取得する書類を選ぶ: 会社ごとに最新の有報と、そこから4年ずつさかのぼった有報
    fetched = set(mets["docID"]) if len(mets) else set()
    if docs.empty:
        state_p.write_text(json.dumps(state))
        return dict(docs=0, fetched=0, listed_until=state["listed_until"])
    want = []
    for code, g in docs.sort_values("periodEnd", ascending=False).groupby("code"):
        last_pe = None
        for _, r in g.iterrows():
            pe = pd.Timestamp(r["periodEnd"])
            if last_pe is None or (last_pe - pe).days >= 4 * 365 - 30:
                want.append(r)
                last_pe = pe
    want = [r for r in want if r["docID"] not in fetched]
    want.sort(key=lambda r: r["periodEnd"], reverse=True)  # 新しい年から
    got = []
    for r in want:
        if time.time() - t0 > budget:
            break
        resp = _get(f"{API}/documents/{r['docID']}", {"type": 5})
        if resp is not None and resp.headers.get("Content-Type", "").startswith("application/octet-stream"):
            try:
                m = parse_doc(resp.content)
                if len(m):
                    pe = pd.Timestamp(r["periodEnd"])
                    m["period_end"] = [pe - pd.DateOffset(years=int(k)) for k in m["years_back"]]
                    got.append(m.assign(code=r["code"], docID=r["docID"], filed=r["submitDateTime"]))
                else:
                    got.append(pd.DataFrame([{"code": r["code"], "docID": r["docID"]}]))  # 取れない書類も記録して再取得しない
            except Exception as e:
                log.warning("EDINET 解析失敗 %s: %s", r["docID"], e)
        time.sleep(1.5)
    if got:
        mets = pd.concat([mets] + got, ignore_index=True)
        mets.to_parquet(met_p)
        if not os.environ.get("GITHUB_ACTIONS"):  # 日本の PC で取得した分をリポジトリ用にも保存（クラウドが読む）
            _repo_metrics().parent.mkdir(exist_ok=True)
            mets.to_parquet(_repo_metrics())
    docs.to_parquet(docs_p)
    state_p.write_text(json.dumps(state))
    have = mets.dropna(subset=["period_end"]) if len(mets) and "period_end" in mets else pd.DataFrame()
    info = dict(listed_until=state["listed_until"], docs=int(len(docs)), fetched=int(len(set(mets["docID"])) if len(mets) else 0),
                todo=int(len(want) - len(got)), companies=int(have["code"].nunique()) if len(have) else 0)
    log.info("EDINET: 書類一覧 %s まで / 有報 %d件取得済み（残り %d件）/ %d社分の財務", info["listed_until"], info["fetched"],
             info["todo"], info["companies"])
    return info


def load(cfg: dict) -> pd.DataFrame | None:
    _, met_p, _ = _paths(cfg)
    parts = [pd.read_parquet(p) for p in (met_p, _repo_metrics()) if p.exists()]
    if not parts:
        return None
    m = pd.concat(parts, ignore_index=True)
    if "period_end" not in m:
        return None
    m = m.dropna(subset=["period_end"])
    # 同じ会社・同じ期は、新しい報告書の値（訂正後）を優先
    return m.sort_values("filed").drop_duplicates(["code", "period_end"], keep="last")


def wide_features(fin: pd.DataFrame | None, close: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """決算期末の95日後から使える財務の値を、日付×銘柄の特徴量にする。"""
    idx, cols = close.index, close.columns
    out = {f: pd.DataFrame(np.nan, index=idx, columns=cols, dtype="float32") for f in FIN_FEATURES}
    if fin is None or fin.empty:
        return out
    f = fin[fin["code"].isin(set(cols))].copy()
    if f.empty:
        return out
    f["known"] = pd.to_datetime(f["period_end"]) + pd.Timedelta(days=95)
    f = f.sort_values(["code", "period_end"])
    f["sales_growth"] = f.groupby("code")["sales"].pct_change(fill_method=None)
    f["profit_growth"] = f.groupby("code")["profit"].pct_change(fill_method=None)
    vals = {}
    for k in ("bps", "eps", "roe", "equity_ratio", "sales_growth", "profit_growth"):
        w = pd.DataFrame(np.nan, index=idx, columns=cols)
        for code, g in f.dropna(subset=[k] if k in f else []).groupby("code"):
            pos = idx.searchsorted(g["known"].to_numpy())
            ok = pos < len(idx)
            w.iloc[pos[ok], cols.get_loc(code)] = g[k].to_numpy()[ok]
        vals[k] = w.ffill(limit=400)
    out["pbr"] = close / vals["bps"].where(vals["bps"] > 0)
    out["per"] = close / vals["eps"].where(vals["eps"] > 0)
    out["roe"], out["equity_ratio"] = vals["roe"], vals["equity_ratio"]
    out["sales_growth"] = vals["sales_growth"].clip(-1, 5)
    out["profit_growth"] = vals["profit_growth"].clip(-5, 10)
    return {k: v.astype("float32") for k, v in out.items()}


# ---------------------------------------------------------------- 大量保有報告書（5%ルール）
HOLDING_TYPES = {"350", "360"}  # 大量保有報告書・変更報告書 / その訂正


def _holding_paths(cfg):
    d = _paths(cfg)[0].parent
    return d / "holdings.parquet", d / "edinet_codes.parquet", d / "holdings_state.json"


def update_holdings(cfg: dict, max_minutes: float = 60, start: str = "2014-06-01") -> dict:
    """書類一覧を日付順にたどり、大量保有報告書（対象会社 = issuerEdinetCode）と、EDINETコード→証券コードの対応を貯める。
    書類の中身（保有割合など）はまだ取らない。途中で止まっても100日ごとに保存する。"""
    if not _key():
        return dict(skipped=True)
    hp, cp, sp = _holding_paths(cfg)
    state = json.loads(sp.read_text()) if sp.exists() else {"listed_until": start}
    hold = [pd.read_parquet(hp)] if hp.exists() else []
    codes = [pd.read_parquet(cp)] if cp.exists() else []
    d, today, t0 = pd.Timestamp(state["listed_until"]), pd.Timestamp.today().normalize(), time.time()
    cols = ["docID", "docTypeCode", "docDescription", "filerName", "edinetCode", "issuerEdinetCode", "submitDateTime",
            "parentDocID", "withdrawalStatus", "csvFlag"]

    def save():
        if hold:
            pd.concat(hold, ignore_index=True).drop_duplicates("docID").to_parquet(hp)
        if codes:
            pd.concat(codes, ignore_index=True).drop_duplicates("edinetCode", keep="last").to_parquet(cp)
        sp.write_text(json.dumps(state))

    n = 0
    while d <= today and time.time() - t0 < max_minutes * 60:
        if d.weekday() >= 5:
            d += pd.Timedelta(days=1)
            continue
        r = _get(f"{API}/documents.json", {"date": d.strftime("%Y-%m-%d"), "type": 2})
        if r is None:
            break
        df = pd.DataFrame(r.json().get("results") or [])
        if len(df):
            h = df[df["docTypeCode"].isin(HOLDING_TYPES)]
            if len(h):
                hold.append(h[[c for c in cols if c in h]].assign(list_date=d))
            c = df[df["secCode"].notna() & df["edinetCode"].notna()][["edinetCode", "secCode"]]
            if len(c):
                codes.append(c.assign(code=c["secCode"].str[:4]).drop_duplicates("edinetCode"))
        d += pd.Timedelta(days=1)
        state["listed_until"] = d.strftime("%Y-%m-%d")
        n += 1
        if n % 100 == 0:
            save()
        time.sleep(1.0)
    save()
    h = pd.concat(hold, ignore_index=True) if hold else pd.DataFrame()
    log.info("大量保有報告書: 書類一覧 %s まで / %d件", state["listed_until"], len(h))
    return dict(listed_until=state["listed_until"], reports=int(len(h)))
