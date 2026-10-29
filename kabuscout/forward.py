"""その日の全候補の予測を残し、結果が出たら「順位の質」を答え合わせする（前向きの検証）。

ペーパートレードは1日2件ほどで、乱数の種だけでも年±8ポイント動く（ジャーナル 24）ので、損益だけでは
「AIが効いているか」の判定に何年もかかる（1取引の損益の標準偏差 約7% → t=2 まで約3年）。
そこで、毎日の全候補（100〜200件）について、AIの点と、その後の実際の売買ルールでの結果を比べる:
- 早期警報（主な指標）: 共通の短期ルール（10営業日）での結果との順位相関（スピアマン）。全候補が10営業日で決着するので早く判定できる。
  過去の検証（2016〜2026年）では週平均 0.16・プラスの週 88%で、結果の出揃った週が約5週あれば「0より上」と言える（ジャーナル 63）。
  比較する過去の分布も同じラベルで作る（backtest.json の forward_ref.common）
- 実際の売買ルール（最長60営業日）での順位相関・上位10%の上乗せは参考と影の比較に使う（全候補が出揃うまで約3ヶ月）
- 集計するのは、その日の全候補の結果が出揃った日だけ（matured）。決着したものから集めると最近の日は早く決着した取引（多くは損切）に偏る

注意: 同じ日の候補は一緒に動き、保有期間も重なるので件数ほど独立ではない。週ごとの平均にしてから比べ、
t 値は週どうしの相関を考えて計算する（Newey-West、4週）。
- 全候補の記録は data/scored/ に「データ日付_実行時刻」の名前で追記するだけ（上書きしない）。同じ日を再実行しても最初の予測が残り、
  答え合わせには各日の最初の実行を使う
- 保存した特徴量は、後で計算し直した値と比べて「本番と検証の食い違い」の検出にも使う（drift）
"""
from __future__ import annotations

import json
import logging
from datetime import datetime

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
MIN_WEEKS = 4      # 判定を始める週数（過去の検証で、順位相関が0より上と言えるまでの目安）
NW_LAGS = 4        # 週どうしの相関（保有期間の重なり）を考える長さ
TOP_Q = 0.9        # 上位10%
MATURE_DAYS = 10   # 特徴量の食い違いを調べるのは、この営業日数より前の日（直近は後から届くデータで変わりうるので結果と一緒に見る）


WIDE_MIN_TURNOVER = 5e7  # 対象を広げたときの売買代金の下限（ジャーナル 37: 5,000万円以上は年率 +15.9% vs +10.7%、最大下落は深い）


def log_dir(cfg: dict, sub: str = "scored"):
    d = cfg["data"]["db_path"].parent / sub
    d.mkdir(parents=True, exist_ok=True)
    return d


def save(cfg: dict, scored: pd.DataFrame, data_date: str, shadows: dict | None = None, sub: str = "scored") -> str:
    """その日の全候補（AIの点・モデルの版・特徴量・影の点）を追記する。同じ名前のファイルがあっても上書きしない。
    sub="scored_wide": 対象を広げたときの候補（record_wide）。"""
    run_at = datetime.now().astimezone()
    df = scored.copy()
    df["live_rank"] = np.arange(1, len(df) + 1)  # 本番の並び順（final_score の高い順）
    df["run_at"] = run_at.isoformat(timespec="seconds")
    for key, (sc, ver) in (shadows or {}).items():
        df[f"score_{key}"] = sc
        df[f"version_{key}"] = ver
    for c in df.columns:  # 文字列以外の object 列（None 混じりなど）を parquet に書けるように
        if df[c].dtype == object and not df[c].map(lambda v: v is None or isinstance(v, str)).all():
            df[c] = df[c].astype(str)
    base = log_dir(cfg, sub) / f"{data_date}_{run_at.strftime('%Y%m%dT%H%M%S')}"
    path, k = base.with_suffix(".parquet"), 1
    while path.exists():
        path, k = base.parent / f"{base.name}_{k}.parquet", k + 1
    df.to_parquet(path, index=False)
    log.info("全候補の予測を記録: %s/%s（%d件）", sub, path.name, len(df))
    return path.name


def record_wide(cfg: dict, store, uni: pd.DataFrame, today: pd.DataFrame, bundle: dict | None, weights: dict,
                last: pd.Timestamp) -> str:
    """対象を広げたとき（売買代金の下限 1億円 → 5,000万円）の候補にも本番のAIで点を付け、記録だけする（学習・売買・通知には使わない）。
    AIの点はその日の候補の中での順位なので、5,000万〜1億円の候補（band=1）をその日の本番の候補（band=0）と一緒に採点し直す
    ＝広げた場合の順位。結果は後で株価から計算できるので、「広げたら上位に入る候補が実際にどうだったか」を、売買せずに前向きに
    確かめられる（ジャーナル 37・64）。data/scored_wide/ に追記するだけ（上書きしない）。"""
    import copy
    from . import alerts, learner, pool
    from .pipeline import dataset
    c = copy.deepcopy(cfg)
    c["universe"]["min_turnover_jpy"] = WIDE_MIN_TURNOVER
    _, rows = pool.build_window(store, c, uni, (last - pd.Timedelta(days=pool.WARMUP_DAYS)).strftime("%Y-%m-%d"),
                                last.strftime("%Y-%m-%d"), keep_from=last.strftime("%Y-%m-%d"))
    band = dataset(cfg, rows[(rows["date"] == last) & (rows["turnover"] < cfg["universe"]["min_turnover_jpy"])])
    danger = alerts.load(cfg, pd.Timestamp.today())
    if danger:
        band = band[~band["code"].isin(danger)]
    both = pd.concat([today.assign(band=0), band.assign(band=1)], ignore_index=True)
    return save(cfg, learner.score(both, bundle, weights), last.strftime("%Y-%m-%d"), sub="scored_wide")


def load(cfg: dict, first_only: bool = True, sub: str = "scored") -> pd.DataFrame | None:
    """記録した全候補。first_only: 同じデータ日付を何度か実行していたら最初の実行だけ（最初の予測で答え合わせする）。"""
    files = sorted(log_dir(cfg, sub).glob("*.parquet"))
    if not files:
        return None
    df = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    if first_only:
        first = df.groupby("date")["run_at"].transform("min")
        df = df[df["run_at"] == first]
    return df.reset_index(drop=True)


def with_outcomes(cfg: dict, logged: pd.DataFrame, ds: pd.DataFrame) -> pd.DataFrame:
    """記録した候補に結果を付ける（決着していないものは NaN）。
    - ret_common: 共通の短期ルール（labeling.horizon_days、10営業日）での結果。全候補が10営業日で決着するので早期警報に使う
    - ret: 実際に使う売買ルール（戦略ごと、最長 plan_horizon 営業日）での結果。過去の検証の ret と同じ定義"""
    from . import learner
    cols = ["date", "code", "strategies", "atr", "ret", "label", "exit_days", "complete"]
    d = ds[cols].merge(logged[["date", "code"]].drop_duplicates(), on=["date", "code"])
    if d.empty:
        return logged.iloc[:0].assign(ret=pd.Series(dtype=float), ret_common=pd.Series(dtype=float))
    p = learner.with_plan_outcomes(d, cfg)
    d = d.assign(ret_common=d["ret"].where(d["complete"]), ret=p["ret_plan"].where(p["complete_plan"]))[["date", "code", "ret_common", "ret"]]
    return logged.drop(columns=[c for c in ("ret", "ret_common") if c in logged]).merge(d, on=["date", "code"])


def matured(df: pd.DataFrame, ds: pd.DataFrame, h: int) -> pd.DataFrame:
    """その日の全候補の結果が出揃った日だけ（最長 h 営業日の保有が終わった日）。
    決着した候補から集めると、最近の日は早く決着した取引（多くは損切）ばかりになり、過去の検証（全候補の結果が出揃ったもの）と条件が揃わない。"""
    dates = np.sort(ds["date"].unique())
    if len(dates) <= h:
        return df.iloc[:0]
    return df[df["date"] <= dates[-h - 1]]


def daily_metrics(df: pd.DataFrame, col: str, ret: str = "ret") -> pd.DataFrame:
    """日ごとの順位相関と上位10%の上乗せ（候補が20件未満の日は除く）。"""
    rows = []
    for d, g in df.groupby("date"):
        g = g[[col, ret]].dropna()
        if len(g) < 20:
            continue
        r = g[col].rank(pct=True)
        rows.append((d, float(g[col].rank().corr(g[ret].rank())), float(g.loc[r > TOP_Q, ret].mean() - g[ret].mean())))
    return pd.DataFrame(rows, columns=["date", "ic", "uplift"]).set_index("date")


def weekly(m: pd.DataFrame) -> pd.DataFrame:
    return m.groupby(m.index.to_period("W")).mean()


def nw_t(x) -> float | None:
    """週ごとの値の平均が0と違うかの t 値（週どうしの相関を考える Newey-West）。"""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return None
    e = x - x.mean()
    v = e @ e / n
    for lag in range(1, min(NW_LAGS, n - 1) + 1):
        v += 2 * (1 - lag / (NW_LAGS + 1)) * (e[lag:] @ e[:-lag]) / n
    return float(x.mean() / np.sqrt(v / n)) if v > 0 else None


def reference(preds: pd.DataFrame, ds: pd.DataFrame | None = None) -> dict:
    """過去の検証（ウォークフォワード）の週ごとの順位相関・上乗せ。backtest.json に保存して、前向きの結果と比べる。
    plan: 実際の売買ルールでの結果（preds の ret）／common: 共通の短期ルール（10営業日）での結果（ds の ret。早期警報と同じラベル）"""
    fmt = lambda m: dict(weeks=[str(p) for p in m.index], ic=[round(float(v), 4) for v in m["ic"]],
                         uplift=[round(float(v), 5) for v in m["uplift"]])
    pr = preds[preds["complete"]].assign(score=preds["prob"])
    out = dict(plan=fmt(weekly(daily_metrics(pr, "score"))))
    if ds is not None:
        c = pr[["date", "code", "score"]].merge(ds.loc[ds["complete"], ["date", "code", "ret"]].rename(columns={"ret": "ret_common"}),
                                                on=["date", "code"])
        out["common"] = fmt(weekly(daily_metrics(c, "score", "ret_common")))
    return out


def _rolling_means(x: np.ndarray, k: int) -> np.ndarray:
    c = np.concatenate([[0.0], np.cumsum(x)])
    return (c[k:] - c[:-k]) / k


def _vs_reference(live: pd.Series, ref: list[float] | None) -> dict:
    """直近 n 週の平均を、過去の検証の「連続 n 週の平均」の分布と比べる（health.py と同じ考え方）。"""
    n = len(live)
    out = dict(weeks=n, mean=float(live.mean()), t=nw_t(live.to_numpy()), positive=float((live > 0).mean()))
    if ref and len(ref) > n:
        dist = _rolling_means(np.asarray(ref, dtype=float), n)
        out.update(percentile=float((dist < live.mean()).mean()), typical=float(np.median(dist)),
                   low10=float(np.quantile(dist, 0.10)), low025=float(np.quantile(dist, 0.025)))
    return out


def drift(cfg: dict, logged: pd.DataFrame, ds: pd.DataFrame, features: list[str]) -> dict:
    """記録した特徴量（予測した日の値）と、今のシグナル表の同じ日・同じ銘柄の値を比べる。
    違いが多い特徴量は、後から届くデータや計算の変更で本番と検証が食い違っている可能性がある。"""
    dates = sorted(logged["date"].unique())
    last = ds["date"].max()
    cut = sorted(ds.loc[ds["date"] <= last, "date"].unique())
    if len(cut) <= MATURE_DAYS:
        return {}
    old = [d for d in dates if d <= cut[-MATURE_DAYS - 1]]
    if not old:
        return {}
    feats = [f for f in features if f in logged and f in ds]
    a = logged[logged["date"].isin(old)][["date", "code"] + feats]
    b = ds[ds["date"].isin(old)][["date", "code"] + feats]
    m = a.merge(b, on=["date", "code"], suffixes=("_l", "_n"))
    if m.empty:
        return {}
    rows = []
    for f in feats:
        x, y = m[f"{f}_l"].astype(float), m[f"{f}_n"].astype(float)
        diff = (x.isna() != y.isna()) | ((x - y).abs() > 1e-6 * np.maximum(1.0, y.abs()))
        if diff.any():
            rows.append((f, float(diff.mean())))
    rows.sort(key=lambda t: -t[1])
    return dict(n=int(len(m)), days=len(old), changed=[dict(feature=f, share=round(s, 4)) for f, s in rows[:10]])


def evaluate(cfg: dict, ds: pd.DataFrame) -> dict:
    """前向きの答え合わせ。本番のAIの点（ai_score）と影の候補の点を、同じ日の候補で比べる。data/forward.json に保存。"""
    from . import learner, shadow
    logged = load(cfg)
    if logged is None:
        return dict(status="waiting", text="全候補の記録がまだありません", weeks=0)
    res = with_outcomes(cfg, logged, ds)
    p = cfg["data"]["db_path"].parent / "backtest.json"
    ref = (json.loads(p.read_text()).get("forward_ref") if p.exists() else None) or {}
    hc, hp = int(cfg["labeling"]["horizon_days"]), learner.plan_horizon(cfg)
    rc = matured(res, ds, hc).dropna(subset=["ret_common"])  # 早期警報: 共通ルール（全候補が hc 営業日で決着）
    rp = matured(res, ds, hp).dropna(subset=["ret"])         # 実際の売買ルール（最長 hp 営業日）: 参考と影の比較
    out = dict(logged_days=int(logged["date"].nunique()), horizon_common=hc, horizon_plan=hp,
               matured_days=int(rc["date"].nunique()), matured_days_plan=int(rp["date"].nunique()))
    if "ai_score" in res:
        out["ai"] = {}
        if len(rc):
            w = weekly(daily_metrics(rc, "ai_score", "ret_common"))
            c = ref.get("common") or {}
            out["ai"].update(ic=_vs_reference(w["ic"], c.get("ic")), uplift=_vs_reference(w["uplift"], c.get("uplift")),
                             series=[[str(k), round(float(a), 4), round(float(b), 5)] for k, a, b in w[["ic", "uplift"]].itertuples()])
        if len(rp):
            w = weekly(daily_metrics(rp, "ai_score"))
            pl = ref.get("plan") or {}
            out["ai"].update(plan_ic=_vs_reference(w["ic"], pl.get("ic")), plan_uplift=_vs_reference(w["uplift"], pl.get("uplift")))
    ic = (out.get("ai") or {}).get("ic") or {}
    n = ic.get("weeks", 0)
    if n < MIN_WEEKS:
        out.update(status="waiting", text=f"判定には全候補の結果（{hc}営業日ルール）が出揃った週が{MIN_WEEKS}週以上必要（現在{n}週、記録{out['logged_days']}日）")
    elif "percentile" not in ic:
        out.update(status="unknown", text="過去の検証の比較用の値がまだありません（土曜の研究ループで作られます）")
    elif ic["percentile"] < 0.025:
        out.update(status="alert", text="順位の質が想定外に低い（過去の同じ週数の下位2.5%より悪い）。AIが壊れたか相場が変わった可能性")
    elif ic["percentile"] < 0.10:
        out.update(status="caution", text="順位の質がやや低い（過去の下位10%）")
    else:
        out.update(status="ok", text="順位の質は想定どおりの範囲")
    # 影の候補: 同じ日の候補での上位10%の上乗せの差（判定に使う。売買に使うのは順位の上の方なので）と順位相関の差（参考）
    out["shadows"] = []
    for v in shadow.variants(cfg):
        col = f"score_{v['key']}"
        r = dict(key=v["key"], label=v["label"], source=v["source"], start=v["start"], judge=v["judge"], rule=v["rule"])
        if len(rp) and col in rp:
            sub = rp[rp[col].notna()]
            if len(sub):
                a, b = daily_metrics(sub, "ai_score"), daily_metrics(sub, col)
                d = weekly((b - a).dropna())
                r.update(weeks=int(len(d)), ic_diff=float(d["ic"].mean()) if len(d) else None, ic_t=nw_t(d["ic"].to_numpy()),
                         uplift_diff=float(d["uplift"].mean()) if len(d) else None, uplift_t=nw_t(d["uplift"].to_numpy()))
        r["verdict"] = shadow.verdict(v, r)
        out["shadows"].append(r)
    try:
        out["drift"] = drift(cfg, logged, ds, learner.model_features(cfg))
    except Exception as e:  # 診断なので、失敗しても本番は止めない
        log.warning("特徴量の食い違いの確認に失敗: %s", e)
    out["at"] = datetime.now().isoformat(timespec="seconds")
    (cfg["data"]["db_path"].parent / "forward.json").write_text(json.dumps(out, ensure_ascii=False, default=float))
    return out


def text(fw: dict | None) -> str:
    """通知・レポート用の1行。"""
    if not fw:
        return ""
    icon = {"ok": "✅", "caution": "🟡", "alert": "🔴", "waiting": "⏳", "unknown": "・"}.get(fw.get("status"), "")
    s = f"{icon} {fw['text']}"
    ic = (fw.get("ai") or {}).get("ic") or {}
    if ic.get("weeks", 0) >= MIN_WEEKS:
        s += f"（直近{ic['weeks']}週の順位相関〔{fw.get('horizon_common', 10)}営業日ルール〕 {ic['mean']:.3f}"
        if "percentile" in ic:
            s += f"：過去の同じ週数の中で上位{max(1 - ic['percentile'], 0.01):.0%}・ふつうは {ic['typical']:.3f}"
        up = (fw.get("ai") or {}).get("uplift") or {}
        s += f"／上位10%の上乗せ {up.get('mean', float('nan')):+.2%}）"
    sh = [x for x in fw.get("shadows") or [] if x.get("weeks")]
    if sh:
        s += "\n影の候補: " + " ／ ".join(f"{x['label']} 上位10%の上乗せの差 {x['uplift_diff']:+.2%}（t={x['uplift_t'] if x['uplift_t'] is not None else float('nan'):.1f}・{x['weeks']}週・判定 {x['judge']}・{x['verdict']}）" for x in sh)
    return s
