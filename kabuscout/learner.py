"""自己学習エンジン。

仕組み:
1. 過去の全営業日・全銘柄でシグナルを再現し、その後どうなったかを
   トリプルバリア法でラベル付け（翌日寄付きで買い → 利確/損切/期限のどれが先か）。
2. 3つの見方で勾配ブースティングを学習し、その日の候補内での順位を平均してAIの評価にする（新しいデータほど重く扱う）:
   - 勝ちやすさ: 共通の短期ルールで利確に先に届くか
   - 手取りリターン: 実際に使う売買ルール（戦略ごと）で、コストを引いた損益
   - その日の順位: 同じ日の候補の中で、実際の売買ルールでのリターンが何番目か
   1つだけだと特定の年頼みになりやすく、平均すると年率は同程度のまま最大下落が大きく縮んだ（ジャーナル ラウンド8）。
3. 再学習のたびに時系列ホールドアウトで成績を確かめ、
   さらに現役モデル(チャンピオン)と「両者にとって未知の期間」で勝負させ、勝った方だけ採用。
4. 戦略ごとの直近成績から重みを更新（今の相場で効いている網を重く）。
日々データが増えるので、放っておくほど学習データが増え、相場の変化にも追従する。
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from .dividends import DIV_FEATURES
from .earnings import EARN_FEATURES
from .events import EVENT_FEATURES
from .indicators import FEATURES
from .macro import MODEL_MACRO_FEATURES
from .strategies import FLAG_COLS, STRATEGIES

log = logging.getLogger(__name__)
# 地合い系（全銘柄共通の値）は特定の時期の丸暗記を招き、検証で成績が不安定になったため
# 銘柄スコアのモデルからは外す（地合いは通知文と戦略の重みで扱う）
MARKET_FEATURES = ["mkt_dev75", "mkt_ret5", "mkt_ret20", "mkt_vol20", "breadth25", "sector_mom20"]
# 財務（EDINET）の特徴量は、データが揃ってから研究ループが検証して採用するまで入れない（research.feature_search の候補）
MODEL_FEATURES = [f for f in FEATURES if f not in MARKET_FEATURES] + EARN_FEATURES + EVENT_FEATURES + DIV_FEATURES + MODEL_MACRO_FEATURES + FLAG_COLS + ["n_strat"]

# 強めの正則化ほど期間をまたいで安定した（実験結果）。学習に使うのは先頭の設定（過去ペーパートレードで確かめた設定）
CANDIDATES = [
    dict(learning_rate=0.03, max_iter=150, max_leaf_nodes=7, min_samples_leaf=2000, l2_regularization=5.0),
    dict(learning_rate=0.05, max_iter=200, max_leaf_nodes=15, min_samples_leaf=500, l2_regularization=2.0),
    dict(learning_rate=0.05, max_iter=300, max_leaf_nodes=15, min_samples_leaf=200, l2_regularization=1.0),
    # さらに細かく学ぶ設定（手元の2つの種で +10.9% → +15.7%・t=1.83 と有望だが種によるぶれが大きいので、研究ループに判定させる）
    dict(learning_rate=0.05, max_iter=300, max_leaf_nodes=31, min_samples_leaf=100, l2_regularization=1.0),
]


# ---------------------------------------------------------------- labels
def forward_paths(o, h, l, c, ti, ci, H: int, fac=None) -> dict:
    """各シグナルの「翌日寄付き(エントリー)」と、その後 H 日分の四本値をまとめて取り出す。
    売買ルールを何通り試しても、この表を使い回せば価格データを読み直さずに済む。"""
    T = o.shape[0]
    entry = np.where(ti + 1 < T, o[np.minimum(ti + 1, T - 1), ci], np.nan)
    fo, fh, fl, fc = (np.full((H, len(ti)), np.nan, dtype="float32") for _ in range(4))
    for k in range(1, H + 1):
        idx = ti + k
        ok = idx < T
        ii = np.minimum(idx, T - 1)
        for arr, src in ((fo, o), (fh, h), (fl, l), (fc, c)):
            arr[k - 1] = np.where(ok, src[ii, ci], np.nan)
    out = dict(entry=entry, o=fo, h=fh, l=fl, c=fc)
    if fac is not None:  # エントリー日の「実際の株価 ÷ 分割調整後の株価」（呼値の単位を実際の株価で決めるため）
        out["fac"] = np.where(ti + 1 < T, fac[np.minimum(ti + 1, T - 1), ci], 1.0)
    return out


def outcomes(p: dict, atr: np.ndarray, k_tp: float, k_sl: float, H: int, stop: str = "intraday"):
    """トリプルバリア法: 利確(+k_tp×ATR)・損切(-k_sl×ATR)・H日経過のどれが先か。
    同じ日に両方に触れたら保守的に損切扱い。2日目以降に窓を開けて超えたら寄付きで約定。
    stop="intraday"（既定）: 損切は場中の逆指値（成行）なので、損切ラインより1ティック不利な値段で約定するものとする
      （買いは寄付・利確は指値・期限は引けで、想定どおりの値段）。安い銘柄ほど1ティックの割合が大きい（ジャーナル ラウンド9）。
    stop="close": 終値が損切ラインを割ったら翌日の寄付で売る（寄付は板寄せなので滑りなし。一晩の窓開けは受ける）。"""
    from .risk import tick_size
    entry = p["entry"]
    n = len(entry)
    tp, sl = entry + k_tp * atr, entry - k_sl * atr
    fac = p.get("fac")
    fac = np.ones(n) if fac is None else np.nan_to_num(np.asarray(fac, dtype="float64"), nan=1.0)
    # 呼値は「その日の実際の株価」で決まる（分割調整後の株価は、後で分割した銘柄では実際より低い）
    slip = tick_size(np.nan_to_num(entry, nan=1000.0) * fac) / fac
    done = np.zeros(n, bool)
    lab = np.zeros(n, dtype="int8")
    exit_px = np.full(n, np.nan)
    exit_k = np.full(n, np.nan)
    pending = np.zeros(n, bool)  # stop="close": 前日の終値で割った → 今日の寄付で売る
    for k in range(1, H + 1):
        hi, lo, op, cl = p["h"][k - 1], p["l"][k - 1], p["o"][k - 1], p["c"][k - 1]
        ok = ~np.isnan(hi)
        if stop == "close":
            hit_sl = ok & ~done & pending
            exit_px = np.where(hit_sl, op, exit_px)
            hit_tp = ok & ~done & ~hit_sl & (hi >= tp)
        else:
            hit_sl = ok & ~done & (lo <= sl)
            hit_tp = ok & ~done & (hi >= tp) & ~hit_sl
            exit_px = np.where(hit_sl, (np.minimum(op, sl) if k > 1 else sl) - slip, exit_px)
        exit_px = np.where(hit_tp, np.maximum(op, tp) if k > 1 else tp, exit_px)
        lab = np.where(hit_tp, 1, lab)
        exit_k = np.where(hit_sl | hit_tp, k, exit_k)
        done |= hit_sl | hit_tp
        if stop == "close":
            pending = ok & ~done & (cl <= sl)
    last = p["c"][H - 1]
    timeout = ~done & ~np.isnan(last)
    exit_px = np.where(timeout, last, exit_px)
    exit_k = np.where(timeout, H, exit_k)
    complete = (done | timeout) & np.isfinite(entry) & np.isfinite(atr)
    return dict(entry=entry, tp=tp, sl=sl, label=lab, ret=exit_px / entry - 1, exit_days=exit_k, complete=complete)


def label(ind: dict, df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """現在の売買ルールでラベル付け。complete=False は結果確定前。"""
    H = cfg["labeling"]["horizon_days"]
    o, h, l, c = (ind[k].to_numpy() for k in ("open", "high", "low", "close"))
    fac = ind["split_factor"].to_numpy() if "split_factor" in ind else None
    p = forward_paths(o, h, l, c, df["_ti"].to_numpy(), df["_ci"].to_numpy(), H, fac=fac)
    r = outcomes(p, df["atr"].to_numpy(), cfg["labeling"]["tp_atr"], cfg["labeling"]["sl_atr"], H,
                 stop=cfg["labeling"].get("stop_mode", "intraday"))
    df = df.copy()
    for k in ("entry", "tp", "sl", "label", "ret", "exit_days", "complete"):
        df[k] = r[k]
    df["label"] = df["label"].astype(int)
    return df


# ---------------------------------------------------------------- AIの学習目標（3つの見方の平均）
def plan_horizon(cfg: dict) -> int:
    """使っている売買ルールの中で最長の保有日数（学習とテストの間を空ける日数に使う）。"""
    lab = cfg["labeling"]
    return max([lab["horizon_days"]] + [p["horizon_days"] for p in (lab.get("per_strategy") or {}).values()])


def with_plan_outcomes(d: pd.DataFrame, cfg: dict, store=None) -> pd.DataFrame:
    """各シグナルに、実際に使う売買ルール（戦略ごと）での結果（ret_plan など）を付ける。"""
    from .backtest import apply_plans
    rel = apply_plans(d[["date", "code", "strategies", "atr", "ret", "label", "exit_days", "complete"]].copy(), cfg, store)
    return d.assign(ret_plan=rel["ret"].astype(float), label_plan=rel["label"].astype(float),
                    exit_plan=rel["exit_days"].astype(float), complete_plan=rel["complete"].astype(bool))


def model_features(cfg: dict) -> list[str]:
    """AIに渡す特徴量。毎週の研究ループで採用された候補の特徴量があれば足す。"""
    extra = [f for f in (cfg["learning"].get("extra_features") or []) if f not in MODEL_FEATURES]
    return MODEL_FEATURES + extra


def model_params(cfg: dict) -> dict:
    """AIのモデル設定。毎週の研究ループで採用された設定があればそれ、無ければ最も正則化の強い設定。"""
    return cfg["learning"].get("model_params") or CANDIDATES[0]


SPECIALIST_MIN = 5000  # 専門モデルを作るのに必要なシグナル数（2010年より前は決算日のデータが少なく、作らない）
SPECIALIST_CANDIDATES = ["earnings_drift"]  # 研究ループが判定する専門モデル（成績の柱の決算好反応だけ。選び過ぎを避けて事前に決めた1つ）


def specialists(cfg: dict) -> list[str]:
    """採用済みの専門モデル（毎週の研究ループが判定して保存したもの）。"""
    return list(cfg["learning"].get("specialists") or [])


def fit_models(tr: pd.DataFrame, cfg: dict, params: dict | None = None, features: list[str] | None = None,
               seeds: int = 1, specialists_: list[str] | None = None) -> dict:
    """3つの学習目標でモデルを作る。tr には with_plan_outcomes の列が必要。features で特徴量を差し替えて比較できる。
    seeds>1 なら乱数の種を変えたモデルを複数作り、予測を平均する（本番用。研究の比較は時間の都合で1つ）。
    specialists_（既定は採用済みのもの）: その戦略のシグナルだけで学ぶ専門モデルも作る（ジャーナル 50）。"""
    params = params or model_params(cfg)
    features = features or model_features(cfg)
    specs = specialists(cfg) if specialists_ is None else specialists_
    if seeds > 1:
        out = dict(members=[fit_models(tr, cfg, dict(params, random_state=42 + k), features, specialists_=[]) for k in range(seeds)],
                   features=features)
    else:
        X, w = tr[features], _weights(tr, cfg["learning"])
        lo, hi = tr["ret_plan"].quantile([0.01, 0.99])  # 極端な値に引っ張られないように
        out = dict(
            win=_fit(params, X, tr["label"], w),
            ret=_fit(params, X, tr["ret_plan"].clip(lo, hi) - cfg["research"]["cost"], w, reg=True),
            rank=_fit(params, X, tr.groupby("date")["ret_plan"].rank(pct=True) - 0.5, w, reg=True),
            features=features,
        )
    spec = {}
    for k in specs:
        sub = tr[tr[f"s_{k}"] == 1] if f"s_{k}" in tr else tr.iloc[:0]
        if len(sub) >= SPECIALIST_MIN:
            spec[k] = fit_models(sub, cfg, params, features, seeds, specialists_=[])
    if spec:
        out["spec"] = spec
    return out


def _parts(models: dict, df: pd.DataFrame) -> pd.DataFrame:
    """3つのモデルの生の予測（複数の種があれば平均）。"""
    if "members" in models:
        ps = [_parts(m, df) for m in models["members"]]
        return sum(ps) / len(ps)
    X = df[models.get("features", MODEL_FEATURES)]
    return pd.DataFrame({"win": models["win"].predict_proba(X)[:, 1], "ret": models["ret"].predict(X),
                         "rank": models["rank"].predict(X)}, index=df.index)


def predict(models: dict, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(勝率予測, AIの評価)。AIの評価は、3つのモデルの予測それぞれの「同じ日の候補内での順位」の平均（0〜1）。"""
    parts = _parts(models, df)
    sc = parts.groupby(df["date"].to_numpy()).rank(pct=True).mean(axis=1).to_numpy()
    for k, sm in (models.get("spec") or {}).items():
        sc = _with_specialist(sm, df, sc, f"s_{k}")
    return parts["win"].to_numpy(), sc


def _with_specialist(sm: dict, df: pd.DataFrame, sc: np.ndarray, flag: str) -> np.ndarray:
    """その戦略の候補どうしの並び順だけを「全体のAIの順位と専門モデルの順位の平均」で並べ直す。
    その戦略の候補が占めていた評価の値をそのまま配り直すので、他の戦略との配分は変わらない（ジャーナル 50）。"""
    if flag not in df:
        return sc
    mask = (df[flag] == 1).to_numpy()
    if not mask.any():
        return sc
    sub = df[mask]
    dates = sub["date"].to_numpy()
    sp = _parts(sm, sub).groupby(dates).rank(pct=True).mean(axis=1).to_numpy()
    g = pd.DataFrame({"d": dates, "v": sc[mask], "sp": sp})
    g["new"] = 0.5 * g.groupby("d")["v"].rank(pct=True) + 0.5 * g.groupby("d")["sp"].rank(pct=True)
    out = g["v"].to_numpy().copy()
    for _, x in g.groupby("d"):
        pos = x.index.to_numpy()
        out[pos[np.argsort(x["new"].to_numpy(), kind="stable")]] = np.sort(x["v"].to_numpy())
    sc = sc.copy()
    sc[mask] = out
    return sc


def agreement(models: dict, df: pd.DataFrame) -> np.ndarray:
    """3つのモデルがどれだけ揃って高く評価しているか: 同じ日の候補内での順位（0〜1）の3つのうち一番低いもの。
    合意モード（paper_trading.consensus）では、これが高い候補だけを買う。"""
    parts = _parts(models, df)
    return parts.groupby(df["date"].to_numpy()).rank(pct=True).min(axis=1).to_numpy()


def bundle_predict(bundle: dict, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """保存済みモデルでの (勝率予測, AIの評価)。ラウンド7までの旧モデル（勝ちやすさだけ）にも対応。"""
    if "models" in bundle:
        return predict(bundle["models"], df)
    prob = bundle["model"].predict_proba(df[bundle["features"]])[:, 1]
    return prob, prob


# ---------------------------------------------------------------- strategy weights
def strategy_stats(ds: pd.DataFrame, lookback_days: int = 120) -> pd.DataFrame:
    d = ds[ds["complete"]]
    if d.empty:
        return pd.DataFrame(columns=["strategy", "n", "win_rate", "avg_ret", "weight"])
    recent_dates = sorted(d["date"].unique())[-lookback_days:]
    d = d[d["date"].isin(recent_dates)]
    rows = []
    for s in STRATEGIES:
        sub = d[d[f"s_{s.key}"] == 1]
        n = len(sub)
        wr = sub["label"].mean() if n else np.nan
        ar = sub["ret"].mean() if n else np.nan
        # 期待値 +2.5% で重み1.5、-2.5% で0.5。サンプルが少ない間は1に寄せる
        raw = 1 + 20 * (ar if n else 0)
        shrink = n / (n + 50)
        w = float(np.clip(1 + shrink * (raw - 1), 0.2, 2.0))
        rows.append(dict(strategy=s.key, n=n, win_rate=wr, avg_ret=ar, weight=w))
    return pd.DataFrame(rows)


def rule_score(df: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    score = sum(df[f"s_{s.key}"] * weights.get(s.key, 1.0) for s in STRATEGIES)
    # 地合いと流動性で軽く補正
    score = score + 0.3 * (df["mkt_dev75"] > 0) + 0.2 * (df["rs60"] > 0)
    return score


# ---------------------------------------------------------------- training
def _weights(df: pd.DataFrame, lc: dict) -> np.ndarray:
    """新しいデータほど重く、危機局面（特例）は軽く。危機のデータも捨てずに参考程度に学ぶ。"""
    age = (df["date"].max() - df["date"]).dt.days.to_numpy()
    w = 0.5 ** (age / lc["recency_half_life_days"])
    if "crisis" in df:
        w = np.where(df["crisis"].to_numpy() == 1, w * lc.get("crisis_weight", 0.3), w)
    return w


def _evaluate(df: pd.DataFrame, score: np.ndarray, prob: np.ndarray | None = None, top_k: int = 5) -> dict:
    """score の高い順に選んだときの成績。AUC（識別力）は勝率予測 prob で測る。"""
    d = df.assign(prob=score)
    y = d["label"].to_numpy()
    p = score if prob is None else prob
    auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else 0.5
    q = d["prob"].quantile(0.9)
    top = d[d["prob"] >= q]
    daily = d.sort_values("prob", ascending=False).groupby("date").head(top_k)
    out = dict(
        n=int(len(d)), auc=float(auc),
        base_win=float(d["label"].mean()), base_ret=float(d["ret"].mean()),
        top10_win=float(top["label"].mean()), top10_ret=float(top["ret"].mean()),
        daily_top_win=float(daily["label"].mean()), daily_top_ret=float(daily["ret"].mean()),
    )
    if "crisis" in d:  # 平常時と危機局面（特例）を分けた成績
        for name, flag in (("normal", 0), ("crisis", 1)):
            dd, dl = d[d["crisis"] == flag], daily[daily["crisis"] == flag]
            out[name] = dict(n=int(len(dd)), base_ret=float(dd["ret"].mean()) if len(dd) else None,
                             daily_top_ret=float(dl["ret"].mean()) if len(dl) else None,
                             daily_top_win=float(dl["label"].mean()) if len(dl) else None)
    return out


def ai_weight(m: dict) -> float:
    """並べ方でのAIの評価の割合。未知期間でAIが全シグナル平均を上回っていればAIだけで並べる。"""
    return 1.0 if m["daily_top_ret"] - m["base_ret"] > 0 else 0.5


def _merit(m: dict) -> float:
    """採用判定の指標: 毎日の上位推奨が全シグナル平均をどれだけ上回ったか（＋識別力を少し加味）。"""
    return (m["daily_top_ret"] - m["base_ret"]) + 0.02 * (m["auc"] - 0.5)


def _usable(s: pd.Series) -> bool:
    """学習に使える列か。scikit-learn は20万行を超えると一部を抜き出して区切り値を決めるので、
    まばらな列は抜き出した中で値が1種類以下になりエラーになる。十分な件数と、2番目に多い値の件数も確かめる。"""
    n = int(s.notna().sum())
    if n < 2000:
        return False
    vc = s.value_counts()
    if len(vc) < 2:
        return False
    return len(vc) > 20 or int(vc.iloc[1]) >= 200


def _fit(params: dict, X, y, w, reg: bool = False):
    """学習期間に値が無い（または1種類しかない）特徴量は外して学習する。
    例: 2008年までのデータには適時開示がまだ無い。新しい scikit-learn はこうした列でエラーになる。
    列の選択ごと1つのモデルにまとめるので、予測時は全特徴量をそのまま渡せばよい。"""
    cols = [c for c in X.columns if _usable(X[c])]
    m = Pipeline([("cols", ColumnTransformer([("keep", "passthrough", cols)], remainder="drop")),
                  ("hgb", (HistGradientBoostingRegressor if reg else HistGradientBoostingClassifier)(**{"random_state": 42, **params}))])
    m.fit(X, y, hgb__sample_weight=w)
    return m


def train(con, cfg: dict, ds: pd.DataFrame, data_end: str) -> dict | None:
    lc = cfg["learning"]
    H = plan_horizon(cfg)
    d = ds[ds["complete"]]
    if len(d) < lc["min_train_samples"]:
        log.warning("学習データ不足: %d 件", len(d))
        return None
    d = with_plan_outcomes(d, cfg)
    d = d[d["complete_plan"]].sort_values("date").reset_index(drop=True)
    # 成績は実際に使う売買ルールでのリターンで測る
    ev = lambda df: df.assign(ret=df["ret_plan"])

    dates = np.array(sorted(d["date"].unique()))
    split = dates[int(len(dates) * (1 - lc["holdout_fraction"]))]
    purge_end = dates[max(0, np.searchsorted(dates, split) - H)]  # ラベル期間の重複を除去
    tr, ho = d[d["date"] < purge_end], d[d["date"] >= split]
    seeds = int(cfg["learning"].get("seeds", 1))
    models = fit_models(tr, cfg, seeds=seeds)
    prob, sc = predict(models, ho)
    best_met = _evaluate(ev(ho), sc, prob)
    log.info("  未知期間: AUC %.3f / 日次上位勝率 %.1f%% / 期待値 %+.2f%%（全体 %+.2f%%）",
             best_met["auc"], best_met["daily_top_win"] * 100, best_met["daily_top_ret"] * 100, best_met["base_ret"] * 100)

    # ---- チャンピオン（現役）との対決：両者にとって未知の期間で比較
    champ = load_active(con, cfg)
    verdict = "初回モデル"
    if champ is not None:
        window = ho[ho["date"] > pd.Timestamp(champ["data_end"])]
        try:
            c_prob, c_sc = bundle_predict(champ, window)
        except Exception:  # 特徴量の構成が変わった旧モデルは比較不能
            window = window.iloc[:0]
        if len(window) >= 300:
            c_met = _evaluate(ev(window), c_sc, c_prob)
            n_prob, n_sc = predict(models, window)
            n_met = _evaluate(ev(window), n_sc, n_prob)
            if _merit(n_met) < _merit(c_met):
                log.info("挑戦者が現役に敗北 → 現役モデルを維持")
                _record_stats(con, cfg, ds, data_end)
                return champ
            verdict = f"現役に勝利 (期待値 {n_met['daily_top_ret']:+.2%} vs {c_met['daily_top_ret']:+.2%})"
        else:
            verdict = "比較期間不足のため最新データで更新"

    # 全データで再学習して本番モデルに
    final = fit_models(d, cfg, seeds=seeds)
    sample = ho.sample(min(4000, len(ho)), random_state=0)
    feats = model_features(cfg)
    pi = permutation_importance((models["members"][0] if "members" in models else models)["win"], sample[feats], sample["label"],
                                n_repeats=3, random_state=0, scoring="roc_auc")
    imp = sorted(zip(feats, pi.importances_mean), key=lambda t: -t[1])[:12]

    auc = best_met["auc"]
    # 並べ方は過去ペーパートレードと同じ「AIの評価だけ」。AIとルールを混ぜると年率・最大下落とも悪化した
    # （AIのみ +10.0%・−27% ／ 半々 +7.8%・−36%、ジャーナル ラウンド9）。
    # 直近の未知期間でAIが全シグナル平均を下回ったときだけ、念のためルールと半々にする
    alpha = ai_weight(best_met)
    version = datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y%m%d-%H%M")
    best_met.update(alpha=alpha, verdict=verdict, importance=[(k, float(v)) for k, v in imp],
                    train_n=int(len(d)), holdout_from=str(split)[:10], targets=["win", "ret", "rank"], seeds=seeds)
    path = cfg["data"]["model_dir"] / f"model_{version}.joblib"
    bundle = dict(models=final, features=model_features(cfg), version=version, alpha=alpha,
                  metrics=best_met, data_end=data_end)
    joblib.dump(bundle, path)
    con.execute("UPDATE models SET active=0")
    con.execute("INSERT OR REPLACE INTO models VALUES (?,?,?,?,?,?,?,1)",
                (version, datetime.now().isoformat(timespec="seconds"), data_end, len(d),
                 json.dumps(model_params(cfg)), json.dumps(best_met, ensure_ascii=False), str(path)))
    con.commit()
    _record_stats(con, cfg, ds, data_end)
    log.info("新モデル v%s 採用: AUC %.3f, %s", version, auc, verdict)
    return bundle


def _record_stats(con, cfg, ds, data_end):
    st = strategy_stats(ds)
    st["date"] = data_end
    con.executemany("INSERT OR REPLACE INTO strategy_stats VALUES (?,?,?,?,?,?)",
                    st[["date", "strategy", "n", "win_rate", "avg_ret", "weight"]].itertuples(index=False, name=None))
    con.commit()


def load_active(con, cfg: dict) -> dict | None:
    row = con.execute("SELECT path FROM models WHERE active=1 ORDER BY version DESC LIMIT 1").fetchone()
    if not row:
        return None
    try:
        return joblib.load(row[0])
    except Exception as e:
        log.warning("モデル読み込み失敗: %s", e)
        return None


def load_strategy_weights(con) -> dict[str, float]:
    rows = con.execute("SELECT strategy, weight FROM strategy_stats WHERE date = (SELECT MAX(date) FROM strategy_stats)").fetchall()
    return dict(rows)


def score(df: pd.DataFrame, bundle: dict | None, weights: dict[str, float]) -> pd.DataFrame:
    df = df.copy()
    df["rule_score"] = rule_score(df, weights)
    if bundle is not None and len(df):
        df["prob"], df["ai_score"] = bundle_predict(bundle, df)
        df["agreement"] = agreement(bundle["models"], df) if "models" in bundle else np.nan
        alpha = ai_weight(bundle["metrics"]) if "daily_top_ret" in bundle.get("metrics", {}) else bundle["alpha"]
        df["final_score"] = alpha * df["ai_score"].rank(pct=True) + (1 - alpha) * df["rule_score"].rank(pct=True)
        df["model_version"] = bundle["version"]
    else:
        df["prob"] = np.nan
        df["final_score"] = df["rule_score"].rank(pct=True)
        df["model_version"] = None
    return df.sort_values("final_score", ascending=False)
