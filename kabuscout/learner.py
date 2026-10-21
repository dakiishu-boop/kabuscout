"""自己学習エンジン。

仕組み:
1. 過去の全営業日・全銘柄でシグナルを再現し、その後どうなったかを
   トリプルバリア法でラベル付け（翌日寄付きで買い → 利確/損切/期限のどれが先か）。
2. 「どんな状況のシグナルが勝ちやすいか」を勾配ブースティングで学習。新しいデータほど重く扱う。
3. 再学習のたびに複数のモデル候補を時系列ホールドアウトで比較し、
   さらに現役モデル(チャンピオン)と「両者にとって未知の期間」で勝負させ、勝った方だけ採用。
4. 戦略ごとの直近成績から重みを更新（今の相場で効いている網を重く）。
日々データが増えるので、放っておくほど学習データが増え、相場の変化にも追従する。
"""
from __future__ import annotations

import json
import logging
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

from .indicators import FEATURES
from .strategies import FLAG_COLS, STRATEGIES

log = logging.getLogger(__name__)
# 地合い系（全銘柄共通の値）は特定の時期の丸暗記を招き、検証で成績が不安定になったため
# 銘柄スコアのモデルからは外す（地合いは通知文と戦略の重みで扱う）
MARKET_FEATURES = ["mkt_dev75", "mkt_ret5", "mkt_ret20", "mkt_vol20", "breadth25", "sector_mom20"]
MODEL_FEATURES = [f for f in FEATURES if f not in MARKET_FEATURES] + FLAG_COLS + ["n_strat"]

# 強めの正則化ほど期間をまたいで安定した（実験結果）
CANDIDATES = [
    dict(learning_rate=0.03, max_iter=150, max_leaf_nodes=7, min_samples_leaf=2000, l2_regularization=5.0),
    dict(learning_rate=0.05, max_iter=200, max_leaf_nodes=15, min_samples_leaf=500, l2_regularization=2.0),
    dict(learning_rate=0.05, max_iter=300, max_leaf_nodes=15, min_samples_leaf=200, l2_regularization=1.0),
]


# ---------------------------------------------------------------- dataset
def extract(ind: dict, sigs: dict[str, pd.DataFrame], only_last: bool = False) -> pd.DataFrame:
    """シグナルが出た (日付, 銘柄) の特徴量を縦持ちで取り出す。"""
    any_sig = None
    for m in sigs.values():
        any_sig = m if any_sig is None else (any_sig | m)
    arr = any_sig.to_numpy()
    if only_last:
        arr = arr.copy()
        arr[:-1] = False
    ti, ci = np.nonzero(arr)
    dates, codes = any_sig.index, any_sig.columns
    out = {"date": dates[ti], "code": codes[ci], "_ti": ti, "_ci": ci}
    for f in FEATURES + ["close", "atr", "turnover"]:
        out[f] = ind[f].to_numpy()[ti, ci]
    for s in STRATEGIES:
        out[f"s_{s.key}"] = sigs[s.key].to_numpy()[ti, ci].astype(int)
    df = pd.DataFrame(out)
    df["n_strat"] = df[FLAG_COLS].sum(axis=1)
    flags = df[FLAG_COLS].to_numpy().astype(bool)
    keys = np.array([s.key for s in STRATEGIES])
    df["strategies"] = [",".join(keys[row]) for row in flags]
    return df


def label(ind: dict, df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """トリプルバリア法でラベル付け。complete=False は結果確定前。"""
    H = cfg["labeling"]["horizon_days"]
    k_tp, k_sl = cfg["labeling"]["tp_atr"], cfg["labeling"]["sl_atr"]
    o, h, l, c = (ind[k].to_numpy() for k in ("open", "high", "low", "close"))
    T = o.shape[0]
    ti, ci = df["_ti"].to_numpy(), df["_ci"].to_numpy()
    n = len(df)

    nxt = np.minimum(ti + 1, T - 1)
    entry = np.where(ti + 1 < T, o[nxt, ci], np.nan)
    atr = df["atr"].to_numpy()
    tp, sl = entry + k_tp * atr, entry - k_sl * atr

    done = np.zeros(n, bool)
    lab = np.zeros(n)
    exit_px = np.full(n, np.nan)
    exit_k = np.full(n, np.nan)
    for k in range(1, H + 1):
        idx = ti + k
        ok = idx < T
        ii = np.minimum(idx, T - 1)
        hi, lo, op = h[ii, ci], l[ii, ci], o[ii, ci]
        hit_sl = ok & ~done & (lo <= sl)
        hit_tp = ok & ~done & (hi >= tp) & ~hit_sl  # 同日両方なら保守的に損切扱い
        exit_px = np.where(hit_sl, np.minimum(op, sl) if k > 1 else sl, exit_px)
        exit_px = np.where(hit_tp, np.maximum(op, tp) if k > 1 else tp, exit_px)
        lab = np.where(hit_tp, 1, lab)
        exit_k = np.where(hit_sl | hit_tp, k, exit_k)
        done |= hit_sl | hit_tp
    end = ti + H
    timeout = ~done & (end < T)
    exit_px = np.where(timeout, c[np.minimum(end, T - 1), ci], exit_px)
    exit_k = np.where(timeout, H, exit_k)

    df = df.copy()
    df["entry"], df["tp"], df["sl"] = entry, tp, sl
    df["label"] = lab.astype(int)
    df["ret"] = exit_px / entry - 1
    df["exit_days"] = exit_k
    df["complete"] = (done | timeout) & np.isfinite(entry) & np.isfinite(atr)
    return df


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
def _weights(dates: pd.Series, half_life: float) -> np.ndarray:
    age = (dates.max() - dates).dt.days.to_numpy()
    return 0.5 ** (age / half_life)


def _evaluate(df: pd.DataFrame, prob: np.ndarray, top_k: int = 5) -> dict:
    d = df.assign(prob=prob)
    y = d["label"].to_numpy()
    auc = roc_auc_score(y, prob) if len(np.unique(y)) > 1 else 0.5
    q = d["prob"].quantile(0.9)
    top = d[d["prob"] >= q]
    daily = d.sort_values("prob", ascending=False).groupby("date").head(top_k)
    return dict(
        n=int(len(d)), auc=float(auc),
        base_win=float(d["label"].mean()), base_ret=float(d["ret"].mean()),
        top10_win=float(top["label"].mean()), top10_ret=float(top["ret"].mean()),
        daily_top_win=float(daily["label"].mean()), daily_top_ret=float(daily["ret"].mean()),
    )


def _merit(m: dict) -> float:
    """採用判定の指標: 毎日の上位推奨が全シグナル平均をどれだけ上回ったか（＋識別力を少し加味）。"""
    return (m["daily_top_ret"] - m["base_ret"]) + 0.02 * (m["auc"] - 0.5)


def _fit(params: dict, X, y, w):
    m = HistGradientBoostingClassifier(random_state=42, **params)
    m.fit(X, y, sample_weight=w)
    return m


def train(con, cfg: dict, ds: pd.DataFrame, data_end: str) -> dict | None:
    lc = cfg["learning"]
    H = cfg["labeling"]["horizon_days"]
    d = ds[ds["complete"]].sort_values("date").reset_index(drop=True)
    if len(d) < lc["min_train_samples"]:
        log.warning("学習データ不足: %d 件", len(d))
        return None

    dates = np.array(sorted(d["date"].unique()))
    split = dates[int(len(dates) * (1 - lc["holdout_fraction"]))]
    purge_end = dates[max(0, np.searchsorted(dates, split) - H)]  # ラベル期間の重複を除去
    tr, ho = d[d["date"] < purge_end], d[d["date"] >= split]
    X_tr, y_tr = tr[MODEL_FEATURES], tr["label"]
    w_tr = _weights(tr["date"], lc["recency_half_life_days"])

    results = []
    for p in CANDIDATES:
        m = _fit(p, X_tr, y_tr, w_tr)
        met = _evaluate(ho, m.predict_proba(ho[MODEL_FEATURES])[:, 1])
        results.append((p, m, met))
        log.info("  候補 %s → AUC %.3f / 日次上位勝率 %.1f%% / 期待値 %+.2f%%",
                 p["max_leaf_nodes"], met["auc"], met["daily_top_win"] * 100, met["daily_top_ret"] * 100)
    best_p, best_m, best_met = max(results, key=lambda r: _merit(r[2]))

    # ---- チャンピオン（現役）との対決：両者にとって未知の期間で比較
    champ = load_active(con, cfg)
    verdict = "初回モデル"
    if champ is not None:
        window = ho[ho["date"] > pd.Timestamp(champ["data_end"])]
        try:
            c_prob = champ["model"].predict_proba(window[champ["features"]])[:, 1]
        except Exception:  # 特徴量の構成が変わった旧モデルは比較不能
            window = window.iloc[:0]
        if len(window) >= 300:
            c_met = _evaluate(window, c_prob)
            n_met = _evaluate(window, best_m.predict_proba(window[MODEL_FEATURES])[:, 1])
            if _merit(n_met) < _merit(c_met):
                log.info("挑戦者が現役に敗北 → 現役モデルを維持")
                _record_stats(con, cfg, ds, data_end)
                return champ
            verdict = f"現役に勝利 (期待値 {n_met['daily_top_ret']:+.2%} vs {c_met['daily_top_ret']:+.2%})"
        else:
            verdict = "比較期間不足のため最新データで更新"

    # 全データで再学習して本番モデルに
    final = _fit(best_p, d[MODEL_FEATURES], d["label"], _weights(d["date"], lc["recency_half_life_days"]))
    sample = ho.sample(min(4000, len(ho)), random_state=0)
    pi = permutation_importance(best_m, sample[MODEL_FEATURES], sample["label"],
                                n_repeats=3, random_state=0, scoring="roc_auc")
    imp = sorted(zip(MODEL_FEATURES, pi.importances_mean), key=lambda t: -t[1])[:12]

    auc = best_met["auc"]
    # モデルの実力（未知期間で全シグナル平均をどれだけ上回ったか）に応じてルールとの配合を決める
    lift = best_met["daily_top_ret"] - best_met["base_ret"]
    alpha = 0.2 if lift <= 0 else 0.5 if lift < 0.005 else 0.8
    version = datetime.now().strftime("%Y%m%d-%H%M")
    best_met.update(alpha=alpha, verdict=verdict, importance=[(k, float(v)) for k, v in imp],
                    train_n=int(len(d)), holdout_from=str(split)[:10])
    path = cfg["data"]["model_dir"] / f"model_{version}.joblib"
    bundle = dict(model=final, features=MODEL_FEATURES, version=version, alpha=alpha,
                  metrics=best_met, data_end=data_end)
    joblib.dump(bundle, path)
    con.execute("UPDATE models SET active=0")
    con.execute("INSERT OR REPLACE INTO models VALUES (?,?,?,?,?,?,?,1)",
                (version, datetime.now().isoformat(timespec="seconds"), data_end, len(d),
                 json.dumps(best_p), json.dumps(best_met, ensure_ascii=False), str(path)))
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
        df["prob"] = bundle["model"].predict_proba(df[bundle["features"]])[:, 1]
        alpha = bundle["alpha"]
        df["final_score"] = alpha * df["prob"].rank(pct=True) + (1 - alpha) * df["rule_score"].rank(pct=True)
        df["model_version"] = bundle["version"]
    else:
        df["prob"] = np.nan
        df["final_score"] = df["rule_score"].rank(pct=True)
        df["model_version"] = None
    return df.sort_values("final_score", ascending=False)
