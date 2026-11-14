"""研究用の共通部品（research/earn_timing.py・research/event_nets.py が使う）。

本番には触らない:
- data/pool.parquet・strategy_params.json・DB には書き込まない（pool.build_full は保存してしまうので使わない）
- 作ったシグナル表・予測・結果は data/research_cache/ にだけ置く（data/ は git の対象外）
- kabuscout の関数の差し替えは、このプロセスの中だけ

比べ方はジャーナルと同じ: 年ごとの学び直し（backtest.walk_forward）→ 本番と同じ口座（backtest.simulate）→
年ごとの差の t。口座の年率は種でぶれる（§80）ので、候補ごとの値（上位2の1回平均・順位相関）と、
AI を使わない「網そのものの成績」も必ず一緒に出す。
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from kabuscout import backtest, data, learner, pool, strategies  # noqa: E402
from kabuscout.db import connect  # noqa: E402
from kabuscout.pipeline import params_path  # noqa: E402

log = logging.getLogger("lab")
KEEP = ["date", "code", "strategies", "prob", "win_prob", "agreement", "ret", "label", "exit_days", "complete",
        "entry", "close", "atr", "crisis", "mkt_dev75", "mkt_ret20", "n_strat"]


def setup_logging() -> None:
    import warnings
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")


def cache_dir(cfg: dict) -> Path:
    d = cfg["data"]["db_path"].parent / "research_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def universe(cfg: dict) -> pd.DataFrame:
    """銘柄一覧（DB にあるものを使う。空のときだけ JPX から取る）。"""
    con = connect(cfg["data"]["db_path"])
    cur = pd.read_sql("SELECT * FROM universe", con)
    return cur if not cur.empty else data.refresh_universe(con, cfg)


def calendar(cfg: dict, store) -> pd.DatetimeIndex:
    proxy = cfg["universe"]["market_proxy"]
    return store.load(fields=["close"], codes=[proxy])["close"][proxy].dropna().index


def build_pool(cfg: dict, store, uni: pd.DataFrame, name: str, rebuild: bool = False) -> pd.DataFrame:
    """シグナル表を最初から作る（pool.build_full と同じ区切り方。本番の pool.parquet には保存しない）。"""
    p = cache_dir(cfg) / f"pool_{name}.parquet"
    if p.exists() and not rebuild:
        pl = pd.read_parquet(p)
        log.info("シグナル表 %s: 保存済みを使用（%d行、〜%s）。作り直すなら --rebuild", name, len(pl), pl["date"].max().date())
        return pl
    start = pd.Timestamp(cfg["data"]["history_start"])
    end = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
    parts, y = [], start
    while y < end:
        y2 = min(y + pd.DateOffset(years=3), end)
        log.info("  シグナル表 %s を作成中: %s〜%s", name, y.date(), y2.date())
        _, df = pool.build_window(store, cfg, uni, (y - pd.Timedelta(days=pool.WARMUP_DAYS)).strftime("%Y-%m-%d"),
                                  (y2 + pd.Timedelta(days=pool._lookahead(cfg))).strftime("%Y-%m-%d"),
                                  keep_from=y.strftime("%Y-%m-%d"), keep_to=y2.strftime("%Y-%m-%d"))
        if len(df):
            parts.append(df)
        y = y2
    pl = pd.concat(parts, ignore_index=True).sort_values(["date", "code"]).reset_index(drop=True)
    pl.to_parquet(p, compression="zstd")
    log.info("シグナル表 %s: %d行", name, len(pl))
    return pl


def dataset(cfg: dict, pl: pd.DataFrame) -> pd.DataFrame:
    """本番と同じ戦略の条件（strategy_params.json があればそれ、無ければ既定値）を当てはめた候補表。"""
    return strategies.apply_flags(pl, strategies.load_params(params_path(cfg)))


def outcomes(cfg: dict, ds: pd.DataFrame) -> pd.DataFrame:
    """各候補に、実際の売買ルール（戦略ごと）での結果 ret_plan を付ける（結果が確定した行だけ）。"""
    d = learner.with_plan_outcomes(ds[ds["complete"]], cfg)
    return d[d["complete_plan"]].reset_index(drop=True)


def _t(x: pd.Series) -> float:
    x = x.dropna()
    return float(x.mean() / x.std(ddof=1) * np.sqrt(len(x))) if len(x) > 2 and x.std(ddof=1) > 0 else float("nan")


def signal_stats(d: pd.DataFrame, mask, cost: float, since: int | None = None) -> dict:
    """網そのものの成績（AI を使わず、当てはまった候補を全部買ったとした1回あたりの手取り）。t は年ごとの平均から。"""
    x = d[np.asarray(mask, dtype=bool)]
    if since:
        x = x[x["date"].dt.year >= since]
    if x.empty:
        return dict(n=0)
    net = x["ret_plan"] - cost
    by = net.groupby(x["date"].dt.year).mean()
    return dict(n=int(len(x)), per_year=float(len(x) / max(by.size, 1)), mean=float(net.mean()), win=float((net > 0).mean()),
                pos_years=float((by > 0).mean()), years=int(by.size), t_years=_t(by),
                first=int(by.index.min()), last=int(by.index.max()))


def fmt_stats(label: str, s: dict) -> str:
    if not s.get("n"):
        return f"| {label} | 0 | | | | |"
    return (f"| {label} | {s['n']:,}（年{s['per_year']:.0f}） | {s['mean']:+.2%} | {s['win']:.0%} | "
            f"{s['pos_years']:.0%}（{s['years']}年） | {s['t_years']:+.2f} |")


STATS_HEAD = "| 候補 | 件数 | 1回の手取り | 勝率 | プラスの年 | t（年ごと） |\n|---|---|---|---|---|---|"


# ---------------------------------------------------------------- AI ＋ 口座
def predictions(cfg: dict, ds: pd.DataFrame, name: str, seed: int, start_year: int, rebuild: bool = False) -> pd.DataFrame:
    """年ごとの学び直しの予測（種ごとに保存して使い回す）。"""
    p = cache_dir(cfg) / f"preds_{name}_s{seed}_{start_year}.parquet"
    if p.exists() and not rebuild:
        return pd.read_parquet(p)
    pr = backtest.walk_forward(cfg, ds, start_year, params=dict(learner.model_params(cfg), random_state=42 + seed))
    pr = pr[pr["complete"]]
    cols = [c for c in dict.fromkeys(KEEP + list(strategies.FLAG_COLS) + ["earn_early", "earn_age"]) if c in pr.columns]
    pr = pr[cols].reset_index(drop=True)
    pr.to_parquet(p, compression="zstd")
    return pr


def account(cfg: dict, pr: pd.DataFrame, cal: pd.DatetimeIndex) -> dict:
    sim = backtest.simulate(pr, cal[cal >= pr["date"].min()], cfg, "ai")
    eq = sim["equity"]
    ends = eq.groupby(eq.index.year).last()
    yearly = pd.Series(ends.to_numpy() / np.r_[eq.iloc[0], ends.to_numpy()[:-1]] - 1, index=ends.index)
    out = dict(stats=backtest._stats(eq, sim["trades"], sim["exposure"]), yearly=yearly)
    if eq.index[0] < pd.Timestamp(backtest.BASE_START) < eq.index[-1]:
        out["since"] = backtest.period_stats(sim, backtest.BASE_START)
    return out


def candidates(pr: pd.DataFrame, cost: float, since: int = 2011) -> dict:
    """口座の経路によらない値: その日の上位2の1回平均（コスト前）と、その日の候補の中での順位相関。"""
    x = pr[pr["date"].dt.year >= since]
    if x.empty:
        x = pr
    top = x.sort_values(["date", "prob"], ascending=[True, False]).groupby("date").head(2)
    ic = x.groupby("date").apply(lambda g: g["prob"].rank().corr(g["ret"].rank()) if len(g) >= 10 else np.nan)
    mix = {}
    for k in strategies.FLAG_COLS:
        if k in top.columns:
            m = top[k] == 1
            if m.any():
                mix[k[2:]] = dict(n=int(m.sum()), net=float(top.loc[m, "ret"].mean() - cost))
    return dict(top2=float(top["ret"].mean()), top2_n=int(len(top)), ic=float(ic.mean()), mix=mix)


def run_arm(cfg: dict, ds: pd.DataFrame, cal: pd.DatetimeIndex, name: str, seeds: list[int], start_year: int,
            rebuild: bool = False) -> list[dict]:
    runs = []
    for s in seeds:
        log.info("AI %s・種%d: 年ごとの学び直し（1回10分前後）", name, s)
        pr = predictions(cfg, ds, name, s, start_year, rebuild)
        r = account(cfg, pr, cal)
        r["cand"] = candidates(pr, cfg["research"]["cost"])
        r["seed"] = s
        st = r["stats"]
        log.info("  %s 種%d: 年率 %+.1f%%・最大下落 %.1f%%・最悪の年 %+.1f%%・上位2の1回平均 %+.2f%%・順位相関 %.3f",
                 name, s, st["cagr"] * 100, st["max_dd"] * 100, st["worst_year"] * 100, r["cand"]["top2"] * 100, r["cand"]["ic"])
        runs.append(r)
    return runs


def paired(base: list[dict], arm: list[dict]) -> dict:
    """年ごとの差（候補 − 今）。種ごとと、種の平均どうし。採用の目安はジャーナルと同じ t≥2.5（平均でも種ごとでも）。"""
    per = [(a["yearly"] - b["yearly"]).dropna() for a, b in zip(arm, base)]
    avg = pd.concat(per, axis=1).mean(axis=1)
    return dict(mean=float(avg.mean()), t=_t(avg), pos=float((avg > 0).mean()), years=int(len(avg)),
                per_seed=[dict(seed=a["seed"], mean=float(d.mean()), t=_t(d)) for a, d in zip(arm, per)],
                yearly={int(k): float(v) for k, v in avg.items()})


def report_ai(title: str, arms: dict[str, list[dict]], base_key: str) -> str:
    """ジャーナルに貼れる形の表。"""
    L = [f"### {title}", "", "| | 年率（種ごと） | 最大下落 | 最悪の年 | 2019年〜の年率 | 上位2の1回平均（2011〜・コスト前） | 順位相関 | 取引/年 |",
         "|---|---|---|---|---|---|---|---|"]
    j = lambda runs, f: " / ".join(f(r) for r in runs)  # Python 3.9 でも動くように、f文字列の入れ子は使わない
    for k, runs in arms.items():
        cells = [j(runs, lambda r: "{:+.1%}".format(r["stats"]["cagr"])), j(runs, lambda r: "{:.1%}".format(r["stats"]["max_dd"])),
                 j(runs, lambda r: "{:+.1%}".format(r["stats"]["worst_year"])),
                 j(runs, lambda r: "{:+.1%}".format(r["since"]["cagr"]) if "since" in r else "-"),
                 j(runs, lambda r: "{:+.2%}".format(r["cand"]["top2"])), j(runs, lambda r: "{:.3f}".format(r["cand"]["ic"])),
                 j(runs, lambda r: "{:.0f}".format(r["stats"].get("per_year", float("nan"))))]
        L.append("| " + " | ".join([k] + cells) + " |")
    L.append("")
    for k, runs in arms.items():
        if k == base_key:
            continue
        p = paired(arms[base_key], runs)
        seeds = "・".join(f"種{s['seed']} {s['mean']:+.1%}（t={s['t']:+.2f}）" for s in p["per_seed"])
        L.append(f"- **{k} − {base_key}**: 年ごとの差 平均 {p['mean']:+.1%}/年（t={p['t']:+.2f}、プラスの年 {p['pos']:.0%}・{p['years']}年）。{seeds}")
    L.append("- 読み方: 口座の年率は種だけで ±3〜4ポイント動く（§80）。上位2の1回平均と順位相関が動いていなければ、年率の差は経路のぶれ")
    return "\n".join(L)


def save_result(cfg: dict, name: str, obj: dict, text: str) -> Path:
    d = cache_dir(cfg)

    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, pd.Series):
            return {str(k): float(v) for k, v in o.items()}
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        return o

    (d / f"result_{name}.json").write_text(json.dumps(clean(obj), ensure_ascii=False, indent=1, default=str))
    p = d / f"result_{name}.md"
    p.write_text(text)
    return p
