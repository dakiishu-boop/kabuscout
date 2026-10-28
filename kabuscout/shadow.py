"""影の候補: 本番の判断には使わず、本番と同じ日の全候補に点を付けて記録するだけのAI（前向きの比較）。

過去のデータでは決着しなかった候補（方向は良いが t<2.5）を、結果が出る前に shadow.yaml に登録する（git の記録が事前登録の証拠）。
同じ日の同じ候補で本番と比べるので、相場の運の差は打ち消し合う。判定は forward.evaluate（上位10%の上乗せの差）。
- 事前登録した候補（source=registered）: 判定日に shadow.yaml の基準で合否を出すだけ。本番への昇格は人が決める
- 研究ループが採用したAIの変更（source=research）: いきなり本番に入れず、ここで一定期間走らせ、
  判定日に「本番よりはっきり悪くない」なら本番に昇格する（基準は shadow.yaml の research_rule）
"""
from __future__ import annotations

import copy
import json
import logging
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import yaml

from .config import ROOT

log = logging.getLogger(__name__)
SHADOW_FILE = ROOT / "shadow.yaml"
KINDS = ("model_params", "extra_features", "specialists")  # 影で試せるのは、同じ候補への点の付け方を変えるものだけ


def _registry() -> dict:
    return yaml.safe_load(SHADOW_FILE.read_text()) if SHADOW_FILE.exists() else {}


def _saved(cfg: dict) -> dict:
    from .pipeline import params_path
    p = params_path(cfg)
    return json.loads(p.read_text()) if p.exists() else {}


def variants(cfg: dict) -> list[dict]:
    """登録されている影の候補（事前登録 ＋ 研究ループが採用したもの）。"""
    reg = _registry()
    out = []
    for v in reg.get("variants") or []:
        out.append(dict(key=v["key"], label=v["label"], source="registered", start=str(reg["start"]), judge=str(reg["judge"]["date"]),
                        rule={k: (str(x) if hasattr(x, "isoformat") else x) for k, x in reg["judge"].items()},
                        changes={k: v[k] for k in KINDS if k in v}))
    rr = reg.get("research_rule") or {}
    for v in _saved(cfg).get("shadow") or []:
        out.append(dict(key=v["key"], label=v["label"], source="research", start=v["start"], judge=v["judge"],
                        rule=dict(rr, kind="not_worse"), changes=v["changes"]))
    return out


def cfg_for(cfg: dict, v: dict) -> dict:
    c = copy.deepcopy(cfg)
    ch = v["changes"]
    if "model_params" in ch:
        c["learning"]["model_params"] = ch["model_params"]
    if "extra_features" in ch:
        c["learning"]["extra_features"] = list(ch["extra_features"])
    if "specialists" in ch:
        c["learning"]["specialists"] = list(ch["specialists"])
    return c


def _path(cfg: dict, key: str):
    return cfg["data"]["model_dir"] / f"shadow_{key}.joblib"


def _load(cfg: dict, key: str) -> dict | None:
    try:
        return joblib.load(_path(cfg, key)) if _path(cfg, key).exists() else None
    except Exception as e:
        log.warning("影のモデル %s の読み込みに失敗: %s", key, e)
        return None


def _stale(cfg: dict, v: dict, b: dict | None, dates: pd.Series) -> bool:
    """学習し直しが要るか: 無い・定義が変わった・本番と同じ間隔（retrain_every_days）を過ぎた。"""
    spec = json.dumps(v["changes"], sort_keys=True)
    return b is None or b.get("spec") != spec or int((dates > pd.Timestamp(b["data_end"])).sum()) >= cfg["learning"]["retrain_every_days"]


def due(cfg: dict, dates: pd.Series) -> list[str]:
    """学習し直しが要る影の候補（毎日の実行の最後に確かめ、要るときだけ別の仕事で学習する。Webの公開を遅らせないため）。"""
    return [v["key"] for v in variants(cfg) if _stale(cfg, v, _load(cfg, v["key"]), dates)]


def ensure(cfg: dict, ds: pd.DataFrame, data_end: str, train: bool = True) -> dict:
    """影のモデルを読み込み、無い・定義が変わった・本番と同じ間隔（retrain_every_days）を過ぎたら学習し直す。
    学習は本番の最終モデルと同じ作り方（全期間・同じ種の数・新しいデータほど重く）。失敗しても本番は止めない。
    train=False: 読み込むだけ（毎日の採点用。学習は shadow-train の別の仕事で行い、通知と Web の公開を遅らせない）。"""
    from . import learner
    vs = variants(cfg)
    if not vs:
        return {}
    out, todo = {}, []
    dates = ds["date"].drop_duplicates()
    for v in vs:
        spec = json.dumps(v["changes"], sort_keys=True)
        b = _load(cfg, v["key"])
        if b is not None and b.get("spec") == spec and (not train or not _stale(cfg, v, b, dates)):
            out[v["key"]] = b
        elif train:
            todo.append((v, spec))
    if todo:
        d = learner.with_plan_outcomes(ds[ds["complete"]], cfg)
        d = d[d["complete_plan"]].sort_values("date").reset_index(drop=True)
        seeds = int(cfg["learning"].get("seeds", 1))
        for v, spec in todo:
            try:
                c = cfg_for(cfg, v)
                log.info("影のモデルを学習中: %s", v["label"])
                models = learner.fit_models(d, c, seeds=seeds)
                b = dict(models=models, features=learner.model_features(c), spec=spec, data_end=data_end,
                         version=f"{v['key']}-{datetime.now().strftime('%Y%m%d-%H%M')}")
                joblib.dump(b, _path(cfg, v["key"]))
                out[v["key"]] = b
            except Exception as e:
                log.warning("影のモデル %s の学習に失敗（本番は続行）: %s", v["key"], e)
    return out


def score(today: pd.DataFrame, bundles: dict) -> dict:
    """{key: (AIの点, モデルの版)}。点は本番の ai_score と同じ「3つのモデルの、その日の候補内での順位の平均」。"""
    from . import learner
    out = {}
    for key, b in bundles.items():
        try:
            _, sc = learner.bundle_predict(b, today)
            out[key] = (np.asarray(sc, dtype=float), b["version"])
        except Exception as e:
            log.warning("影のモデル %s の採点に失敗: %s", key, e)
    return out


def verdict(v: dict, r: dict) -> str:
    """判定日までは記録中。判定日の後は shadow.yaml の基準で合否。"""
    today = pd.Timestamp.today().normalize()
    if today < pd.Timestamp(v["judge"]):
        return "記録中"
    rule = v["rule"]
    if (r.get("weeks") or 0) < int(rule.get("min_weeks", 8)):
        return "判定日を過ぎたが結果の出た週が足りない"
    t = r.get("uplift_t")  # 上位10%の上乗せの差（影 − 本番）の t
    if v["source"] == "registered":
        ok = t is not None and t >= float(rule.get("t_min", 2.0))
        return "合格（本番への昇格は人が判断）" if ok else "不合格"
    return "昇格" if t is None or t > float(rule.get("t_floor", -1.0)) else "見送り（本番よりはっきり悪い）"


def register_from_research(cfg: dict, kind: str, value, label: str, reason: str) -> dict:
    """研究ループが採用したAIの変更を、本番ではなく影に登録する。判定日は research_rule の週数の後。"""
    from .pipeline import params_path
    rr = _registry().get("research_rule") or {}
    start = (pd.Timestamp.today().normalize() + pd.offsets.BDay(1))
    v = dict(key=f"r{start:%Y%m%d}_{kind}", label=label, start=str(start.date()),
             judge=str((start + pd.Timedelta(weeks=int(rr.get("weeks", 12)))).date()), changes={kind: value}, reason=reason)
    saved = _saved(cfg)
    saved["shadow"] = [x for x in saved.get("shadow") or [] if x["key"] != v["key"]] + [v]
    saved.setdefault("history", []).append(dict(at=datetime.now().isoformat(timespec="seconds"), shadow={label: value}))
    params_path(cfg).write_text(json.dumps(saved, ensure_ascii=False, indent=1))
    return v


def in_shadow(cfg: dict, kind: str) -> list:
    """影で走らせている変更（研究ループが同じ候補を二重に試さないため）。"""
    return [v["changes"][kind] for v in variants(cfg) if kind in v["changes"]]


def settle(cfg: dict, fw: dict) -> list[str]:
    """判定日を過ぎた研究ループ由来の影を、基準どおり本番に昇格するか、外す。事前登録の候補は自動では昇格しない。"""
    from .pipeline import params_path
    from .research import _save_param
    done = []
    res = {r["key"]: r for r in fw.get("shadows") or []}
    for v in variants(cfg):
        if v["source"] != "research" or v["key"] not in res:
            continue
        verd = res[v["key"]]["verdict"]
        if verd not in ("昇格", "見送り（本番よりはっきり悪い）"):
            continue
        saved = _saved(cfg)
        saved["shadow"] = [x for x in saved.get("shadow") or [] if x["key"] != v["key"]]
        saved.setdefault("history", []).append(dict(at=datetime.now().isoformat(timespec="seconds"), shadow_result={v["label"]: verd}))
        params_path(cfg).write_text(json.dumps(saved, ensure_ascii=False, indent=1))
        if verd == "昇格":
            for kind, value in v["changes"].items():
                _save_param(cfg, kind, value, v["label"])
        done.append(f"{v['label']}: {verd}")
        log.info("影の判定: %s → %s", v["label"], verd)
    return done
