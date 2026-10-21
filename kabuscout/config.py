"""設定の読み込み。config.yaml + 環境変数（.env も可）。"""
from __future__ import annotations

import os
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _expand(obj):
    if isinstance(obj, dict):
        return {k: _expand(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand(v) for v in obj]
    if isinstance(obj, str):
        return re.sub(r"\$\{(\w+)\}", lambda m: os.environ.get(m.group(1), ""), obj)
    return obj


def load_config(path: Path | None = None) -> dict:
    _load_dotenv()
    path = path or ROOT / "config.yaml"
    cfg = _expand(yaml.safe_load(path.read_text()))
    for key in ("db_path", "model_dir", "report_dir", "price_dir", "pool_path"):
        p = Path(cfg["data"][key])
        cfg["data"][key] = p if p.is_absolute() else ROOT / p
    cfg["data"]["model_dir"].mkdir(parents=True, exist_ok=True)
    cfg["data"]["report_dir"].mkdir(parents=True, exist_ok=True)
    cfg["data"]["db_path"].parent.mkdir(parents=True, exist_ok=True)
    return cfg
