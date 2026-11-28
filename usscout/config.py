"""設定の読み込み（usscout.yaml）。日本株版と同じく ${ENV} を環境変数・.env で置き換える。"""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load(path: Path | None = None) -> dict:
    from kabuscout.config import _expand, _load_dotenv
    _load_dotenv()
    path = Path(path) if path else ROOT / "usscout.yaml"
    cfg = _expand(yaml.safe_load(path.read_text()))
    d = Path(cfg["data"]["dir"])
    cfg["data"]["dir"] = d if d.is_absolute() else ROOT / d
    return cfg
