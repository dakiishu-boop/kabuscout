"""テストの共通部品。ネットにつながらず、実データ・APIキーも使わない。
- cfg: 本物の config.yaml を読み、データの置き場所だけ一時フォルダに差し替えた設定（config.yaml は書き換えない。
  手元の data/strategy_params.json も読まないので、どこで動かしても同じ設定になる）
- panel: tests/fixtures/prices.csv（作り物の株価）を「日付×銘柄」の表にしたもの
- FakeStore: PriceStore の代わり（backtest が終値を読むところだけ）"""
from pathlib import Path

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
FIX = Path(__file__).parent / "fixtures"


def make_cfg(base: Path) -> dict:
    from kabuscout.config import load_config
    raw = yaml.safe_load((ROOT / "config.yaml").read_text())
    d = base / "data"
    raw["data"].update(db_path=str(d / "kabuscout.db"), price_dir=str(d / "prices"), pool_path=str(d / "pool.parquet"),
                       model_dir=str(d / "models"), report_dir=str(base / "reports"))
    base.mkdir(parents=True, exist_ok=True)
    p = base / "config.yaml"
    p.write_text(yaml.safe_dump(raw, allow_unicode=True))
    return load_config(p)


@pytest.fixture
def cfg(tmp_path):
    return make_cfg(tmp_path)


@pytest.fixture(scope="session")
def panel():
    df = pd.read_csv(FIX / "prices.csv", dtype={"code": str}, parse_dates=["date"])
    return {f: df.pivot(index="date", columns="code", values=f).astype("float64") for f in ("open", "high", "low", "close", "volume")}


class FakeStore:
    """backtest._closes が使う load(fields=["close"], codes=[...]) だけを持つ。"""

    def __init__(self, close: pd.DataFrame):
        self.close = close

    def load(self, start=None, end=None, codes=None, fields=None):
        return {"close": self.close[list(codes)] if codes is not None else self.close}


@pytest.fixture(autouse=True)
def _fresh_close_cache():
    """backtest は終値の表を使い回すので、テストごとに空にする（別のテストの表が混ざらないように）。"""
    from kabuscout import backtest
    backtest._CLOSE.clear()
    yield
    backtest._CLOSE.clear()
