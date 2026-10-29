"""バックアップ（backup.py）: 写して戻すと中身が一致する。1バイト壊すと検出して戻さない。
ペーパートレードの記録がある場所には、force なしでは上書きしない。"""
import hashlib
import json
import sqlite3

import pandas as pd
import pytest

from kabuscout import backup
from kabuscout.db import connect
from tests.conftest import make_cfg


def _sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _fill(cfg):
    """取り直せないデータの作り物: DB（ペーパートレード・実行の記録）・全候補の記録・需給・小さな結果のファイル。"""
    d = cfg["data"]["db_path"].parent
    con = connect(cfg["data"]["db_path"])
    con.execute("INSERT INTO runs VALUES ('2026-10-29','t',NULL,'ok',NULL)")
    con.execute("INSERT INTO paper_positions(code,name,signal_date,status,entry,shares,pnl) VALUES ('1001','A','2026-10-28','closed',1000,100,1234.5)")
    con.execute("INSERT INTO paper_equity VALUES ('2026-10-29', 1001234.5, 1000000)")
    con.commit()
    con.close()
    (d / "scored").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"date": ["2026-10-29"] * 3, "code": ["1", "2", "3"], "ai_score": [0.1, 0.5, 0.9]}).to_parquet(d / "scored" / "2026-10-29_x.parquet")
    (d / "flows").mkdir(exist_ok=True)
    pd.DataFrame({"code": ["1001"], "rate": [2.5]}).to_parquet(d / "flows" / "lending.parquet")
    (d / "strategy_params.json").write_text(json.dumps({"params": {}, "shadow": []}))
    return d


def _rows(db):
    con = sqlite3.connect(str(db))
    try:
        return {t: con.execute(f"SELECT * FROM {t} ORDER BY 1").fetchall() for t in ("runs", "paper_positions", "paper_equity")}
    finally:
        con.close()


def test_round_trip_restores_identical_data(cfg, tmp_path):
    src = _fill(cfg)
    snap = tmp_path / "snap"
    man = backup.snapshot(cfg, snap)
    assert set(man["files"]) >= {"kabuscout.db", "scored/2026-10-29_x.parquet", "flows/lending.parquet", "strategy_params.json"}
    assert man["db"]["paper_positions"] == 1 and man["latest"]["runs"] == "2026-10-29"
    assert backup.verify(snap)["ok"]
    dr = backup.drill(snap)
    assert dr["ok"] and dr["paper_closed"] == 1 and dr["scored_files"] == 1 and dr["scored_rows_last"] == 3

    cfg2 = make_cfg(tmp_path / "other")                       # 空の別の場所に戻す
    backup.restore(cfg2, snap)
    dst = cfg2["data"]["db_path"].parent
    for rel in man["files"]:
        if rel != "kabuscout.db":                             # DB は写しの方式（SQLite のバックアップ）でファイルの並びが変わりうる
            assert _sha(dst / rel) == _sha(src / rel), rel
    assert _rows(dst / "kabuscout.db") == _rows(src / "kabuscout.db")


def test_one_corrupt_byte_is_detected_and_not_restored(cfg, tmp_path):
    _fill(cfg)
    snap = tmp_path / "snap"
    backup.snapshot(cfg, snap)
    f = snap / "scored" / "2026-10-29_x.parquet"
    b = bytearray(f.read_bytes())
    b[len(b) // 2] ^= 0xFF
    f.write_bytes(bytes(b))
    v = backup.verify(snap)
    assert not v["ok"] and any("scored/2026-10-29_x.parquet" in p for p in v["problems"])
    cfg2 = make_cfg(tmp_path / "other")
    with pytest.raises(SystemExit):
        backup.restore(cfg2, snap)
    assert not (cfg2["data"]["db_path"].parent / "scored").exists()   # 何も戻していない
    assert not cfg2["data"]["db_path"].exists()


def test_corrupt_db_is_detected(cfg, tmp_path):
    _fill(cfg)
    snap = tmp_path / "snap"
    backup.snapshot(cfg, snap)
    db = snap / "kabuscout.db"
    b = bytearray(db.read_bytes())
    b[-100] ^= 0xFF
    db.write_bytes(bytes(b))
    assert not backup.verify(snap)["ok"]


def test_does_not_overwrite_paper_records_without_force(cfg, tmp_path):
    _fill(cfg)
    snap = tmp_path / "snap"
    backup.snapshot(cfg, snap)
    cfg2 = make_cfg(tmp_path / "other")
    _fill(cfg2)                                               # 戻し先にもペーパートレードの記録がある
    con = sqlite3.connect(str(cfg2["data"]["db_path"]))
    con.execute("UPDATE paper_positions SET pnl=-1")
    con.commit()
    con.close()
    with pytest.raises(SystemExit):
        backup.restore(cfg2, snap)
    con = sqlite3.connect(str(cfg2["data"]["db_path"]))
    assert con.execute("SELECT pnl FROM paper_positions").fetchall() == [(-1.0,)]   # 戻し先の記録はそのまま
    con.close()
    backup.restore(cfg2, snap, force=True)
    assert _rows(cfg2["data"]["db_path"]) == _rows(cfg["data"]["db_path"])
