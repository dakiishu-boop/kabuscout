"""取り直せないデータのバックアップと復元（private の別リポジトリ dakiishu-boop/kabuscout-backup）。

data/ は GitHub Actions のキャッシュだけで引き継いでいる。キャッシュは7日間使われないと消え、公開リポジトリは60日間
動きがないと定時実行が止まる。株価やシグナル表は取り直せるが、次のものは二度と取れないので毎日ここに控える:
- kabuscout.db（ペーパートレード・通知した候補・モデルの記録）・scored/（その日の全候補の予測、forward.py）・scored_wide/（対象を広げたときの候補）
- flows/（JPX の信用残は数日分しか公開されない）・tdnet_xbrl/（TDnet は書類を約1ヶ月で消す）
- jquants/（無料版は過去2年分しか取れないので、貯めた分は取り直せない。J-Quants 由来なので公開の場所には置かない）
- strategy_params.json（研究ループが採用した設定と影の候補）、forward.json・backtest.json・research.json など小さい結果
保存先は private のリポジトリで、その日の分を「latest」と、曜日・週・月の枠のブランチに上書きで置く
（壊れた日の分が上書きしても、別の枠に前の日の分が残る。履歴は枠の数だけなので容量は増え続けない）。
復元は restore（manifest の指紋・行数を確かめてから data/ に戻す）。drill は一時フォルダに戻して確かめる（月1回の試し戻し）。
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import sqlite3
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)
ITEMS = ["kabuscout.db", "scored", "scored_wide", "flows", "tdnet_xbrl", "jquants", "strategy_params.json", "alerts.json", "core.json",
         "forward.json", "backtest.json", "research.json"]
BOT = ["-c", "user.name=kabuscout-bot", "-c", "user.email=41898282+github-actions[bot]@users.noreply.github.com"]


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _tables(db: Path) -> dict:
    con = sqlite3.connect(str(db))
    try:
        names = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
        return {n: int(con.execute(f'SELECT COUNT(*) FROM "{n}"').fetchone()[0]) for n in names}
    finally:
        con.close()


def snapshot(cfg: dict, out: Path) -> dict:
    """data/ の取り直せない部分を out に書き出し、manifest.json（各ファイルの指紋・DBの行数・最新日）を付ける。"""
    src = cfg["data"]["db_path"].parent
    out.mkdir(parents=True, exist_ok=True)
    for item in ITEMS:
        p = src / item
        if not p.exists():
            continue
        if item == "kabuscout.db":  # 書き込み中でも整合した写しを作る（WAL の分も含む）
            s, d = sqlite3.connect(str(p)), sqlite3.connect(str(out / item))
            with d:
                s.backup(d)
            d.execute("PRAGMA journal_mode=DELETE")  # 写しは1つのファイルで完結させる（-wal・-shm を作らない）
            s.close(); d.close()
        elif p.is_dir():
            shutil.copytree(p, out / item, dirs_exist_ok=True)
        else:
            shutil.copy2(p, out / item)
    for f in out.glob("kabuscout.db-*"):
        f.unlink()
    files = {str(f.relative_to(out)): dict(size=f.stat().st_size, sha256=_sha(f))
             for f in sorted(out.rglob("*")) if f.is_file() and ".git" not in f.parts and f.name not in ("manifest.json", "README.md")}
    man = dict(created_at=datetime.now().astimezone().isoformat(timespec="seconds"), run_id=os.environ.get("GITHUB_RUN_ID"),
               files=files, db=_tables(out / "kabuscout.db") if (out / "kabuscout.db").exists() else {})
    if (out / "kabuscout.db").exists():
        con = sqlite3.connect(str(out / "kabuscout.db"))
        man["latest"] = {k: con.execute(q).fetchone()[0] for k, q in (
            ("runs", "SELECT MAX(data_date) FROM runs WHERE status='ok'"), ("paper_equity", "SELECT MAX(date) FROM paper_equity"))}
        con.close()
    (out / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=1))
    return man


def push(cfg: dict, repo: Path, branches: list[str] | None = None) -> dict:
    """repo（バックアップ用リポジトリの checkout。push の認証は済んでいるもの）に今日の写しを1つのコミットとして置き、
    latest・daily-曜日・weekly-週番号の4で割った余り・monthly-月 のブランチに上書きで送る。"""
    git = lambda *a: subprocess.run(["git", "-C", str(repo), *BOT, *a], check=True, capture_output=True, text=True).stdout
    git("checkout", "-q", "--orphan", f"_snapshot_{datetime.now():%Y%m%d%H%M%S%f}")
    git("rm", "-rqf", "--ignore-unmatch", ".")
    for p in repo.iterdir():  # 追跡されていない残りも消す（.git 以外）
        if p.name != ".git":
            shutil.rmtree(p) if p.is_dir() else p.unlink()
    man = snapshot(cfg, repo)
    (repo / "README.md").write_text("KabuScout の取り直せないデータの控え（private）。復元の手順は本体の docs/RESTORE.md。\n")
    git("add", "-A")
    git("commit", "-qm", f"snapshot {man['created_at']} run {man.get('run_id')}")
    now = datetime.now()
    branches = branches or ["latest", f"daily-{now.strftime('%a').lower()}", f"weekly-{now.isocalendar()[1] % 4}", f"monthly-{now:%m}"]
    for b in branches:
        git("push", "-qf", "origin", f"HEAD:refs/heads/{b}")
    log.info("バックアップ: %d個のファイル → %s", len(man["files"]), ", ".join(branches))
    return dict(files=len(man["files"]), branches=branches, latest=man.get("latest"))


def verify(src: Path) -> dict:
    """写しが使えるか: 各ファイルの指紋、DBの壊れの検査と行数、parquet が読めるか。"""
    man = json.loads((src / "manifest.json").read_text())
    problems = []
    for rel, info in man["files"].items():
        f = src / rel
        if not f.exists():
            problems.append(f"無い: {rel}")
        elif _sha(f) != info["sha256"]:
            problems.append(f"指紋が違う: {rel}")
        elif f.suffix == ".parquet":
            try:
                pd.read_parquet(f)
            except Exception as e:
                problems.append(f"読めない: {rel}（{e}）")
    db = src / "kabuscout.db"
    if db.exists():
        con = sqlite3.connect(str(db))
        chk = con.execute("PRAGMA integrity_check").fetchone()[0]
        con.close()
        if chk != "ok":
            problems.append(f"DBの検査: {chk}")
        got = _tables(db)
        problems += [f"DBの行数が違う: {t} {n} → {got.get(t)}" for t, n in man.get("db", {}).items() if got.get(t) != n]
    elif "kabuscout.db" in man["files"]:
        problems.append("DBが無い")
    return dict(ok=not problems, problems=problems, created_at=man["created_at"], files=len(man["files"]),
                db=man.get("db", {}), latest=man.get("latest"))


def restore(cfg: dict, src: Path, force: bool = False) -> dict:
    """写しを確かめてから data/ に戻す。data/ にペーパートレードの記録があるときは force のときだけ上書きする。"""
    v = verify(src)
    if not v["ok"]:
        raise SystemExit("バックアップが壊れているので戻しません: " + "; ".join(v["problems"]))
    dst = cfg["data"]["db_path"].parent
    db = dst / "kabuscout.db"
    if db.exists() and not force and _tables(db).get("paper_positions", 0) > 0:
        raise SystemExit(f"{db} にペーパートレードの記録があります。上書きするなら --force")
    for item in ITEMS:
        p = src / item
        if not p.exists():
            continue
        if item == "kabuscout.db":
            for ext in ("-wal", "-shm"):
                (dst / f"kabuscout.db{ext}").unlink(missing_ok=True)
        if p.is_dir():
            shutil.copytree(p, dst / item, dirs_exist_ok=True)
        else:
            shutil.copy2(p, dst / item)
    log.info("バックアップ（%s）から %d個のファイルを戻しました", v["created_at"], v["files"])
    return v


def drill(src: Path) -> dict:
    """試し戻し: 一時フォルダに写して確かめ、戻した DB からペーパートレードと全候補の記録が読めるかまで見る。"""
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp) / "data"
        shutil.copytree(src, t, ignore=shutil.ignore_patterns(".git"))
        v = verify(t)
        if (t / "kabuscout.db").exists():
            con = sqlite3.connect(str(t / "kabuscout.db"))
            v["paper_closed"] = con.execute("SELECT COUNT(*) FROM paper_positions WHERE status='closed'").fetchone()[0]
            con.close()
        sc = sorted((t / "scored").glob("*.parquet")) if (t / "scored").exists() else []
        v["scored_files"] = len(sc)
        if sc:
            v["scored_rows_last"] = len(pd.read_parquet(sc[-1]))
    return v


def days_since_push(root: Path) -> float | None:
    try:
        ts = subprocess.run(["git", "-C", str(root), "log", "-1", "--format=%ct"], check=True, capture_output=True, text=True).stdout.strip()
        return (datetime.now().timestamp() - int(ts)) / 86400
    except Exception:
        return None


def remind_push(cfg: dict, warn_days: int = 45) -> float | None:
    """公開リポジトリは60日間 push が無いと定時実行が自動で止まる。45日を過ぎたら Discord で知らせる。"""
    from . import notify
    from .config import ROOT
    d = days_since_push(ROOT)
    if d is not None and d >= warn_days:
        notify.send_text(cfg, "⏰ KabuScout: push のお願い",
                         f"最後の push から {d:.0f}日たちました。60日を過ぎると GitHub が毎日の自動実行を止めます"
                         f"（止まって7日たつとキャッシュも消えます）。何か1つ commit して push してください。")
    return d
