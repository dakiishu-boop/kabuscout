"""手元に貯めた TDnet の PDF（data/tdnet_pdf、tdnet_archive.py）を private のバックアップ用リポジトリに控える（ジャーナル §103）。

PDF は Mac の中にしか無く（毎日の実行は GitHub Actions なので、そこには無い）、TDnet は約1ヶ月で消すので、Mac が壊れると二度と取れない。
- 置き場所: dakiishu-boop/kabuscout-backup の「リリース」。月ごとに1つ（tdnet-pdf-YYYY-MM）。ブランチの控え（backup.py）には入れない
  （毎日まるごと1コミットにする方式なので、1ヶ月で約1.6GB 増える PDF は git に向かない）
- 毎回、まだ送っていない PDF だけを tar（PDF はすでに圧縮されているので圧縮しない）にまとめ、新しい名前で足す。前に送った分は上書きしない
- どの PDF をどの tar に入れたかは index.json（パス → tar の名前・sha256・大きさ）。手元と、各月のリリースの両方に置く
  （Mac が壊れたときはリリース側の index.json と tar だけで戻せる）
- 送る前に毎回、リポジトリが private かを確かめる（公開のところには置かない）
使い方: python -m kabuscout.pdf_backup [backup | restore [戻す先] | check]
"""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime
from pathlib import Path

log = logging.getLogger("kabuscout")
REPO = "dakiishu-boop/kabuscout-backup"
PREFIX = "tdnet-pdf-"
CHUNK = 400_000_000  # 1つの tar の上限（GitHub は1ファイル 2GB まで。失敗したときの送り直しを小さくする）


def _gh_bin() -> str:
    from .config import ROOT
    p = ROOT / ".tools" / "gh"
    return str(p) if p.exists() else "gh"


def gh(*args: str) -> str:
    return subprocess.run([_gh_bin(), *args], check=True, capture_output=True, text=True).stdout


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _root(cfg: dict) -> Path:
    return cfg["data"]["db_path"].parent / "tdnet_pdf"


def _index_path(cfg: dict) -> Path:
    return _root(cfg) / "backup_index.json"


def _check_private(run) -> None:
    if run("api", f"repos/{REPO}", "--jq", ".private").strip() != "true":
        raise SystemExit(f"{REPO} が private ではないので、PDF は送りません")


def _tags(run) -> list[str]:
    out = run("release", "list", "-R", REPO, "--limit", "1000", "--json", "tagName", "--jq", ".[].tagName")
    return sorted(t for t in out.split() if t.startswith(PREFIX))


def _assets(run, tag: str) -> list[str]:
    return run("release", "view", tag, "-R", REPO, "--json", "assets", "--jq", ".assets[].name").split()


def _remote_index(run, tag: str, tmp: Path) -> dict:
    if "index.json" not in _assets(run, tag):
        return {}
    d = tmp / tag
    run("release", "download", tag, "-R", REPO, "-p", "index.json", "-D", str(d), "--clobber")
    return json.loads((d / "index.json").read_text())


def load_index(cfg: dict, run=gh) -> dict:
    """手元の index.json。無いとき（Mac を新しくしたときなど）はリリース側から作り直す。"""
    p = _index_path(cfg)
    if p.exists():
        return json.loads(p.read_text())
    idx = {}
    with tempfile.TemporaryDirectory() as tmp:
        for tag in _tags(run):
            idx.update(_remote_index(run, tag, Path(tmp)))
    return idx


def _save_index(cfg: dict, idx: dict) -> None:
    p = _index_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".part")
    tmp.write_text(json.dumps(idx, ensure_ascii=False, indent=0, sort_keys=True))
    tmp.rename(p)


def _chunks(files: list[Path], limit: int) -> list[list[Path]]:
    out, cur, size = [], [], 0
    for f in files:
        s = f.stat().st_size
        if cur and size + s > limit:
            out.append(cur)
            cur, size = [], 0
        cur.append(f)
        size += s
    return out + ([cur] if cur else [])


def backup(cfg: dict, run=gh, limit: int = CHUNK, now: datetime | None = None) -> dict:
    """まだ送っていない PDF を月ごとの tar にまとめてリリースに足す。tar が送れたあとで index を書く。"""
    root = _root(cfg)
    if not root.exists():
        return dict(sent=0, mb=0.0, total=0, chunks=0)
    _check_private(run)
    idx = load_index(cfg, run)
    tags = set(_tags(run))
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    new = sorted(f for f in root.rglob("*.pdf") if str(f.relative_to(root)) not in idx)
    by_month: dict[str, list[Path]] = {}
    for f in new:
        rel = f.relative_to(root)
        by_month.setdefault(f"{rel.parts[0]}-{rel.parts[1]}", []).append(f)
    sent = chunks = size = 0
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for month, files in sorted(by_month.items()):
            tag = PREFIX + month
            if tag not in tags:
                run("release", "create", tag, "-R", REPO, "--target", "latest", "--title", f"TDnet PDF {month}",
                    "--notes", "KabuScout が手元に保存した TDnet の PDF の控え（private）。戻し方は本体の docs/RESTORE.md。")
                tags.add(tag)
            taken, k = set(_assets(run, tag)), 0
            for part in _chunks(files, limit):
                k += 1
                while f"pdf-{month}-{stamp}-{k:02d}.tar" in taken:  # 同じ時刻に2回動いても前の tar を上書きしない
                    k += 1
                name = f"pdf-{month}-{stamp}-{k:02d}.tar"
                tar = tmp / name
                entries = {}
                with tarfile.open(tar, "w") as t:
                    for f in part:
                        rel = str(f.relative_to(root))
                        t.add(f, arcname=rel)
                        entries[rel] = dict(tag=tag, asset=name, sha256=_sha(f), size=f.stat().st_size)
                run("release", "upload", tag, str(tar), "-R", REPO)
                tar.unlink()
                idx.update(entries)
                _save_index(cfg, idx)
                month_idx = {k2: v for k2, v in idx.items() if v["tag"] == tag}
                (tmp / "index.json").write_text(json.dumps(month_idx, ensure_ascii=False, indent=0, sort_keys=True))
                run("release", "upload", tag, str(tmp / "index.json"), "-R", REPO, "--clobber")
                sent += len(part)
                chunks += 1
                size += sum(v["size"] for v in entries.values())
                log.info("PDF の控え: %s に %d件（%.0f MB）", name, len(part), sum(v["size"] for v in entries.values()) / 1e6)
    log.info("PDF の控え: 新しく %d件（%.0f MB、tar %d個）・控え済みの合計 %d件", sent, size / 1e6, chunks, len(idx))
    return dict(sent=sent, mb=round(size / 1e6, 1), total=len(idx), chunks=chunks)


def _safe(name: str) -> bool:
    p = Path(name)
    return not p.is_absolute() and ".." not in p.parts and p.suffix == ".pdf"


def restore(cfg: dict, dest: Path | None = None, run=gh) -> dict:
    """リリースの tar を全部取ってきて dest（既定 data/tdnet_pdf）に戻し、index の sha256 と照らす。すでにある PDF は上書きしない。"""
    dest = Path(dest) if dest else _root(cfg)
    dest.mkdir(parents=True, exist_ok=True)
    restored = bad = 0
    idx = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for tag in _tags(run):
            d = tmp / tag
            run("release", "download", tag, "-R", REPO, "-D", str(d), "--clobber")
            idx.update(json.loads((d / "index.json").read_text()) if (d / "index.json").exists() else {})
            for tar in sorted(d.glob("*.tar")):
                with tarfile.open(tar) as t:
                    for m in t.getmembers():
                        if not (m.isfile() and _safe(m.name)):
                            continue
                        out = dest / m.name
                        if out.exists():
                            continue
                        out.parent.mkdir(parents=True, exist_ok=True)
                        with t.extractfile(m) as src, open(out.with_suffix(".part"), "wb") as w:
                            shutil.copyfileobj(src, w)
                        out.with_suffix(".part").rename(out)
                        restored += 1
                tar.unlink()
    for rel, v in idx.items():
        f = dest / rel
        if not f.exists() or _sha(f) != v["sha256"]:
            bad += 1
    log.info("PDF の戻し: %d件を戻しました・控えの合計 %d件・合わないもの %d件", restored, len(idx), bad)
    return dict(restored=restored, total=len(idx), bad=bad)


def check(cfg: dict, run=gh) -> dict:
    """手元の PDF がすべて控えてあるか（index にあり、tar がリリースに載っているか）。"""
    root = _root(cfg)
    local = {str(f.relative_to(root)) for f in root.rglob("*.pdf")} if root.exists() else set()
    idx = load_index(cfg, run)
    remote = {tag: set(_assets(run, tag)) for tag in _tags(run)}
    missing_asset = sum(1 for v in idx.values() if v["asset"] not in remote.get(v["tag"], set()))
    return dict(local=len(local), indexed=len(local & set(idx)), not_sent=len(local - set(idx)), missing_asset=missing_asset)


if __name__ == "__main__":
    from .config import load_config
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    a = sys.argv[1:] or ["backup"]
    c = load_config()
    if a[0] == "restore":
        print(restore(c, Path(a[1]) if len(a) > 1 else None))
    elif a[0] == "check":
        print(check(c))
    else:
        print(backup(c))
