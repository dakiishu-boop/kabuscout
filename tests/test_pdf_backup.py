"""手元の PDF を private のリポジトリのリリースに控える仕組み（ジャーナル §103）。ネットにはつながず、gh の代わりの作り物で確かめる。"""
import shutil
from pathlib import Path

import pytest

from kabuscout import pdf_backup as pb


class FakeGh:
    """gh の release / api の一部だけを、一時フォルダの中で真似する。"""

    def __init__(self, base: Path, private=True):
        self.base, self.private, self.uploads = base, private, []

    def __call__(self, *a):
        a = list(a)
        if a[0] == "api":
            return "true\n" if self.private else "false\n"
        sub, rest = a[1], a[2:]
        if sub == "list":
            return "\n".join(p.name for p in sorted(self.base.iterdir())) + "\n"
        tag = rest[0]
        d = self.base / tag
        if sub == "create":
            d.mkdir()
        elif sub == "view":
            return "\n".join(p.name for p in sorted(d.iterdir())) + "\n"
        elif sub == "upload":
            src = Path(rest[1])
            if (d / src.name).exists() and "--clobber" not in rest:
                raise RuntimeError("already exists")
            shutil.copy(src, d / src.name)
            self.uploads.append(src.name)
        elif sub == "download":
            out = Path(rest[rest.index("-D") + 1])
            out.mkdir(parents=True, exist_ok=True)
            pat = rest[rest.index("-p") + 1] if "-p" in rest else None
            for p in d.iterdir():
                if pat is None or p.name == pat:
                    shutil.copy(p, out / p.name)
        return ""


def _pdf(cfg, rel, body):
    p = cfg["data"]["db_path"].parent / "tdnet_pdf" / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"%PDF-1.7 " + body)
    return p


def test_backup_sends_each_pdf_once_and_restores(cfg, tmp_path):
    remote = tmp_path / "remote"
    remote.mkdir()
    g = FakeGh(remote)
    _pdf(cfg, "2026/10/a.pdf", b"a" * 50)
    _pdf(cfg, "2026/11/b.pdf", b"b" * 50)
    _pdf(cfg, "2026/11/c.pdf", b"c" * 50)
    got = pb.backup(cfg, run=g, limit=100)
    assert (got["sent"], got["total"]) == (3, 3)
    assert sorted(p.name for p in remote.iterdir()) == ["tdnet-pdf-2026-10", "tdnet-pdf-2026-11"]
    assert len(list((remote / "tdnet-pdf-2026-11").glob("*.tar"))) == 2  # 上限を超える分は別の tar に分ける
    assert pb.backup(cfg, run=g)["sent"] == 0  # 2回目は何も送らない
    _pdf(cfg, "2026/11/d.pdf", b"d")
    assert pb.backup(cfg, run=g)["sent"] == 1  # 増えた分だけ
    assert pb.check(cfg, run=g) == dict(local=4, indexed=4, not_sent=0, missing_asset=0)

    # Mac が壊れた想定: 手元の index も PDF も無いところへ、リリースだけから戻す
    out = tmp_path / "back"
    r = pb.restore(cfg, dest=out, run=g)
    assert (r["restored"], r["total"], r["bad"]) == (4, 4, 0)
    assert (out / "2026/11/d.pdf").read_bytes() == b"%PDF-1.7 d"
    shutil.rmtree(cfg["data"]["db_path"].parent / "tdnet_pdf" / "2026")
    (cfg["data"]["db_path"].parent / "tdnet_pdf" / "backup_index.json").unlink()
    assert pb.load_index(cfg, run=g).keys() == {"2026/10/a.pdf", "2026/11/b.pdf", "2026/11/c.pdf", "2026/11/d.pdf"}


def test_backup_refuses_a_public_repo(cfg, tmp_path):
    remote = tmp_path / "remote"
    remote.mkdir()
    g = FakeGh(remote, private=False)
    _pdf(cfg, "2026/11/a.pdf", b"a")
    with pytest.raises(SystemExit):
        pb.backup(cfg, run=g)
    assert g.uploads == []
