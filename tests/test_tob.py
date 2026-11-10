import pandas as pd

from kabuscout import tdnet


def test_classify_buyout_targets():
    c = tdnet.classify
    assert "tob_target" in c("エリアリンク株式会社による当社株券等に対する公開買付けに関する賛同の意見表明及び応募推奨のお知らせ")
    assert "tob_target" in c("ＭＢＯの実施及び応募の推奨に関するお知らせ")
    assert "tob_target" in c("住友商事株式会社による当社株式に対する公開買付けの結果及び主要株主である筆頭株主の異動に関するお知らせ")
    assert "tob_target" in c("株式併合並びに単元株式数の定めの廃止及び定款の一部変更に係る承認決議に関するお知らせ")
    assert "tob_target" not in c("自己株式の取得及び自己株式の公開買付けに関するお知らせ")
    assert "tob_target" not in c("株式会社カカクコムによる自己株式の公開買付けへの応募に関するお知らせ")
    assert "tob_target" not in c("株式会社ＳＥＲＩＯホールディングス株券（証券コード：6567）に対する公開買付けの開始に関するお知らせ")
    assert "tob_target" not in c("（訂正）「当社株券等に対する公開買付けに関する意見表明のお知らせ」の一部訂正について")
    assert c("株式会社Ａによる当社株券等に対する公開買付けの結果（不成立）に関するお知らせ") == ["tob_end"]
    assert c("自己株式の取得に係る事項の決定に関するお知らせ") == ["buyback"]


def test_pending_buyouts_window_and_end():
    td = pd.DataFrame({"pubdate": pd.to_datetime(["2026-08-06 16:00", "2026-09-01 15:00", "2026-09-20 15:00", "2025-01-10 15:00"]),
                       "code": ["3676", "4320", "4320", "9999"], "title": "", "events": ["tob_target", "tob_target", "tob_end", "tob_target"]})
    p = tdnet.pending_buyouts(td, pd.Timestamp("2026-11-06"))
    assert set(p) == {"3676"}  # 4320 は不成立で終わり、9999 は1年より前
    assert tdnet.pending_buyouts(None, pd.Timestamp("2026-11-06")) == {}


def test_stored_titles_get_tob_target_after_rule_change(tmp_path):
    """別の分類で保存済みだった題名にも、買収の印が付く（取り直しでも、手元の分類し直しでも）。"""
    title = "株式会社Ａによる当社株式に対する公開買付けに関する意見表明及び第三者割当による新株式発行のお知らせ"
    assert tdnet.classify(title) == ["tob_target", "offering"]
    ts = pd.Timestamp("2026-03-24 11:30")
    old = pd.DataFrame({"pubdate": [ts, pd.Timestamp("2024-01-10 15:00")], "code": ["5202", "7088"], "title": [title, title],
                        "events": ["offering", "offering"]})
    assert tdnet._reclassify(old, pd.Timestamp("2025-10-01")) == 1  # 期間より前の行は触らない
    assert list(old["events"]) == ["tob_target,offering", "offering"]
    old["events"] = "offering"
    new = tdnet._save(tmp_path / "t.parquet", tmp_path / "t.json", old, [dict(pubdate=ts, code="5202", title=title, events="tob_target,offering")], set())
    assert len(new) == 2 and set(new[new["code"] == "5202"]["events"]) == {"tob_target,offering"}
