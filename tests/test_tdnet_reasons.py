import json
from types import SimpleNamespace

from kabuscout import tdnet_reasons as R


def test_targets_revision_notices_only():
    assert R.is_target("業績予想の修正に関するお知らせ")
    assert R.is_target("2027年３月期第２四半期（中間期）連結業績予想と実績値との差異に関するお知らせ")
    assert R.is_target("業績予想値と決算値との差異および通期連結業績予想の修正に関するお知らせ")
    assert not R.is_target("2027年3月期 第2四半期（中間期）決算短信〔日本基準〕（連結）")
    assert not R.is_target("（訂正）業績予想の修正に関するお知らせ")
    assert not R.is_target("自己株式の取得に係る事項の決定に関するお知らせ")


def test_schema_requires_every_field():
    assert set(R.SCHEMA["required"]) == set(R.SCHEMA["properties"])
    assert set(R.DRIVER_JA) == set(R.DRIVERS)


def test_classify_pdf_parses_structured_output():
    out = {k: None for k in R.SCHEMA["properties"]}
    out.update(main_driver="demand", fy_direction="up")
    calls = {}

    def create(**kw):
        calls.update(kw)
        return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text=json.dumps(out))],
                               usage=SimpleNamespace(input_tokens=10, output_tokens=5))

    client = SimpleNamespace(messages=SimpleNamespace(create=create))
    res, use = R.classify_pdf(client, b"%PDF-1.4 x", "m")
    assert res["main_driver"] == "demand" and use == dict(in_tok=10, out_tok=5)
    assert calls["messages"][0]["content"][0]["type"] == "document"
    assert calls["output_config"]["format"]["schema"] is R.SCHEMA


def test_xbrl_parse_reads_revision_notice_files():
    import io
    import zipfile
    from kabuscout import tdnet_xbrl as X
    fact = ('<ix:nonFraction name="tse-ed-t:OperatingIncome" contextRef="CurrentYearDuration_ConsolidatedMember_PreviousMember_ForecastMember" '
            'scale="6" decimals="-6">1,234</ix:nonFraction>')
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("tse-rvfc-12340-20261106000000-ixbrl.htm", f"<html>{fact}</html>")
        z.writestr("XBRLData/Attachment/0101010-acbs01-tse-acedjpfr-12340-ixbrl.htm", f"<html>{fact}</html>")  # 本文の財務諸表は読まない
    f = X.parse(buf.getvalue())
    assert len(f) == 1 and f["element"].iloc[0] == "OperatingIncome" and f["value"].iloc[0] == 1234e6


def test_quals_targets_and_page_cut():
    import io
    from pypdf import PdfReader, PdfWriter
    from kabuscout import tdnet_quals as Q
    assert Q.is_target("2027年3月期 第2四半期（中間期）決算短信〔日本基準〕（連結）")
    assert not Q.is_target("（訂正）2027年3月期 第2四半期決算短信〔日本基準〕（連結）の一部訂正について")
    assert not Q.is_target("業績予想の修正に関するお知らせ")
    assert set(Q.SCHEMA["required"]) == set(Q.SCHEMA["properties"])
    w = PdfWriter()
    for _ in range(12):
        w.add_blank_page(width=100, height=100)
    buf = io.BytesIO()
    w.write(buf)
    assert len(PdfReader(io.BytesIO(Q.first_pages(buf.getvalue()))).pages) == Q.PAGES
