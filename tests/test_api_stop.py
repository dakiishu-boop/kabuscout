"""Claude API のキーが無効・残高切れのときは、適時開示のサイトに取りに行かずにすぐ止める（ジャーナル 79）。"""
import pytest

anthropic = pytest.importorskip("anthropic")
httpx = pytest.importorskip("httpx")

from kabuscout import tdnet_quals, tdnet_reasons


def _err(cls, status, msg):
    return cls(msg, response=httpx.Response(status, request=httpx.Request("POST", "https://api.anthropic.com/v1/messages")), body=None)


def test_api_stop_reason():
    assert "無効" in tdnet_reasons.api_stop_reason(_err(anthropic.AuthenticationError, 401, "invalid x-api-key"))
    assert "残高" in tdnet_reasons.api_stop_reason(_err(anthropic.BadRequestError, 400, "Your credit balance is too low"))
    assert tdnet_reasons.api_stop_reason(_err(anthropic.BadRequestError, 400, "pdf too large")) is None
    assert tdnet_reasons.api_stop_reason(RuntimeError("stop_reason=max_tokens")) is None


@pytest.mark.parametrize("mod", [tdnet_reasons, tdnet_quals])
def test_update_stops_before_fetching_when_key_is_invalid(mod, tmp_path, monkeypatch):
    class Models:
        def list(self, **_):
            raise _err(anthropic.AuthenticationError, 401, "invalid x-api-key")

    class Client:
        models = Models()

    def no_network(*a, **k):
        raise AssertionError("キーが無効なのに適時開示のサイトに取りに行った")

    monkeypatch.setattr(mod, "_client", lambda: Client())
    monkeypatch.setattr(mod.requests, "get", no_network)
    info = mod.update({"data": {"db_path": tmp_path / "x.db"}})
    assert "無効" in info["stopped"] and info["targets"] == 0
