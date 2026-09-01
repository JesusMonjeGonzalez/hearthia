import httpx
import respx

from hearthia.gateway import Gateway
from hearthia.shadow_eval import canary_check

BASE = "http://test-gw:9292"


@respx.mock
async def test_canary_check_ok_on_real_completion():
    respx.post(f"{BASE}/v1/chat/completions").respond(
        200, json={"choices": [{"message": {"content": "OK"}}]}
    )
    gw = Gateway(BASE)
    result = await canary_check(gw, "big-coder")
    assert result == {"ok": True, "text": "OK"}
    await gw.close()


@respx.mock
async def test_canary_check_fails_on_empty_completion():
    respx.post(f"{BASE}/v1/chat/completions").respond(
        200, json={"choices": [{"message": {"content": "   "}}]}
    )
    gw = Gateway(BASE)
    result = await canary_check(gw, "big-coder")
    assert result["ok"] is False
    assert "error" not in result
    await gw.close()


@respx.mock
async def test_canary_check_fails_on_no_choices():
    respx.post(f"{BASE}/v1/chat/completions").respond(200, json={"choices": []})
    gw = Gateway(BASE)
    result = await canary_check(gw, "big-coder")
    assert result["ok"] is False
    await gw.close()


@respx.mock
async def test_canary_check_reports_transport_error():
    respx.post(f"{BASE}/v1/chat/completions").mock(side_effect=httpx.ConnectError("down"))
    gw = Gateway(BASE)
    result = await canary_check(gw, "big-coder")
    assert result["ok"] is False
    assert "error" in result
    await gw.close()


@respx.mock
async def test_canary_check_reports_http_status_error():
    respx.post(f"{BASE}/v1/chat/completions").respond(500)
    gw = Gateway(BASE)
    result = await canary_check(gw, "big-coder")
    assert result["ok"] is False
    assert "error" in result
    await gw.close()
