import httpx
import respx

from hearthia.gateway import Gateway

BASE = "http://test-gw:9292"


@respx.mock
async def test_is_up_true_and_false():
    respx.get(f"{BASE}/health").respond(200)
    gw = Gateway(BASE)
    assert await gw.is_up() is True

    respx.get(f"{BASE}/health").mock(side_effect=httpx.ConnectError("down"))
    assert await gw.is_up() is False
    await gw.close()


@respx.mock
async def test_running_returns_list_and_empty_on_error():
    respx.get(f"{BASE}/running").respond(
        200, json={"running": [{"model": "big-coder", "state": "ready"}]}
    )
    gw = Gateway(BASE)
    assert await gw.running() == [{"model": "big-coder", "state": "ready"}]

    respx.get(f"{BASE}/running").mock(side_effect=httpx.ConnectError("down"))
    assert await gw.running() == []
    await gw.close()


@respx.mock
async def test_warm_hits_upstream_health():
    route = respx.get(f"{BASE}/upstream/big-coder/health").respond(200)
    gw = Gateway(BASE)
    assert await gw.warm("big-coder") is True
    assert route.called
    await gw.close()


@respx.mock
async def test_cool_one_and_all():
    one = respx.post(f"{BASE}/api/models/unload/big-coder").respond(200)
    every = respx.post(f"{BASE}/api/models/unload").respond(200)
    gw = Gateway(BASE)
    assert await gw.cool("big-coder") is True
    assert await gw.cool() is True
    assert one.called and every.called
    await gw.close()


@respx.mock
async def test_warm_returns_false_when_gateway_down():
    respx.get(f"{BASE}/upstream/big-coder/health").mock(side_effect=httpx.ConnectError("down"))
    gw = Gateway(BASE)
    assert await gw.warm("big-coder") is False
    await gw.close()


@respx.mock
async def test_cool_returns_false_when_gateway_down():
    respx.post(f"{BASE}/api/models/unload/big-coder").mock(side_effect=httpx.ConnectError("down"))
    respx.post(f"{BASE}/api/models/unload").mock(side_effect=httpx.ConnectError("down"))
    gw = Gateway(BASE)
    assert await gw.cool("big-coder") is False
    assert await gw.cool() is False
    await gw.close()
