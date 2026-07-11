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
async def test_running_returns_empty_on_non_json_body():
    respx.get(f"{BASE}/running").respond(200, text="<html>oops</html>")
    gw = Gateway(BASE)
    assert await gw.running() == []
    await gw.close()


@respx.mock
async def test_running_returns_empty_when_running_key_is_null():
    respx.get(f"{BASE}/running").respond(200, json={"running": None})
    gw = Gateway(BASE)
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
async def test_warm_url_encodes_model_id():
    respx.get(f"{BASE}/upstream/a%2Fb/health").respond(200)
    gw = Gateway(BASE)
    assert await gw.warm("a/b") is True
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


@respx.mock
async def test_metrics_returns_text():
    respx.get(f"{BASE}/metrics").respond(200, text="llamacpp:predicted_tokens_seconds 42.0\n")
    gw = Gateway(BASE)
    assert await gw.metrics() == "llamacpp:predicted_tokens_seconds 42.0\n"
    await gw.close()


@respx.mock
async def test_metrics_returns_empty_when_down():
    respx.get(f"{BASE}/metrics").mock(side_effect=httpx.ConnectError("down"))
    gw = Gateway(BASE)
    assert await gw.metrics() == ""
    await gw.close()


@respx.mock
async def test_metrics_returns_empty_on_non_200():
    respx.get(f"{BASE}/metrics").respond(500)
    gw = Gateway(BASE)
    assert await gw.metrics() == ""
    await gw.close()


@respx.mock
async def test_events_yields_parsed_json():
    sse = 'data: {"type":"modelStatus","data":[{"model":"big-coder","state":"ready"}]}\n\n'
    route = respx.get(f"{BASE}/api/events").respond(200, text=sse)
    gw = Gateway(BASE)
    events = []
    async for evt in gw.events():
        events.append(evt)
        break
    assert route.called
    assert events[0]["type"] == "modelStatus"
    assert events[0]["data"][0]["model"] == "big-coder"
    await gw.close()


@respx.mock
async def test_events_skips_non_data_lines():
    sse = ': comment\nevent: ping\ndata: {"type":"logData","data":{"data":"hi"}}\n\n'
    respx.get(f"{BASE}/api/events").respond(200, text=sse)
    gw = Gateway(BASE)
    events = []
    async for evt in gw.events():
        events.append(evt)
        break
    assert events[0]["type"] == "logData"
    await gw.close()


@respx.mock
async def test_events_skips_unparseable_json():
    sse = 'data: {bad json}\ndata: {"type":"logData","data":{}}\n\n'
    respx.get(f"{BASE}/api/events").respond(200, text=sse)
    gw = Gateway(BASE)
    events = []
    async for evt in gw.events():
        events.append(evt)
        break
    assert events[0]["type"] == "logData"
    await gw.close()


@respx.mock
async def test_logs_stream_yields_bytes():
    respx.get(f"{BASE}/logs/stream").respond(200, text="line1\nline2\n")
    gw = Gateway(BASE)
    chunks = []
    async for chunk in gw.logs_stream():
        chunks.append(chunk)
    assert b"".join(chunks) == b"line1\nline2\n"
    await gw.close()


@respx.mock
async def test_chat_stream_yields_bytes():
    route = respx.post(f"{BASE}/v1/chat/completions").respond(
        200, text="data: chunk1\n\ndata: chunk2\n\n"
    )
    gw = Gateway(BASE)
    chunks = []
    async for chunk in gw.chat_stream(b'{"model":"big-coder"}'):
        chunks.append(chunk)
    assert route.called
    assert b"".join(chunks) == b"data: chunk1\n\ndata: chunk2\n\n"
    await gw.close()
