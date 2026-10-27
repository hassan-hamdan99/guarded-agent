from types import SimpleNamespace

from guarded_agent.llm import ClaudeLLM
from guarded_agent.tools import TOOL_SPECS


class Block:
    def __init__(self, **data):
        self.data = data

    def model_dump(self, **_):
        return dict(self.data)


class FakeClient:
    def __init__(self, response):
        self.calls = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))
        self.response = response

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def test_request_shape_and_block_replay():
    thinking = Block(type="thinking", thinking="", signature="sig")
    call = Block(type="tool_use", id="toolu_1", name="get_booking", input={"booking_id": "BK-1001"})
    client = FakeClient(SimpleNamespace(stop_reason="tool_use", content=[thinking, call]))

    turn = ClaudeLLM(client=client).complete("sys", TOOL_SPECS, [{"role": "user", "content": "hi"}])

    sent = client.calls[0]
    assert sent["model"] == "claude-opus-5-5"
    assert sent["output_config"] == {"effort": "medium"}
    assert sent["fallbacks"] == "default" and sent["betas"] == ["server-side-fallback-2026-07-01"]
    assert "tool_choice" not in sent  # forced tool choice is rejected on this model
    assert all(t["strict"] for t in sent["tools"])
    # thinking blocks must be passed back unchanged on the next request
    assert turn.content[0] == {"type": "thinking", "thinking": "", "signature": "sig"}
    assert turn.stop_reason == "tool_use"
