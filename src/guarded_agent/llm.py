"""Model access. The agent only needs `complete()`, so tests can swap in a scripted model."""
from dataclasses import dataclass
from typing import Protocol


@dataclass
class ModelTurn:
    stop_reason: str
    content: list[dict]  # assistant content blocks as plain dicts, replayed verbatim on the next request


class LLM(Protocol):
    def complete(self, system: str, tools: list[dict], messages: list[dict]) -> ModelTurn: ...


class ClaudeLLM:
    """Claude via the official Anthropic SDK.

    Refusal fallbacks are enabled (`fallbacks: "default"`): if a request is declined by a
    safety classifier, the API re-runs it on a fallback model inside the same call.
    """

    def __init__(self, model: str = "claude-opus-5-5", effort: str = "medium", client=None):
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self.client = client
        self.model = model
        self.effort = effort

    def complete(self, system: str, tools: list[dict], messages: list[dict]) -> ModelTurn:
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=16000,
            system=system,
            tools=tools,
            messages=messages,
            output_config={"effort": self.effort},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        # Every block (thinking, fallback, text, tool_use) goes back unchanged on the next turn.
        content = [block.model_dump(mode="json", exclude_none=True) for block in response.content]
        return ModelTurn(stop_reason=response.stop_reason, content=content)


class ScriptedLLM:
    """Replays pre-written assistant turns and records every request it receives."""

    def __init__(self, turns: list[ModelTurn]):
        self.turns = list(turns)
        self.requests: list[dict] = []

    def complete(self, system: str, tools: list[dict], messages: list[dict]) -> ModelTurn:
        import copy

        self.requests.append({"system": system, "tools": tools, "messages": copy.deepcopy(messages)})
        if not self.turns:
            raise AssertionError("ScriptedLLM ran out of turns")
        return self.turns.pop(0)


def text(t: str) -> dict:
    return {"type": "text", "text": t}


def tool_use(id: str, name: str, **input) -> dict:
    return {"type": "tool_use", "id": id, "name": name, "input": input}
