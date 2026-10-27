"""Checks the installed Anthropic SDK accepts every parameter ClaudeLLM sends. No network calls."""
import inspect

import pytest

anthropic = pytest.importorskip("anthropic")


def test_beta_messages_create_accepts_our_parameters():
    client = anthropic.Anthropic(api_key="test-key")
    params = inspect.signature(client.beta.messages.create).parameters
    for name in ("model", "max_tokens", "system", "tools", "messages", "output_config", "betas", "fallbacks"):
        assert name in params, f"SDK has no '{name}' parameter on beta.messages.create"


def test_messages_create_accepts_judge_parameters():
    client = anthropic.Anthropic(api_key="test-key")
    params = inspect.signature(client.messages.create).parameters
    for name in ("model", "max_tokens", "messages", "output_config"):
        assert name in params
