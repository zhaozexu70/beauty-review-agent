import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import LLMSettings  # noqa: E402
from src.llm_client import ChatResult, UsageTracker  # noqa: E402


class ScriptedClient:
    """Mock LLM: returns pre-scripted ChatResults in order; records requests."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.usage = UsageTracker()
        self.settings = LLMSettings(api_key="test", base_url="http://x", model="mock")

    def chat(self, messages, *, prompt_version, tools=None, json_mode=False):
        self.requests.append({"messages": json.loads(json.dumps(messages)),
                              "prompt_version": prompt_version, "tools": tools})
        r = self.responses.pop(0)
        r = r(messages) if callable(r) else r
        self.usage.add(r)
        return r


@pytest.fixture
def scripted():
    return ScriptedClient


def fake_sdk(contents):
    """Fake OpenAI SDK object whose chat.completions.create returns given contents."""
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        item = contents[min(len(calls) - 1, len(contents) - 1)]
        if isinstance(item, Exception):
            raise item
        msg = SimpleNamespace(content=item, tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)],
                               usage=SimpleNamespace(prompt_tokens=100, completion_tokens=50))

    sdk = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return sdk, calls


@pytest.fixture
def make_fake_sdk():
    return fake_sdk


def tool_call(name, args, id_="call_1"):
    return {"id": id_, "name": name, "arguments": json.dumps(args, ensure_ascii=False)}


def chat_result(content=None, tool_calls=None):
    return ChatResult(content=content, tool_calls=tool_calls or [], prompt_tokens=10,
                      completion_tokens=5)
