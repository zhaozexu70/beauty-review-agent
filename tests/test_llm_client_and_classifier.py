import json

import pytest

from src.classifier import classify_reviews
from src.config import LLMSettings
from src.llm_client import DemoCacheMiss, LLMClient
from tests.conftest import chat_result

S = LLMSettings(api_key="test", base_url="http://x", model="mock", max_retries=2)


def ok_json(ids, tag="香味"):
    return json.dumps({"results": [{"id": i, "sentiment": "positive", "tags": [tag]} for i in ids]})


def test_cache_hit_costs_nothing(tmp_path, make_fake_sdk):
    sdk, calls = make_fake_sdk([ok_json([1])])
    c = LLMClient(S, cache_dir=tmp_path, sdk_client=sdk)
    msgs = [{"role": "user", "content": "hi"}]
    c.chat(msgs, prompt_version="v1", json_mode=True)
    r2 = c.chat(msgs, prompt_version="v1", json_mode=True)
    assert len(calls) == 1 and r2.cached
    assert c.usage.calls == 1 and c.usage.cache_hits == 1 and c.usage.prompt_tokens == 100
    c.chat(msgs, prompt_version="v2", json_mode=True)  # new prompt version -> cache miss
    assert len(calls) == 2
    assert calls[0]["temperature"] == 0
    assert calls[0]["response_format"] == {"type": "json_object"}


def test_cache_can_be_disabled(tmp_path, make_fake_sdk):
    sdk, calls = make_fake_sdk([ok_json([1])])
    msgs = [{"role": "user", "content": "hi"}]
    LLMClient(S, cache_dir=tmp_path, sdk_client=sdk).chat(msgs, prompt_version="v1")
    LLMClient(S, cache_dir=tmp_path, sdk_client=sdk, read_cache=False).chat(msgs, prompt_version="v1")
    assert len(calls) == 2


def test_retry_with_backoff(tmp_path, make_fake_sdk, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    sdk, calls = make_fake_sdk([ConnectionError("boom"), ok_json([1])])
    c = LLMClient(S, cache_dir=tmp_path, sdk_client=sdk)
    r = c.chat([{"role": "user", "content": "x"}], prompt_version="v1")
    assert len(calls) == 2 and "results" in r.content


def test_gives_up_after_max_retries(tmp_path, make_fake_sdk, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    sdk, calls = make_fake_sdk([ConnectionError("boom")])
    c = LLMClient(S, cache_dir=tmp_path, sdk_client=sdk)
    with pytest.raises(RuntimeError):
        c.chat([{"role": "user", "content": "x"}], prompt_version="v1")
    assert len(calls) == S.max_retries + 1


def test_demo_mode_raises_on_cache_miss(tmp_path):
    c = LLMClient(LLMSettings(api_key="", base_url="", model="m"), cache_dir=tmp_path)
    assert c.demo_mode
    with pytest.raises(DemoCacheMiss):
        c.chat([{"role": "user", "content": "x"}], prompt_version="v1")


def test_classifier_reasks_once_then_marks_failed(scripted):
    reviews = [(1, "很好"), (2, "物流太慢")]
    first = chat_result(json.dumps({"results": [
        {"id": 1, "sentiment": "positive", "tags": ["其他"]},
        {"id": 2, "sentiment": "negative", "tags": ["快递"]}]}))   # invalid tag
    second = chat_result("still not json")                           # re-ask also fails
    client = scripted([first, second])
    labels, st = classify_reviews(client, reviews, "v1", max_workers=1)
    assert len(client.requests) == 2 and st.reasks == 1
    assert labels[1].sentiment == "positive" and not labels[1].failed
    assert labels[2].failed and st.failed_items == 1
    reask = client.requests[1]["messages"][-1]["content"]
    assert '"id": 2' in reask and '"id": 1,' not in reask  # only the broken id is re-asked


def test_classifier_reask_fixes(scripted):
    client = scripted([chat_result("{bad"), chat_result(ok_json([1, 2]))])
    labels, st = classify_reviews(client, [(1, "a"), (2, "b")], "v1", max_workers=1)
    assert st.failed_items == 0 and st.reasks == 1


def test_batches_of_20(scripted):
    def respond(messages):
        items = json.loads(messages[-1]["content"].split("\n", 1)[1])
        return chat_result(ok_json([x["id"] for x in items]))
    client = scripted([respond] * 3)
    reviews = [(i, f"r{i}") for i in range(45)]
    labels, st = classify_reviews(client, reviews, "v1", max_workers=1)
    assert st.batches == 3 and len(labels) == 45 and st.failed_items == 0
