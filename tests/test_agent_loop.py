import json

from src.agent import run_agent
from src.tools import TOOL_SCHEMAS, ToolBox
from tests.conftest import chat_result, tool_call


def test_tool_schemas_are_the_five_tools():
    assert [t["function"]["name"] for t in TOOL_SCHEMAS] == [
        "load_reviews", "classify_reviews", "aggregate_stats", "search_kb", "write_report"]


def test_agent_calls_kb_and_cites(scripted):
    client = scripted([
        chat_result(tool_calls=[tool_call("search_kb", {"query": "漏液怎么办"})]),
        chat_result("签收后48小时内拍照联系客服【来源：after_sales_policy.md】"),
    ])
    res = run_agent("洗发水漏液了怎么办？", ToolBox(client), client)
    assert [s.tool for s in res.steps] == ["search_kb"]
    assert res.retrieved_sources[0] == "after_sales_policy.md"
    assert res.cited_sources == ["after_sales_policy.md"]
    tool_msg = client.requests[1]["messages"][-1]
    assert tool_msg["role"] == "tool" and tool_msg["tool_call_id"] == "call_1"
    assert client.requests[1]["messages"][-2]["tool_calls"][0]["function"]["name"] == "search_kb"
    assert client.requests[0]["tools"] == TOOL_SCHEMAS


def test_agent_kb_no_answer_flag(scripted):
    client = scripted([
        chat_result(tool_calls=[tool_call("search_kb", {"query": "今天股票涨了吗"})]),
        chat_result("知识库中没有相关信息。"),
    ])
    res = run_agent("今天股票涨了吗", ToolBox(client), client)
    assert res.kb_no_answer and res.cited_sources == []
    assert "没有找到" in json.loads(client.requests[1]["messages"][-1]["content"])["note"]


def test_agent_chains_data_tools(scripted, tmp_path):
    def classify_answer(messages):
        items = json.loads(messages[-1]["content"].split("\n", 1)[1])
        return chat_result(json.dumps({"results": [
            {"id": x["id"], "sentiment": "negative", "tags": ["物流"]} for x in items]}))
    client = scripted([
        chat_result(tool_calls=[tool_call("load_reviews", {"source": "sample", "limit": 5}, "c1")]),
        chat_result(tool_calls=[tool_call("classify_reviews", {}, "c2")]),
        classify_answer,  # the classifier's own LLM call, made inside the tool
        chat_result(tool_calls=[tool_call("aggregate_stats", {}, "c3")]),
        chat_result(tool_calls=[tool_call("write_report", {}, "c4")]),
        chat_result("not json"),  # report LLM call -> invalid -> rule-based fallback
        chat_result("差评主要集中在物流。"),
    ])
    tb = ToolBox(client, report_dir=tmp_path / "outputs" / "reports")
    res = run_agent("差评集中在哪？写个报告", tb, client)
    assert [s.tool for s in res.steps] == ["load_reviews", "classify_reviews",
                                           "aggregate_stats", "write_report"]
    assert res.steps[2].result["sentiment_counts"]["negative"] == 5
    assert "物流" in res.steps[3].result["report_markdown"]
    assert res.answer == "差评主要集中在物流。"


def test_agent_handles_bad_args_and_unknown_tool(scripted):
    client = scripted([
        chat_result(tool_calls=[{"id": "a", "name": "search_kb", "arguments": "{bad"},
                                {"id": "b", "name": "rm_rf", "arguments": "{}"}]),
        chat_result("抱歉"),
    ])
    res = run_agent("x", ToolBox(client), client)
    assert "invalid arguments" in res.steps[0].result["error"]
    assert "unknown tool" in res.steps[1].result["error"]


def test_tool_order_errors_are_returned_not_raised(scripted):
    client = scripted([chat_result(tool_calls=[tool_call("aggregate_stats", {})]),
                       chat_result("需要先加载数据")])
    res = run_agent("统计", ToolBox(client), client)
    assert "classify_reviews" in res.steps[0].result["error"]


def test_agent_stops_at_max_steps(scripted):
    client = scripted([chat_result(tool_calls=[tool_call("search_kb", {"query": "硅油"})])] * 3)
    res = run_agent("loop", ToolBox(client), client, max_steps=3)
    assert res.stopped_reason == "max_steps" and len(res.steps) == 3
