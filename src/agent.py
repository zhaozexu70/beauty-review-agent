"""A hand-written tool-calling agent loop (no framework), so every step is visible.

loop:
  1. send messages + tool schemas to the LLM
  2. if it returns tool_calls -> run each tool, append results as role=tool, go to 1
  3. else it returned a final answer -> stop
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional

from src.llm_client import LLMClient
from src.tools import TOOL_SCHEMAS, ToolBox, tool_result_for_llm

AGENT_PROMPT_VERSION = "agent_v1"
MAX_STEPS = 8

SYSTEM_PROMPT = """你是“个护美妆评论洞察 Agent”，服务于一家美妆电商的数据分析团队，当前品类是洗发水。
你可以调用工具：load_reviews、classify_reviews、aggregate_stats、search_kb、write_report。

规则：
1. 涉及评论数据的问题（如“差评主要集中在哪”），按 load_reviews → classify_reviews → aggregate_stats 的顺序调用工具；回答中的数字必须来自工具结果，不要自己估算。用户没指定数据源时用 source="sample"。
2. 涉及产品成分、头皮类型、售后政策的问题，必须先调用 search_kb。回答时在相关句子后用【来源：文件名】标注出处。
3. 如果 search_kb 没有返回段落，或返回的段落不能回答问题，必须明确说“知识库中没有相关信息”，不要凭常识猜测。
4. 不提供医疗诊断或治疗建议；涉及身体不适时建议停用并咨询医生。
5. 用户要求写报告时调用 write_report，然后用几句话总结报告要点并告知保存路径。
6. 用简洁的中文回答。"""


@dataclass
class ToolStep:
    tool: str
    arguments: dict
    result: dict


@dataclass
class AgentResult:
    answer: str
    steps: list[ToolStep] = field(default_factory=list)
    retrieved_sources: list[str] = field(default_factory=list)
    cited_sources: list[str] = field(default_factory=list)
    kb_no_answer: bool = False  # search_kb was called but returned nothing
    stopped_reason: str = "final_answer"


def run_agent(question: str, toolbox: ToolBox, client: LLMClient,
              history: Optional[list[dict]] = None, max_steps: int = MAX_STEPS) -> AgentResult:
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += history or []
    messages.append({"role": "user", "content": question})
    steps: list[ToolStep] = []
    answer, reason = "", "final_answer"

    for _ in range(max_steps):
        r = client.chat(messages, prompt_version=AGENT_PROMPT_VERSION, tools=TOOL_SCHEMAS)
        if not r.tool_calls:
            answer = r.content or ""
            break
        messages.append({"role": "assistant", "content": r.content or "", "tool_calls": [
            {"id": tc["id"], "type": "function",
             "function": {"name": tc["name"], "arguments": tc["arguments"]}}
            for tc in r.tool_calls]})
        for tc in r.tool_calls:
            try:
                args = json.loads(tc["arguments"] or "{}")
                if not isinstance(args, dict):
                    raise ValueError("arguments must be an object")
            except (json.JSONDecodeError, ValueError) as e:
                result = {"error": f"invalid arguments JSON: {e}"}
                args = {}
            else:
                result = toolbox.dispatch(tc["name"], args)
            steps.append(ToolStep(tc["name"], args, result))
            messages.append({"role": "tool", "tool_call_id": tc["id"],
                             "content": tool_result_for_llm(tc["name"], result)})
    else:
        answer, reason = "已达到最大工具调用步数，未能得出最终回答。", "max_steps"

    retrieved: list[str] = []
    kb_called, kb_empty = False, True
    for s in steps:
        if s.tool == "search_kb":
            kb_called = True
            for p in s.result.get("passages", []):
                kb_empty = False
                if p["source"] not in retrieved:
                    retrieved.append(p["source"])
    cited = [src for src in retrieved if src in answer or src.removesuffix(".md") in answer]
    # also catch citations of files the model names that were NOT retrieved (hallucinated)
    for m in re.findall(r"来源[:：]\s*([\w\-]+\.md)", answer):
        if m not in cited:
            cited.append(m)
    return AgentResult(answer=answer, steps=steps, retrieved_sources=retrieved,
                       cited_sources=cited, kb_no_answer=kb_called and kb_empty,
                       stopped_reason=reason)
