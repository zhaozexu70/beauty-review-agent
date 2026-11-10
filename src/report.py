"""Chinese improvement report.

Numbers, shares and quotes are filled in by CODE from aggregate_stats, so the
LLM cannot invent them. The LLM only writes the "key findings" and the three
"suggestions", based on the same stats JSON. If the LLM is unavailable (demo
mode cache miss) or returns invalid JSON, a rule-based fallback text is used.
"""
from __future__ import annotations

import json
from typing import Optional

from pydantic import BaseModel, Field, ValidationError

from src.aggregation import compact_stats
from src.llm_client import DemoCacheMiss, LLMClient

REPORT_PROMPT_VERSION = "report_v1"

REPORT_SYSTEM = """你是美妆个护电商的数据分析师。根据给定的评论统计 JSON，写给品牌方的洞察。
要求：
- 只能使用 JSON 里出现的数字，不要编造任何数字、比例或原因；推测要写明“可能”。
- findings：3 条关键发现，每条一句话。
- suggestions：恰好 3 条可执行建议，分别对应 top_problems 中的问题；每条包含 problem（问题标签）、action（具体动作）、metric（用什么指标验证效果）。
- 不提供医疗建议。
只输出 JSON：{"findings": ["..."], "suggestions": [{"problem": "...", "action": "...", "metric": "..."}]}"""


class Suggestion(BaseModel):
    problem: str
    action: str
    metric: str


class ReportLLMOutput(BaseModel):
    findings: list[str] = Field(min_length=1, max_length=5)
    suggestions: list[Suggestion] = Field(min_length=3, max_length=3)


FALLBACK_ACTIONS = {
    "物流": ("与快递服务商复盘时效与破损，针对慢件地区设置预警并主动告知买家", "物流差评占比、平均签收时长"),
    "包装": ("加固瓶盖/压泵并增加防漏封口，发货前抽检", "包装相关差评数、破损退款率"),
    "真伪": ("详情页展示正品溯源/批次查询方式，客服提供批次核验话术", "真伪类差评占比、相关咨询量"),
    "去屑": ("在详情页说明去屑需持续使用的周期与适用人群，管理消费者预期", "去屑类差评占比、复购率"),
    "控油": ("按头皮类型细分推荐，详情页标注适合油性头皮的使用方法", "控油类差评占比"),
    "洗后发质": ("针对干涩反馈推荐搭配护发素，或评估配方调理性", "洗后发质类差评占比"),
    "香味": ("提供香型说明与小样试用，减少香味预期落差", "香味类差评占比"),
    "质地": ("在详情页说明产品正常质地，客服准备批次差异解释话术", "质地类差评占比"),
    "起泡": ("说明泡沫与清洁力的关系，减少“泡沫少=没洗干净”的误解", "起泡类差评占比"),
    "头皮刺激/过敏": ("醒目提示敏感人群先小范围试用，建立不适反馈快速处理流程", "刺激/过敏类差评数、退款时长"),
    "价格": ("复盘大促前后价格策略，避免先涨后降引发不满，落实价保规则", "价格类差评占比"),
    "客服": ("梳理高频问题话术与响应时效 SLA，抽检客服会话", "客服类差评占比、首响时长"),
}


def _fallback(stats: dict) -> ReportLLMOutput:
    tp = stats["top_problems"]
    s = stats["sentiment_shares"]
    findings = [f"已分类 {stats['n_classified']} 条评论，负向占 {s['negative']:.1%}，正向占 {s['positive']:.1%}，中性占 {s['neutral']:.1%}。"]
    if tp:
        findings.append("负向评论中提及最多的问题依次是：" +
                        "、".join(f"{x['tag']}（{x['share_of_negative']:.1%}）" for x in tp) + "。")
    sugg = []
    for x in tp:
        a, m = FALLBACK_ACTIONS.get(x["tag"], ("针对该问题做专项复盘", "该标签差评占比"))
        sugg.append(Suggestion(problem=x["tag"], action=a, metric=m))
    while len(sugg) < 3:
        sugg.append(Suggestion(problem="数据", action="扩大样本量后再做判断", metric="样本量"))
    return ReportLLMOutput(findings=findings, suggestions=sugg)


def llm_sections(stats: dict, client: Optional[LLMClient]) -> tuple[ReportLLMOutput, str]:
    if client is None:
        return _fallback(stats), "rule-based fallback"
    cs = compact_stats(stats)
    cs["top_problems"] = [x["tag"] for x in stats["top_problems"]]
    messages = [{"role": "system", "content": REPORT_SYSTEM},
                {"role": "user", "content": json.dumps(cs, ensure_ascii=False)}]
    try:
        r = client.chat(messages, prompt_version=REPORT_PROMPT_VERSION, json_mode=True)
        return ReportLLMOutput.model_validate_json(r.content or ""), "LLM"
    except (DemoCacheMiss, ValidationError, ValueError) as e:
        return _fallback(stats), f"rule-based fallback ({type(e).__name__})"


def build_report(stats: dict, client: Optional[LLMClient] = None, *, source: str = "",
                 prompt_version: str = "", model: str = "") -> str:
    out, mode = llm_sections(stats, client)
    s, c = stats["sentiment_shares"], stats["sentiment_counts"]
    L = ["# 洗发水评论洞察报告", "",
         f"> 数据来源：{source or '未注明'}；分类 prompt：{prompt_version}；模型：{model}；"
         f"文字部分生成方式：{mode}。所有数字由代码根据分类结果统计得出。", "",
         "## 一、核心数据", "",
         "| 指标 | 数值 |", "|---|---:|",
         f"| 评论总数 | {stats['n_total']} |",
         f"| 成功分类 | {stats['n_classified']} |",
         f"| 分类失败（JSON无效） | {stats['n_failed']} |",
         f"| 正向 | {c['positive']}（{s['positive']:.1%}） |",
         f"| 中性 | {c['neutral']}（{s['neutral']:.1%}） |",
         f"| 负向 | {c['negative']}（{s['negative']:.1%}） |", "",
         "## 二、关键发现", ""]
    L += [f"{i}. {f}" for i, f in enumerate(out.findings, 1)]
    L += ["", "## 三、Top 3 问题（按负向评论中的提及次数，不含“其他”）", ""]
    if not stats["top_problems"]:
        L.append("（负向评论不足，无法得出 Top 问题）")
    for i, x in enumerate(stats["top_problems"], 1):
        L += [f"### {i}. {x['tag']}",
              f"- 提及次数：{x['count']}；占负向评论 {x['share_of_negative']:.1%}；"
              f"占全部已分类评论 {x['share_of_all']:.1%}",
              "- 典型原话："]
        L += [f"  - “{q['text']}”（#{q['review_id']}）" for q in x["quotes"]]
        L.append("")
    L += ["## 四、改进建议", ""]
    for i, sg in enumerate(out.suggestions, 1):
        L.append(f"{i}. **{sg.problem}**：{sg.action}（验证指标：{sg.metric}）")
    L += ["", "## 五、数据说明与局限", "",
          "- 情感与标签由大模型自动标注；情感已在 300 条人工标签评测集上评估，标签尚无人工金标准，占比仅供参考。",
          "- 原始数据集正负样本约各占一半（构建时人为平衡），因此正/负向占比不代表真实店铺的好评率。",
          "- 仅覆盖“洗发水”一个品类，数据来自公开语料，商品/品牌/时间信息缺失，无法做分商品或趋势分析。",
          "- 一条评论最多 2 个标签，各标签占比之和可能超过 100%。",
          f"- 本次有 {stats['n_failed']} 条评论分类失败，未计入占比。", ""]
    return "\n".join(L)
