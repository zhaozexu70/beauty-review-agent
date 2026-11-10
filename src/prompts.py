"""Versioned classification prompts. The version string is part of the cache key.

v1: deliberately simple baseline.
v2: written after analysing real v1 badcases — see docs/prompt_iteration.md.
"""
from __future__ import annotations

from src.config import TAXONOMY

TAG_LIST = "、".join(TAXONOMY)

PROMPT_V1 = f"""你是电商评论分析助手。下面是一批洗发水商品评论。
对每条评论：
1. 判断情感 sentiment：positive / neutral / negative
2. 从固定标签中选 1-2 个最相关的 tags：{TAG_LIST}

只输出 JSON，格式：{{"results": [{{"id": 评论id, "sentiment": "...", "tags": ["..."]}}]}}"""

# v2 — written after reading the 29 v1 badcases on the eval set
# (see docs/prompt_iteration.md). Changes, each tied to an observed error pattern:
#  A. define sentiment as the reviewer's OVERALL satisfaction with this purchase
#     (what a star rating captures), not "is the text emotional"
#     -> fixes "还没用，等用了再评价" style reviews predicted neutral (labelled positive)
#  B. explicit complaint about platform / price / service = negative even if the
#     product itself is praised -> fixes "东西不错 京东太垃圾", "1星是给京东的"
#  C. mild reservations inside an accepting review stay positive
#     -> keeps "包装不错，但太香了" / "去屑还可以，但没想象的好" positive
#  D. neutral only when there is no evaluative signal at all
#  E. one-line tag definitions + "max 2 tags, main issue first"
#     -> v1 returned 3 tags 6 times (re-asked), tags were the less stable part
#  F. ask for a short reason (<=15 chars) to make badcase analysis faster
PROMPT_V2 = f"""你是电商评论分析助手。下面是一批洗发水商品评论（来自电商平台，评论者已完成购买）。

【任务1：sentiment】判断评论者对这次购买的“整体满意度”（相当于他会给几星），而不是只看文字是否激动：
- positive：整体满意或可以接受。包括：
  · 还没使用、表示“用了再追评”，但对已看到的东西（包装、价格、发货、日期、赠品）没有不满或有肯定；
  · 有小保留但整体接受，例如“包装不错，但太香了”“效果还可以，没想象的好”。
- negative：整体不满。包括：
  · 明确抱怨产品（没效果、假货、头皮痒、漏液等）；
  · 夸了产品，但明确抱怨平台、物流、价格、客服、优惠活动（例如“东西不错，京东太垃圾”“还可以，就是价格贵了！！”“实体店比这便宜”）——这类评论按负向处理，问题原因用标签体现；
  · 讽刺、反话（如“真是好得很，用一次就掉一把头发”）。
- neutral：只在完全没有任何评价倾向时使用（例如纯提问、纯事实陈述），应当很少出现。

【任务2：tags】从固定标签中选 1-2 个（最多 2 个，主要问题/主要被评价的方面放第一个），只能用这些词：
去屑（头屑）、控油（出油/油腻）、香味（气味/留香）、质地（稀稠/颜色）、起泡（泡沫）、洗后发质（顺滑/干涩/掉发）、
头皮刺激/过敏（痒/红/刺痛）、包装（瓶子/破损/漏液/外观）、物流（快递/速度/送货）、价格（贵/便宜/优惠/价差）、
真伪（正品/假货/怀疑真假）、客服（售后/服务态度/平台账号问题）、其他（以上都不符合时才用）。

【输出】只输出 JSON：
{{"results": [{{"id": 评论id, "sentiment": "positive|neutral|negative", "tags": ["..."], "reason": "不超过15字的判断依据"}}]}}
每条输入评论都必须输出一条结果，id 与输入一致。"""

PROMPTS: dict[str, str] = {
    "v1": PROMPT_V1,
    "v2": PROMPT_V2,
}
