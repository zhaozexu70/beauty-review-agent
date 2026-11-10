"""Versioned classification prompts. The version string is part of the cache key.

v1: deliberately simple baseline.
(v2 is added only after analysing real v1 badcases — see docs/prompt_iteration.md.)
"""
from __future__ import annotations

from src.config import TAXONOMY

TAG_LIST = "、".join(TAXONOMY)

PROMPT_V1 = f"""你是电商评论分析助手。下面是一批洗发水商品评论。
对每条评论：
1. 判断情感 sentiment：positive / neutral / negative
2. 从固定标签中选 1-2 个最相关的 tags：{TAG_LIST}

只输出 JSON，格式：{{"results": [{{"id": 评论id, "sentiment": "...", "tags": ["..."]}}]}}"""

PROMPTS: dict[str, str] = {
    "v1": PROMPT_V1,
}
