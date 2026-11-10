"""The 5 agent tools + their OpenAI function-calling schemas.

A ToolBox holds the session state (loaded reviews, classifications, stats) so
tools can be chained: load_reviews -> classify_reviews -> aggregate_stats ->
write_report. search_kb is independent.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.aggregation import aggregate, compact_stats, to_frame
from src.classifier import classify_reviews as _classify
from src.config import CLEAN_CSV, EVAL_CSV, REPORTS_DIR
from src.llm_client import LLMClient
from src.prompts import PROMPTS
from src.report import build_report
from src.retrieval import KBIndex

LATEST_PROMPT = sorted(PROMPTS)[-1]

TOOL_SCHEMAS: list[dict] = [
    {"type": "function", "function": {
        "name": "load_reviews",
        "description": "加载洗发水评论。source: sample=仓库自带的300条评测样本；full=本地完整清洗数据（需先下载）；upload=用户上传的CSV。",
        "parameters": {"type": "object", "properties": {
            "source": {"type": "string", "enum": ["sample", "full", "upload"]},
            "limit": {"type": "integer", "description": "最多加载多少条，不填则全部"}},
            "required": ["source"]}}},
    {"type": "function", "function": {
        "name": "classify_reviews",
        "description": "用大模型对已加载的评论做情感（positive/neutral/negative）和问题标签分类。review_ids 为空则分类全部已加载评论。",
        "parameters": {"type": "object", "properties": {
            "review_ids": {"type": "array", "items": {"type": "integer"}}}}}},
    {"type": "function", "function": {
        "name": "aggregate_stats",
        "description": "统计已分类评论：情感/标签的数量与占比、负向评论中的高频问题标签及原话示例。",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "search_kb",
        "description": "在知识库（洗发水成分通识、头皮类型、示例售后政策）中检索，返回相关段落及来源文件名。回答产品知识/售后问题前必须调用。",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "write_report",
        "description": "基于 aggregate_stats 的结果生成给品牌方的中文改进报告（关键发现、Top3问题、3条建议、数据局限），并保存为 markdown 文件。",
        "parameters": {"type": "object", "properties": {}}}},
]


class ToolError(Exception):
    pass


class ToolBox:
    def __init__(self, client: Optional[LLMClient], *, prompt_version: str = LATEST_PROMPT,
                 kb: Optional[KBIndex] = None, uploaded_df: Optional[pd.DataFrame] = None,
                 report_dir: Path = REPORTS_DIR):
        self.client = client
        self.prompt_version = prompt_version
        self.kb = kb or KBIndex()
        self.uploaded_df = uploaded_df
        self.report_dir = Path(report_dir)
        self.reviews: Optional[pd.DataFrame] = None
        self.source: str = ""
        self.labels: dict = {}
        self.stats: Optional[dict] = None
        self.report_md: Optional[str] = None

    # ---- tools ------------------------------------------------------------
    def load_reviews(self, source: str = "sample", limit: Optional[int] = None) -> dict:
        if source == "sample":
            df = pd.read_csv(EVAL_CSV, encoding="utf-8-sig")
        elif source == "full":
            if not CLEAN_CSV.exists():
                raise ToolError("完整数据不存在，请先运行 scripts/prepare_data.py")
            df = pd.read_csv(CLEAN_CSV, encoding="utf-8-sig")
        elif source == "upload":
            if self.uploaded_df is None:
                raise ToolError("没有上传的CSV")
            df = normalize_upload(self.uploaded_df)
        else:
            raise ToolError(f"未知 source: {source}")
        if limit:
            df = df.head(int(limit))
        self.reviews = df.reset_index(drop=True)
        self.source, self.labels, self.stats, self.report_md = source, {}, None, None
        return {"source": source, "loaded": len(df), "has_human_label": "label" in df.columns,
                "preview": df["review"].head(3).tolist()}

    def classify_reviews(self, review_ids: Optional[list[int]] = None) -> dict:
        if self.reviews is None:
            raise ToolError("请先调用 load_reviews")
        if self.client is None:
            raise ToolError("没有可用的 LLM 客户端")
        df = self.reviews
        if review_ids:
            df = df[df["review_id"].isin([int(i) for i in review_ids])]
        items = [(int(r.review_id), str(r.review)) for r in df.itertuples()]
        before_calls = self.client.usage.calls
        labels, st = _classify(self.client, items, self.prompt_version)
        self.labels.update(labels)
        self.stats = None
        sent: dict[str, int] = {}
        for lab in labels.values():
            k = lab.sentiment or "failed"
            sent[k] = sent.get(k, 0) + 1
        return {"classified": len(labels), "failed": st.failed_items, "reasks": st.reasks,
                "prompt_version": self.prompt_version, "sentiment_counts": sent,
                "new_api_calls": self.client.usage.calls - before_calls}

    def aggregate_stats(self) -> dict:
        if not self.labels:
            raise ToolError("请先调用 classify_reviews")
        df = to_frame(self.reviews[self.reviews["review_id"].isin(self.labels)], self.labels)
        self.stats = aggregate(df)
        return compact_stats(self.stats)

    def search_kb(self, query: str) -> dict:
        hits = self.kb.search(query)
        if not hits:
            return {"query": query, "passages": [],
                    "note": "知识库中没有找到相关内容。请如实告诉用户知识库无此信息，不要猜测。"}
        return {"query": query, "passages": hits}

    def write_report(self) -> dict:
        if self.stats is None:
            self.aggregate_stats()
        model = self.client.settings.model if self.client else ""
        self.report_md = build_report(self.stats, self.client, source=self.source,
                                      prompt_version=self.prompt_version, model=model)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        path = self.report_dir / f"report_{self.source}_{self.prompt_version}.md"
        path.write_text(self.report_md, encoding="utf-8")
        return {"saved_to": str(path.relative_to(path.parents[2])).replace("\\", "/"),
                "report_markdown": self.report_md}

    # ---- dispatch ---------------------------------------------------------
    def dispatch(self, name: str, args: dict[str, Any]) -> dict:
        fn = {"load_reviews": self.load_reviews, "classify_reviews": self.classify_reviews,
              "aggregate_stats": self.aggregate_stats, "search_kb": self.search_kb,
              "write_report": self.write_report}.get(name)
        if fn is None:
            return {"error": f"unknown tool {name}"}
        try:
            return fn(**args)
        except ToolError as e:
            return {"error": str(e)}
        except TypeError as e:  # bad arguments from the model
            return {"error": f"bad arguments: {e}"}


def normalize_upload(df: pd.DataFrame) -> pd.DataFrame:
    """Accept any CSV with a 'review' column; add review_id if missing; clean."""
    cols = {c.lower().strip(): c for c in df.columns}
    if "review" not in cols:
        raise ToolError("上传的CSV必须包含 review 列")
    out = pd.DataFrame({"review": df[cols["review"]].astype("string").str.strip()})
    if "label" in cols:
        out["label"] = df[cols["label"]]
    out["review_id"] = df[cols["review_id"]].astype(int) if "review_id" in cols else range(len(df))
    out = out[out["review"].notna() & (out["review"] != "")]
    out = out.drop_duplicates(subset="review")
    return out[["review_id", "review"] + (["label"] if "label" in out else [])].reset_index(drop=True)


def tool_result_for_llm(name: str, result: dict) -> str:
    """Serialize a tool result for the model; trim the long report body."""
    if name == "write_report" and "report_markdown" in result:
        result = {**result, "report_markdown": result["report_markdown"][:1500]}
    return json.dumps(result, ensure_ascii=False)
