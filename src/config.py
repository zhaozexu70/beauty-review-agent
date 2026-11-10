"""Central configuration: paths, taxonomy, and LLM settings loaded from .env."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EVAL_DIR = DATA_DIR / "eval"
KB_DIR = ROOT / "kb"
OUTPUTS_DIR = ROOT / "outputs"
CACHE_DIR = OUTPUTS_DIR / "cache" / "llm"
EVAL_RESULTS_DIR = OUTPUTS_DIR / "eval_results"
REPORTS_DIR = OUTPUTS_DIR / "reports"

RAW_ZIP = RAW_DIR / "online_shopping_10_cats.zip"
RAW_URL = (
    "https://raw.githubusercontent.com/SophonPlus/ChineseNlpCorpus/master/"
    "datasets/online_shopping_10_cats/online_shopping_10_cats.zip"
)
CLEAN_CSV = PROCESSED_DIR / "shampoo_clean.csv"
EVAL_CSV = EVAL_DIR / "eval_set.csv"
HOLDOUT_CSV = EVAL_DIR / "holdout_set.csv"
TAG_SHEET_CSV = EVAL_DIR / "tag_labeling_sheet.csv"

# FIXED issue-tag taxonomy. The model may only use these values.
TAXONOMY: list[str] = [
    "去屑", "控油", "香味", "质地", "起泡", "洗后发质", "头皮刺激/过敏",
    "包装", "物流", "价格", "真伪", "客服", "其他",
]
SENTIMENTS: list[str] = ["positive", "neutral", "negative"]


@dataclass(frozen=True)
class LLMSettings:
    api_key: str
    base_url: str
    model: str
    timeout: float = 60.0
    max_retries: int = 4
    price_input_per_m: float = 2.0
    price_output_per_m: float = 3.0
    currency: str = "CNY"


def get_settings() -> LLMSettings:
    return LLMSettings(
        api_key=os.getenv("LLM_API_KEY", "").strip(),
        base_url=os.getenv("LLM_BASE_URL", "").strip() or "https://api.deepseek.com",
        model=os.getenv("LLM_MODEL", "").strip() or "deepseek-chat",
        price_input_per_m=float(os.getenv("LLM_PRICE_INPUT_PER_M", "2") or 2),
        price_output_per_m=float(os.getenv("LLM_PRICE_OUTPUT_PER_M", "3") or 3),
        currency=os.getenv("LLM_PRICE_CURRENCY", "CNY") or "CNY",
    )


def has_api_key() -> bool:
    key = get_settings().api_key
    return bool(key) and not key.lower().startswith(("your", "sk-xxx", "<"))
