"""Batch review classification with JSON validation and one re-ask."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from src.llm_client import LLMClient
from src.prompts import PROMPTS
from src.schemas import ClassifiedReview, validate_batch

BATCH_SIZE = 20


@dataclass
class ClassifyStats:
    batches: int = 0
    reasks: int = 0
    failed_items: int = 0
    errors: list[str] = field(default_factory=list)        # still invalid after re-ask
    first_errors: list[str] = field(default_factory=list)  # what triggered a re-ask


def _user_message(batch: list[tuple[int, str]]) -> str:
    items = [{"id": rid, "text": text} for rid, text in batch]
    return "评论列表（JSON）：\n" + json.dumps(items, ensure_ascii=False)


def classify_batch(client: LLMClient, batch: list[tuple[int, str]], version: str
                   ) -> tuple[dict[int, ClassifiedReview], int, list[str], list[str]]:
    """Classify one batch. Returns (results, n_reasks, final_errors, first_errors)."""
    system = PROMPTS[version]
    pv = f"classify_{version}"
    ids = [rid for rid, _ in batch]
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": _user_message(batch)}]
    r = client.chat(messages, prompt_version=pv, json_mode=True)
    valid, errors = validate_batch(r.content or "", ids)
    first_errors = list(errors)
    reasks = 0
    if errors:  # re-ask ONCE, only for the broken ids
        reasks = 1
        bad_ids = [i for i in ids if i not in valid]
        bad_batch = [(rid, t) for rid, t in batch if rid in bad_ids]
        messages2 = messages + [
            {"role": "assistant", "content": r.content or ""},
            {"role": "user", "content": (
                "你的输出有问题：\n" + "\n".join(errors[:20]) +
                "\n请只针对下列评论重新输出，严格遵守 JSON 格式与固定标签：\n" + _user_message(bad_batch))},
        ]
        r2 = client.chat(messages2, prompt_version=pv, json_mode=True)
        valid2, errors = validate_batch(r2.content or "", bad_ids)
        valid.update(valid2)
    out: dict[int, ClassifiedReview] = {}
    for rid in ids:
        if rid in valid:
            v = valid[rid]
            out[rid] = ClassifiedReview(id=rid, sentiment=v.sentiment, tags=v.tags, reason=v.reason)
        else:
            err = next((e for e in errors if e.startswith(f"id={rid}:")), "invalid output")
            out[rid] = ClassifiedReview(id=rid, failed=True, error=err)
    return out, reasks, errors, first_errors


def classify_reviews(client: LLMClient, reviews: list[tuple[int, str]], version: str = "v2",
                     batch_size: int = BATCH_SIZE, max_workers: int = 4
                     ) -> tuple[dict[int, ClassifiedReview], ClassifyStats]:
    """Classify many reviews in fixed-order batches (fixed order => stable cache keys)."""
    batches = [reviews[i:i + batch_size] for i in range(0, len(reviews), batch_size)]
    stats = ClassifyStats(batches=len(batches))
    results: dict[int, ClassifiedReview] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for out, reasks, errs, first in pool.map(lambda b: classify_batch(client, b, version), batches):
            results.update(out)
            stats.reasks += reasks
            stats.errors.extend(errs)
            stats.first_errors.extend(first)
    stats.failed_items = sum(1 for r in results.values() if r.failed)
    return results, stats
