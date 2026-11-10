"""Pydantic schemas: the contract between the LLM's JSON output and our code."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

from src.config import TAXONOMY

Sentiment = Literal["positive", "neutral", "negative"]


class ReviewLabel(BaseModel):
    """One classified review, exactly as the LLM must return it."""

    id: int
    sentiment: Sentiment
    tags: list[str] = Field(min_length=1, max_length=2)
    reason: Optional[str] = None  # short evidence (prompt v2 asks for it)

    @field_validator("tags")
    @classmethod
    def tags_in_taxonomy(cls, v: list[str]) -> list[str]:
        bad = [t for t in v if t not in TAXONOMY]
        if bad:
            raise ValueError(f"tags not in fixed taxonomy: {bad}")
        if len(set(v)) != len(v):
            raise ValueError("duplicate tags")
        return v


class BatchOutput(BaseModel):
    results: list[ReviewLabel]


class ClassifiedReview(BaseModel):
    """What we store per review after validation (failed items included)."""

    id: int
    sentiment: Optional[Sentiment] = None
    tags: list[str] = Field(default_factory=list)
    reason: Optional[str] = None
    failed: bool = False
    error: Optional[str] = None


def validate_batch(raw_json: str, expected_ids: list[int]) -> tuple[dict[int, ReviewLabel], list[str]]:
    """Parse + validate one batch response.

    Returns (valid items by id, list of error strings). Items that fail
    validation, are missing, or have unexpected ids are reported as errors so
    the caller can re-ask once and then mark them failed.
    """
    import json

    from pydantic import ValidationError

    errors: list[str] = []
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as e:
        return {}, [f"invalid JSON: {e}"]
    items = data.get("results") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return {}, ["top-level object must be {\"results\": [...]}"]

    valid: dict[int, ReviewLabel] = {}
    expected = set(expected_ids)
    for item in items:
        try:
            lab = ReviewLabel.model_validate(item)
        except ValidationError as e:
            rid = item.get("id") if isinstance(item, dict) else None
            msg = "; ".join(err["msg"] for err in e.errors())
            errors.append(f"id={rid}: {msg}")
            continue
        if lab.id not in expected:
            errors.append(f"id={lab.id}: unexpected id")
            continue
        valid[lab.id] = lab
    missing = sorted(expected - set(valid))
    already = {e.split(":")[0] for e in errors}
    for rid in missing:
        if f"id={rid}" not in already:
            errors.append(f"id={rid}: missing from output")
    return valid, errors
