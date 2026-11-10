"""OpenAI-compatible LLM client with: temperature 0, timeout, retries with
exponential backoff, an on-disk cache, token tracking, and a demo mode.

Cache key = sha256(prompt_version, model, messages, tools, json_mode).
So re-running the same prompt version on the same input costs nothing, and a
new prompt version automatically misses the cache.
"""
from __future__ import annotations

import hashlib
import json
import random
import threading
import time
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from src.config import CACHE_DIR, OUTPUTS_DIR, LLMSettings, get_settings


class DemoCacheMiss(RuntimeError):
    """Raised in demo mode (no API key) when a request is not in the cache."""


@dataclass
class ChatResult:
    content: Optional[str]
    tool_calls: list[dict] = field(default_factory=list)  # [{id, name, arguments(str)}]
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached: bool = False


@dataclass
class UsageTracker:
    calls: int = 0
    cache_hits: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, r: ChatResult) -> None:
        with self._lock:
            if r.cached:
                self.cache_hits += 1
            else:
                self.calls += 1
                self.prompt_tokens += r.prompt_tokens
                self.completion_tokens += r.completion_tokens

    def cost(self, s: LLMSettings) -> float:
        return (self.prompt_tokens * s.price_input_per_m
                + self.completion_tokens * s.price_output_per_m) / 1_000_000

    def summary(self, s: LLMSettings) -> dict:
        return {
            "api_calls": self.calls,
            "cache_hits": self.cache_hits,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "estimated_cost": round(self.cost(s), 4),
            "currency": s.currency,
            "price_note": (f"input {s.price_input_per_m}/M, output {s.price_output_per_m}/M "
                           f"({s.currency}, cache-miss price, estimate only)"),
        }


def cache_key(prompt_version: str, model: str, messages: list[dict],
              tools: Optional[list[dict]], json_mode: bool) -> str:
    payload = json.dumps(
        {"v": prompt_version, "model": model, "messages": messages,
         "tools": tools, "json": json_mode},
        ensure_ascii=False, sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class LLMClient:
    def __init__(self, settings: Optional[LLMSettings] = None, *, cache_dir: Path = CACHE_DIR,
                 read_cache: bool = True, write_cache: bool = True,
                 demo_mode: Optional[bool] = None, sdk_client: Any = None):
        self.settings = settings or get_settings()
        self.cache_dir = Path(cache_dir)
        self.read_cache = read_cache
        self.write_cache = write_cache
        self.demo_mode = (not self.settings.api_key) if demo_mode is None else demo_mode
        self.usage = UsageTracker()
        self._sdk = sdk_client  # injectable for tests

    # ---- SDK ---------------------------------------------------------------
    def _client(self):
        if self._sdk is None:
            from openai import OpenAI
            self._sdk = OpenAI(api_key=self.settings.api_key, base_url=self.settings.base_url,
                               timeout=self.settings.timeout, max_retries=0)  # we retry ourselves
        return self._sdk

    def _call_api(self, messages, tools, json_mode) -> ChatResult:
        kwargs: dict[str, Any] = dict(model=self.settings.model, messages=messages, temperature=0)
        if tools:
            kwargs["tools"] = tools
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        last_err: Exception | None = None
        for attempt in range(self.settings.max_retries + 1):
            try:
                resp = self._client().chat.completions.create(**kwargs)
                msg = resp.choices[0].message
                tcs = [{"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments}
                       for tc in (msg.tool_calls or [])]
                u = resp.usage
                return ChatResult(content=msg.content, tool_calls=tcs,
                                  prompt_tokens=getattr(u, "prompt_tokens", 0) or 0,
                                  completion_tokens=getattr(u, "completion_tokens", 0) or 0)
            except Exception as e:  # network / rate limit / 5xx
                if _is_non_retryable(e):
                    raise
                last_err = e
                if attempt < self.settings.max_retries:
                    wait = min(30.0, 2 ** attempt) + random.uniform(0, 0.5)
                    print(f"[llm] {type(e).__name__}; retry {attempt + 1} in {wait:.1f}s")
                    time.sleep(wait)
        raise RuntimeError(f"LLM call failed after retries: {last_err}")

    # ---- public ------------------------------------------------------------
    def chat(self, messages: list[dict], *, prompt_version: str,
             tools: Optional[list[dict]] = None, json_mode: bool = False) -> ChatResult:
        key = cache_key(prompt_version, self.settings.model, messages, tools, json_mode)
        path = self.cache_dir / f"{key}.json"
        if (self.read_cache or self.demo_mode) and path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            r = ChatResult(content=data["content"], tool_calls=data.get("tool_calls", []),
                           prompt_tokens=data.get("prompt_tokens", 0),
                           completion_tokens=data.get("completion_tokens", 0), cached=True)
            self.usage.add(r)
            return r
        if self.demo_mode:
            raise DemoCacheMiss("演示模式（未配置 API Key）：该请求不在缓存中。")
        r = self._call_api(messages, tools, json_mode)
        self.usage.add(r)
        if self.write_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(
                {"prompt_version": prompt_version, "model": self.settings.model,
                 "content": r.content, "tool_calls": r.tool_calls,
                 "prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens},
                ensure_ascii=False, indent=1), encoding="utf-8")
        return r


def _is_non_retryable(e: Exception) -> bool:
    name = type(e).__name__
    return name in {"AuthenticationError", "PermissionDeniedError", "BadRequestError",
                    "NotFoundError"}


def log_usage(script: str, extra: dict, usage: dict) -> None:
    OUTPUTS_DIR.mkdir(exist_ok=True)
    with open(OUTPUTS_DIR / "usage_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"time": datetime.now().isoformat(timespec="seconds"),
                            "script": script, **extra, **usage}, ensure_ascii=False) + "\n")
