"""Run the agent on the preset demo questions and save full transcripts.

This also fills the LLM cache, so the Streamlit app can replay these exact
questions in demo mode (no API key).

Usage (Windows):  .venv\\Scripts\\python scripts\\run_agent_demo.py
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.agent import run_agent  # noqa: E402
from src.config import OUTPUTS_DIR, ROOT, get_settings  # noqa: E402
from src.llm_client import LLMClient, log_usage  # noqa: E402
from src.tools import LATEST_PROMPT, ToolBox  # noqa: E402


def main() -> None:
    s = get_settings()
    client = LLMClient(s)
    questions = json.loads((ROOT / "scripts" / "demo_questions.json").read_text(encoding="utf-8"))
    out = []
    for q in questions:
        res = run_agent(q, ToolBox(client, prompt_version=LATEST_PROMPT), client)
        print("=" * 80, f"\nQ: {q}")
        for i, step in enumerate(res.steps, 1):
            print(f"  [{i}] {step.tool}({json.dumps(step.arguments, ensure_ascii=False)})")
        print(f"A: {res.answer}\n   retrieved={res.retrieved_sources} cited={res.cited_sources} "
              f"kb_no_answer={res.kb_no_answer}")
        d = asdict(res)
        for st in d["steps"]:
            if "report_markdown" in st["result"]:
                st["result"]["report_markdown"] = st["result"]["report_markdown"][:500] + "…"
        out.append({"question": q, **d})
    dst = OUTPUTS_DIR / "agent_demos" / "demo_transcripts.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsaved -> {dst}\nusage: {client.usage.summary(s)}")
    log_usage("run_agent_demo.py", {}, client.usage.summary(s))


if __name__ == "__main__":
    main()
