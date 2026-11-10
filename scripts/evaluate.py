"""Evaluate one prompt version on the fixed 300-review eval set.

Reports (saved to outputs/eval_results/<version>/):
  - sentiment accuracy & macro-F1 vs human labels (neutral = its own class, i.e. always
    counted wrong because gold labels are binary) + neutral rate
  - confusion matrix (true: positive/negative  x  pred: positive/neutral/negative/failed)
  - JSON failure rate (items still invalid after one re-ask)
  - stability (--stability): run the prompt twice with the cache DISABLED and report the
    % of reviews whose sentiment and tags are identical across both runs

Usage (Windows):
  .venv\\Scripts\\python scripts\\evaluate.py --version v1 --stability
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.classifier import classify_reviews  # noqa: E402
from src.config import EVAL_CSV, EVAL_RESULTS_DIR, get_settings, has_api_key  # noqa: E402
from src.llm_client import LLMClient, log_usage  # noqa: E402
from src.prompts import PROMPTS  # noqa: E402

GOLD = {1: "positive", 0: "negative"}
PRED_COLS = ["positive", "neutral", "negative", "failed"]


def predictions_frame(ev: pd.DataFrame, labels: dict) -> pd.DataFrame:
    rows = []
    for r in ev.itertuples():
        c = labels[int(r.review_id)]
        pred = "failed" if c.failed else c.sentiment
        rows.append({"review_id": r.review_id, "label": r.label, "gold": GOLD[r.label],
                     "pred": pred, "tags": "|".join(c.tags), "reason": c.reason or "",
                     "failed": c.failed, "correct": pred == GOLD[r.label], "review": r.review})
    return pd.DataFrame(rows)


def compute_metrics(p: pd.DataFrame) -> dict:
    n = len(p)
    acc = float(p["correct"].mean())
    macro = float(f1_score(p["gold"], p["pred"], labels=["positive", "negative"],
                           average="macro", zero_division=0))
    prec, rec, f1, sup = precision_recall_fscore_support(
        p["gold"], p["pred"], labels=["positive", "negative"], zero_division=0)
    decided = p[p["pred"].isin(["positive", "negative"])]
    cm = pd.crosstab(p["gold"], p["pred"]).reindex(index=["positive", "negative"],
                                                   columns=PRED_COLS, fill_value=0)
    return {
        "n": n,
        "accuracy": round(acc, 4),
        "macro_f1": round(macro, 4),
        "neutral_rate": round(float((p["pred"] == "neutral").mean()), 4),
        "json_failure_rate": round(float(p["failed"].mean()), 4),
        "accuracy_on_non_neutral": round(float(decided["correct"].mean()), 4) if len(decided) else None,
        "n_non_neutral": int(len(decided)),
        "per_class": {c: {"precision": round(float(prec[i]), 4), "recall": round(float(rec[i]), 4),
                          "f1": round(float(f1[i]), 4), "support": int(sup[i])}
                      for i, c in enumerate(["positive", "negative"])},
        "confusion_matrix": {"rows_true": ["positive", "negative"], "cols_pred": PRED_COLS,
                             "values": cm.values.tolist()},
    }, cm


def stability(p1: pd.DataFrame, p2: pd.DataFrame) -> dict:
    m = p1.merge(p2, on="review_id", suffixes=("_1", "_2"))
    tags_same = m.apply(lambda r: set(r.tags_1.split("|")) == set(r.tags_2.split("|")), axis=1)
    sent_same = m["pred_1"] == m["pred_2"]
    return {"n": len(m),
            "identical_sentiment_and_tags": round(float((sent_same & tags_same).mean()), 4),
            "identical_sentiment": round(float(sent_same.mean()), 4),
            "identical_tags": round(float(tags_same.mean()), 4)}


def write_comparison() -> None:
    rows = []
    for d in sorted(EVAL_RESULTS_DIR.glob("*/metrics.json")):
        m = json.loads(d.read_text(encoding="utf-8"))
        st = m.get("stability") or {}
        rows.append({"version": m["version"], "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
                     "neutral_rate": m["neutral_rate"], "json_failure_rate": m["json_failure_rate"],
                     "accuracy_on_non_neutral": m["accuracy_on_non_neutral"],
                     "stability_identical": st.get("identical_sentiment_and_tags"),
                     "api_cost": m["usage_run1"]["estimated_cost"]})
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(EVAL_RESULTS_DIR / "comparison.csv", index=False, encoding="utf-8-sig")
        print("\n=== comparison ===\n" + df.to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True, choices=sorted(PROMPTS))
    ap.add_argument("--stability", action="store_true",
                    help="run twice with cache disabled and measure agreement")
    args = ap.parse_args()
    if not has_api_key():
        sys.exit("LLM_API_KEY is not set in .env")

    s = get_settings()
    ev = pd.read_csv(EVAL_CSV, encoding="utf-8-sig")
    items = [(int(r.review_id), str(r.review)) for r in ev.itertuples()]
    out = EVAL_RESULTS_DIR / args.version
    out.mkdir(parents=True, exist_ok=True)

    # run 1: if --stability, bypass cache reads (fresh API call) but refresh the cache,
    # so demo mode later shows exactly these predictions.
    c1 = LLMClient(s, read_cache=not args.stability, write_cache=True)
    labels1, st1 = classify_reviews(c1, items, args.version)
    p1 = predictions_frame(ev, labels1)
    metrics, cm = compute_metrics(p1)
    metrics.update({"version": args.version, "model": s.model,
                    "time": datetime.now().isoformat(timespec="seconds"),
                    "batches": st1.batches, "reasked_batches": st1.reasks,
                    "usage_run1": c1.usage.summary(s), "stability": None})
    log_usage("evaluate.py", {"version": args.version, "run": 1}, c1.usage.summary(s))

    if args.stability:
        c2 = LLMClient(s, read_cache=False, write_cache=False)
        labels2, _ = classify_reviews(c2, items, args.version)
        p2 = predictions_frame(ev, labels2)
        p2.to_csv(out / "predictions_run2.csv", index=False, encoding="utf-8-sig")
        metrics["stability"] = stability(p1, p2)
        metrics["usage_run2"] = c2.usage.summary(s)
        log_usage("evaluate.py", {"version": args.version, "run": 2}, c2.usage.summary(s))

    p1.to_csv(out / "predictions.csv", index=False, encoding="utf-8-sig")
    p1[~p1["correct"]].to_csv(out / "badcases.csv", index=False, encoding="utf-8-sig")
    cm.to_csv(out / "confusion_matrix.csv", encoding="utf-8-sig")
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2),
                                      encoding="utf-8")

    print(f"=== {args.version} ({s.model}) ===")
    for k in ["accuracy", "macro_f1", "neutral_rate", "json_failure_rate", "accuracy_on_non_neutral"]:
        print(f"{k:>26}: {metrics[k]}")
    print("confusion matrix (rows=gold, cols=pred):\n" + cm.to_string())
    if metrics["stability"]:
        print("stability:", metrics["stability"])
    print("usage run1:", metrics["usage_run1"])
    if "usage_run2" in metrics:
        print("usage run2:", metrics["usage_run2"])
    write_comparison()


if __name__ == "__main__":
    main()
