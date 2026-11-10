"""Agreement between YOUR manual tag labels and the model's tags.

1. Open data/eval/tag_labeling_sheet.csv in Excel (UTF-8 with BOM, Chinese shows fine).
2. For each review fill human_tag_1 (required) and human_tag_2 (optional) using ONLY
   these tags: 去屑 控油 香味 质地 起泡 洗后发质 头皮刺激/过敏 包装 物流 价格 真伪 客服 其他
3. Save as CSV, then run:
     .venv\\Scripts\\python scripts\\tag_agreement.py --version v2

Metrics (on rows you labelled):
  - exact_set_match: human tag set == model tag set
  - any_overlap:     at least one tag in common
  - mean_jaccard:    |A∩B| / |A∪B| averaged
  - per-tag precision / recall of the model, using your labels as reference
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import EVAL_RESULTS_DIR, TAG_SHEET_CSV, TAXONOMY  # noqa: E402


def _set(*vals) -> set[str]:
    out = set()
    for v in vals:
        if isinstance(v, str):
            out |= {t.strip() for t in v.replace("｜", "|").split("|") if t.strip()}
    return out


def agreement(sheet: pd.DataFrame, pred: pd.DataFrame) -> dict:
    m = sheet.merge(pred[["review_id", "tags"]], on="review_id", how="inner")
    m = m[m["human_tag_1"].fillna("").astype(str).str.strip() != ""]
    if m.empty:
        return {"n_labelled": 0, "message": "还没有人工标注，请先填写 human_tag_1。"}
    human = [_set(a, b) for a, b in zip(m["human_tag_1"], m["human_tag_2"])]
    model = [_set(t) for t in m["tags"]]
    bad = sorted({t for h in human for t in h if t not in TAXONOMY})
    exact = sum(h == p for h, p in zip(human, model))
    overlap = sum(bool(h & p) for h, p in zip(human, model))
    jac = sum(len(h & p) / len(h | p) for h, p in zip(human, model) if h | p)
    per_tag = {}
    for t in TAXONOMY:
        tp = sum(t in h and t in p for h, p in zip(human, model))
        fp = sum(t not in h and t in p for h, p in zip(human, model))
        fn = sum(t in h and t not in p for h, p in zip(human, model))
        if tp + fp + fn:
            per_tag[t] = {"precision": round(tp / (tp + fp), 3) if tp + fp else None,
                          "recall": round(tp / (tp + fn), 3) if tp + fn else None,
                          "human_count": tp + fn, "model_count": tp + fp}
    n = len(m)
    return {"n_labelled": n, "exact_set_match": round(exact / n, 4),
            "any_overlap": round(overlap / n, 4), "mean_jaccard": round(jac / n, 4),
            "per_tag": per_tag, "invalid_human_tags": bad}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v2")
    ap.add_argument("--sheet", default=str(TAG_SHEET_CSV))
    a = ap.parse_args()
    sheet = pd.read_csv(a.sheet, encoding="utf-8-sig", dtype={"human_tag_1": str, "human_tag_2": str})
    pred = pd.read_csv(EVAL_RESULTS_DIR / a.version / "predictions.csv", encoding="utf-8-sig")
    res = agreement(sheet, pred)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    if res["n_labelled"]:
        path = EVAL_RESULTS_DIR / a.version / "tag_agreement.json"
        path.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved -> {path}")


if __name__ == "__main__":
    main()
