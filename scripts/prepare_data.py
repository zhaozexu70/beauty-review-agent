"""Download (if needed) and clean the online_shopping_10_cats dataset.

Steps (each step's row count is recorded in outputs/data_quality.csv):
  1. read raw CSV from the zip
  2. keep cat == "洗发水"
  3. drop empty / missing reviews (after stripping whitespace)
  4. drop texts that appear with BOTH labels (conflicting, ambiguous)
  5. drop exact duplicate reviews (keep first)
  6. drop near-duplicates (same text ignoring whitespace/punctuation)
Then build a fixed eval set: 150 positive + 150 negative, random_state=42,
and a 50-review tag labeling sheet drawn from the eval set.

Usage (Windows):  .venv\\Scripts\\python scripts\\prepare_data.py
"""
from __future__ import annotations

import re
import sys
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (  # noqa: E402
    CLEAN_CSV, EVAL_CSV, OUTPUTS_DIR, PROCESSED_DIR, RAW_DIR, RAW_URL, RAW_ZIP,
    TAG_SHEET_CSV, TAXONOMY,
)

SEED = 42


def download() -> None:
    if RAW_ZIP.exists():
        print(f"[skip] raw file exists: {RAW_ZIP}")
        return
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[download] {RAW_URL}")
    try:
        urllib.request.urlretrieve(RAW_URL, RAW_ZIP)
    except Exception as e:  # stop loudly; never fabricate data
        sys.exit(f"Download failed: {e}. Please download manually into {RAW_DIR}.")


def clean() -> tuple[pd.DataFrame, pd.DataFrame]:
    steps = []
    df = pd.read_csv(RAW_ZIP, encoding="utf-8")
    steps.append(("原始数据（全部10个品类）", len(df)))

    df = df[df["cat"] == "洗发水"].copy()
    steps.append(("仅保留 cat == 洗发水", len(df)))

    df["review"] = df["review"].astype("string").str.strip()
    df = df[df["review"].notna() & (df["review"] != "")]
    steps.append(("去除空评论（strip后为空/缺失）", len(df)))

    n_labels = df.groupby("review")["label"].nunique()
    conflicting = set(n_labels[n_labels > 1].index)
    df = df[~df["review"].isin(conflicting)]
    steps.append((f"去除标签冲突的重复文本（{len(conflicting)}种文本）", len(df)))

    df = df.drop_duplicates(subset="review", keep="first")
    steps.append(("去除完全重复的评论（保留首条）", len(df)))

    # near-duplicates: identical after removing whitespace & punctuation
    norm = df["review"].map(lambda t: re.sub(r"[\s\W_]+", "", str(t)))
    df = df[~norm.duplicated(keep="first")]
    steps.append(("去除近似重复（忽略空格/标点后相同，保留首条）", len(df)))

    df = df.reset_index(drop=True)
    df.insert(0, "review_id", range(len(df)))
    df["label"] = df["label"].astype(int)

    dq = pd.DataFrame(steps, columns=["step", "rows"])
    dq["removed"] = (-dq["rows"].diff()).fillna(0).astype(int)
    pos, neg = int((df.label == 1).sum()), int((df.label == 0).sum())
    dq.loc[len(dq)] = [f"清洗后标签分布：正向 {pos} / 负向 {neg}", len(df), 0]
    return df, dq


def build_eval(df: pd.DataFrame) -> pd.DataFrame:
    pos = df[df.label == 1].sample(n=150, random_state=SEED)
    neg = df[df.label == 0].sample(n=150, random_state=SEED)
    ev = pd.concat([pos, neg]).sample(frac=1, random_state=SEED)  # shuffle
    return ev[["review_id", "label", "review"]].reset_index(drop=True)


def build_tag_sheet(ev: pd.DataFrame) -> pd.DataFrame:
    sheet = ev.sample(n=50, random_state=SEED)[["review_id", "review"]].copy()
    sheet["human_tag_1"] = ""
    sheet["human_tag_2"] = ""
    sheet["notes"] = ""
    return sheet.sort_values("review_id").reset_index(drop=True)


def main() -> None:
    download()
    df, dq = clean()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_CSV, index=False, encoding="utf-8-sig")

    EVAL_CSV.parent.mkdir(parents=True, exist_ok=True)
    ev = build_eval(df)
    ev.to_csv(EVAL_CSV, index=False, encoding="utf-8-sig")

    if TAG_SHEET_CSV.exists():
        print(f"[skip] {TAG_SHEET_CSV} already exists (won't overwrite your labels)")
    else:
        build_tag_sheet(ev).to_csv(TAG_SHEET_CSV, index=False, encoding="utf-8-sig")

    OUTPUTS_DIR.mkdir(exist_ok=True)
    dq.to_csv(OUTPUTS_DIR / "data_quality.csv", index=False, encoding="utf-8-sig")
    md = ["| 步骤 | 剩余行数 | 本步移除 |", "|---|---:|---:|"]
    md += [f"| {r.step} | {r.rows} | {r.removed} |" for r in dq.itertuples()]
    (OUTPUTS_DIR / "data_quality.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print("\n".join(md))
    print(f"\nclean -> {CLEAN_CSV} ({len(df)} rows)")
    print(f"eval  -> {EVAL_CSV} ({len(ev)} rows, pos={int(ev.label.sum())})")
    print(f"sheet -> {TAG_SHEET_CSV}  (taxonomy: {' / '.join(TAXONOMY)})")


if __name__ == "__main__":
    main()
