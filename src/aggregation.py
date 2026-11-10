"""Pure-pandas aggregation of classified reviews (no LLM, fully testable)."""
from __future__ import annotations

import pandas as pd

from src.config import SENTIMENTS, TAXONOMY


def to_frame(reviews: pd.DataFrame, labels: dict) -> pd.DataFrame:
    """Join reviews (review_id, review[, label]) with ClassifiedReview objects."""
    rows = []
    for r in reviews.itertuples(index=False):
        c = labels.get(int(r.review_id))
        rows.append({
            "review_id": int(r.review_id), "review": r.review,
            "label": getattr(r, "label", None),
            "sentiment": c.sentiment if c and not c.failed else None,
            "tags": list(c.tags) if c and not c.failed else [],
            "reason": c.reason if c else None,
            "failed": bool(c.failed) if c else True,
        })
    return pd.DataFrame(rows)


def _quotes(df: pd.DataFrame, n: int = 3, max_len: int = 80) -> list[dict]:
    """Deterministic example quotes: prefer readable lengths, then by review_id."""
    d = df.assign(_len=df["review"].str.len())
    d = d.assign(_ok=d["_len"].between(8, max_len)).sort_values(["_ok", "review_id"],
                                                                ascending=[False, True])
    return [{"review_id": int(r.review_id), "text": str(r.review)[:max_len * 2]}
            for r in d.head(n).itertuples()]


def aggregate(df: pd.DataFrame, n_quotes: int = 3) -> dict:
    total = len(df)
    ok = df[~df["failed"]]
    n = len(ok)

    def share(x: int, base: int) -> float:
        return round(x / base, 4) if base else 0.0

    sent_counts = {s: int((ok["sentiment"] == s).sum()) for s in SENTIMENTS}
    exploded = ok.explode("tags").dropna(subset=["tags"])
    tag_counts = {t: int((exploded["tags"] == t).sum()) for t in TAXONOMY}

    tag_by_sent = {
        t: {s: int(((exploded["tags"] == t) & (exploded["sentiment"] == s)).sum()) for s in SENTIMENTS}
        for t in TAXONOMY
    }

    neg = ok[ok["sentiment"] == "negative"]
    n_neg = len(neg)
    neg_tags = []
    for t in TAXONOMY:
        sub = neg[neg["tags"].map(lambda ts, t=t: t in ts)]
        if len(sub) == 0:
            continue
        neg_tags.append({
            "tag": t, "count": len(sub),
            "share_of_negative": share(len(sub), n_neg),
            "share_of_all": share(len(sub), n),
            "quotes": _quotes(sub, n_quotes),
        })
    neg_tags.sort(key=lambda x: (-x["count"], TAXONOMY.index(x["tag"])))
    top_problems = [x for x in neg_tags if x["tag"] != "其他"][:3]

    return {
        "n_total": total, "n_classified": n, "n_failed": total - n,
        "sentiment_counts": sent_counts,
        "sentiment_shares": {s: share(c, n) for s, c in sent_counts.items()},
        "tag_counts": tag_counts,
        "tag_shares": {t: share(c, n) for t, c in tag_counts.items()},
        "tag_by_sentiment": tag_by_sent,
        "n_negative": n_neg,
        "negative_tags": neg_tags,
        "top_problems": top_problems,
        "note": "一条评论最多2个标签，因此各标签占比之和可能超过100%。",
    }


def compact_stats(stats: dict) -> dict:
    """Smaller version of stats to send back to the LLM (saves tokens)."""
    return {
        "n_total": stats["n_total"], "n_classified": stats["n_classified"],
        "n_failed": stats["n_failed"],
        "sentiment_counts": stats["sentiment_counts"],
        "sentiment_shares": stats["sentiment_shares"],
        "tag_counts": {t: c for t, c in stats["tag_counts"].items() if c},
        "n_negative": stats["n_negative"],
        "top_negative_tags": [
            {k: x[k] for k in ("tag", "count", "share_of_negative", "share_of_all")}
            | {"quotes": [q["text"] for q in x["quotes"][:2]]}
            for x in stats["negative_tags"][:5]
        ],
        "note": stats["note"],
    }
