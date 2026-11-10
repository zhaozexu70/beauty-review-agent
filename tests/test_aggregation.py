import pandas as pd

from src.aggregation import aggregate, compact_stats, to_frame
from src.report import build_report
from src.schemas import ClassifiedReview


def _data():
    reviews = pd.DataFrame({"review_id": [1, 2, 3, 4, 5],
                            "review": ["物流太慢了，盒子也破了", "味道很好闻", "假货，用了头皮痒",
                                       "快递三天才到", "一般般"]})
    labels = {
        1: ClassifiedReview(id=1, sentiment="negative", tags=["物流", "包装"]),
        2: ClassifiedReview(id=2, sentiment="positive", tags=["香味"]),
        3: ClassifiedReview(id=3, sentiment="negative", tags=["真伪", "头皮刺激/过敏"]),
        4: ClassifiedReview(id=4, sentiment="negative", tags=["物流"]),
        5: ClassifiedReview(id=5, failed=True, error="x"),
    }
    return to_frame(reviews, labels)


def test_counts_and_shares():
    st = aggregate(_data())
    assert st["n_total"] == 5 and st["n_classified"] == 4 and st["n_failed"] == 1
    assert st["sentiment_counts"] == {"positive": 1, "neutral": 0, "negative": 3}
    assert st["sentiment_shares"]["negative"] == 0.75
    assert st["tag_counts"]["物流"] == 2 and st["tag_counts"]["香味"] == 1
    assert st["tag_by_sentiment"]["物流"]["negative"] == 2


def test_top_negative_tags_and_quotes():
    st = aggregate(_data())
    top = st["negative_tags"][0]
    assert top["tag"] == "物流" and top["count"] == 2
    assert top["share_of_negative"] == round(2 / 3, 4)
    assert {q["review_id"] for q in top["quotes"]} == {1, 4}
    assert st["top_problems"][0]["tag"] == "物流"
    assert all(x["tag"] != "其他" for x in st["top_problems"])


def test_compact_and_report_use_code_numbers():
    st = aggregate(_data())
    cs = compact_stats(st)
    assert cs["top_negative_tags"][0]["tag"] == "物流"
    md = build_report(st, client=None, source="unit")
    assert "rule-based fallback" in md and "66.7%" in md and "物流太慢了" in md
    for section in ["核心数据", "关键发现", "Top 3 问题", "改进建议", "数据说明与局限"]:
        assert section in md
