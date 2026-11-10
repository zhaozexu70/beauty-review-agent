from src.config import KB_DIR
from src.retrieval import KBIndex, load_passages


def test_passages_have_sources():
    ps = load_passages()
    assert len(ps) >= 10
    assert {p.source for p in ps} >= {"shampoo_ingredients.md", "scalp_types.md",
                                      "after_sales_policy.md"}


def test_relevant_queries_hit_right_section():
    kb = KBIndex()
    cases = {"洗发水漏液了怎么办": "包装破损与漏液", "硅油有什么作用": "硅油",
             "油性头皮适合什么": "油性头皮", "怀疑是假货": "正品保障",
             "快递太慢了能赔吗": "物流问题"}
    for q, sec in cases.items():
        hits = kb.search(q)
        assert hits and sec in hits[0]["section"], (q, hits[:1])
        assert hits[0]["source"].endswith(".md")


def test_off_topic_returns_nothing():
    kb = KBIndex()
    assert kb.search("今天股票涨了吗") == []
    assert kb.search("") == []


def test_kb_marked_as_demo():
    for p in KB_DIR.glob("*.md"):
        assert "演示" in p.read_text(encoding="utf-8")
