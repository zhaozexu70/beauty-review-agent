import json

import pandas as pd

from scripts.evaluate import compute_metrics, predictions_frame, stability, write_comparison
from src.schemas import ClassifiedReview


def _ev():
    return pd.DataFrame({"review_id": [1, 2, 3, 4], "label": [1, 1, 0, 0],
                         "review": ["好", "还没用", "差", "假货"]})


def _labels(**over):
    base = {1: ClassifiedReview(id=1, sentiment="positive", tags=["其他"]),
            2: ClassifiedReview(id=2, sentiment="neutral", tags=["其他"]),
            3: ClassifiedReview(id=3, sentiment="negative", tags=["物流"]),
            4: ClassifiedReview(id=4, failed=True, error="x")}
    base.update(over)
    return base


def test_neutral_and_failed_count_as_wrong():
    p = predictions_frame(_ev(), _labels())
    m, cm = compute_metrics(p)
    assert m["accuracy"] == 0.5                  # 2 of 4 correct
    assert m["neutral_rate"] == 0.25 and m["json_failure_rate"] == 0.25
    assert m["accuracy_on_non_neutral"] == 1.0 and m["n_non_neutral"] == 2
    assert cm.loc["positive", "neutral"] == 1 and cm.loc["negative", "failed"] == 1
    # recall: positive 1/2, negative 1/2; precision 1/1 each -> F1 = 2/3 each
    assert abs(m["macro_f1"] - 2 / 3) < 1e-3


def test_stability_requires_same_sentiment_and_tag_set():
    p1 = predictions_frame(_ev(), _labels())
    p2 = predictions_frame(_ev(), _labels())
    assert stability(p1, p2)["identical_sentiment_and_tags"] == 1.0
    changed = _labels()
    changed[3] = ClassifiedReview(id=3, sentiment="negative", tags=["包装"])  # tag differs
    s = stability(p1, predictions_frame(_ev(), changed))
    assert s["identical_sentiment"] == 1.0 and s["identical_tags"] == 0.75
    assert s["identical_sentiment_and_tags"] == 0.75


def test_comparison_written_into_its_own_folder(tmp_path):
    for base, acc in [(tmp_path, 0.9), (tmp_path / "holdout", 0.8)]:
        d = base / "v1"
        d.mkdir(parents=True)
        (d / "metrics.json").write_text(json.dumps({
            "version": "v1", "accuracy": acc, "macro_f1": acc, "neutral_rate": 0,
            "json_failure_rate": 0, "accuracy_on_non_neutral": acc, "stability": None,
            "usage_run1": {"estimated_cost": 0}}), encoding="utf-8")
    write_comparison(tmp_path)
    write_comparison(tmp_path / "holdout")
    main = pd.read_csv(tmp_path / "comparison.csv", encoding="utf-8-sig")
    ho = pd.read_csv(tmp_path / "holdout" / "comparison.csv", encoding="utf-8-sig")
    assert main["accuracy"].tolist() == [0.9] and ho["accuracy"].tolist() == [0.8]
