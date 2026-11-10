import json

import pytest
from pydantic import ValidationError

from src.schemas import ReviewLabel, validate_batch


def test_valid_label():
    lab = ReviewLabel(id=1, sentiment="negative", tags=["物流", "包装"])
    assert lab.tags == ["物流", "包装"]


@pytest.mark.parametrize("bad", [
    {"id": 1, "sentiment": "angry", "tags": ["物流"]},                    # bad sentiment
    {"id": 1, "sentiment": "negative", "tags": ["脱发"]},                  # tag outside taxonomy
    {"id": 1, "sentiment": "negative", "tags": []},                        # 0 tags
    {"id": 1, "sentiment": "negative", "tags": ["物流", "包装", "价格"]},  # 3 tags
    {"id": 1, "sentiment": "negative", "tags": ["物流", "物流"]},          # duplicate
])
def test_invalid_labels(bad):
    with pytest.raises(ValidationError):
        ReviewLabel.model_validate(bad)


def test_validate_batch_ok():
    raw = json.dumps({"results": [{"id": 1, "sentiment": "positive", "tags": ["香味"]},
                                  {"id": 2, "sentiment": "negative", "tags": ["物流"]}]})
    valid, errors = validate_batch(raw, [1, 2])
    assert set(valid) == {1, 2} and errors == []


def test_validate_batch_reports_missing_bad_and_unexpected():
    raw = json.dumps({"results": [{"id": 1, "sentiment": "positive", "tags": ["香味"]},
                                  {"id": 2, "sentiment": "positive", "tags": ["不存在"]},
                                  {"id": 99, "sentiment": "positive", "tags": ["香味"]}]})
    valid, errors = validate_batch(raw, [1, 2, 3])
    assert set(valid) == {1}
    joined = " ".join(errors)
    assert "id=2" in joined and "id=3: missing" in joined and "id=99: unexpected" in joined


def test_validate_batch_invalid_json():
    valid, errors = validate_batch("not json {", [1])
    assert valid == {} and "invalid JSON" in errors[0]


def test_validate_batch_wrong_shape():
    valid, errors = validate_batch(json.dumps([1, 2]), [1])
    assert valid == {} and "results" in errors[0]
