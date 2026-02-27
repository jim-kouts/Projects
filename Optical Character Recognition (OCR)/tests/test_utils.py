# tests/test_utils.py
# ====================
# Unit tests for utility functions.
# pytest finds and runs these automatically.
#
# Unit test: a small test that checks one specific function works correctly.
# We test functions that do NOT need the model or dataset loaded —
# just pure logic functions that we can verify with simple inputs.

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import pytest
from utils.config import ID2LABEL, LABEL2ID, NUM_LABELS, MAX_TOKEN_LENGTH


# ── Config tests ───────────────────────────────────────────────

def test_label_mapping_is_consistent():
    """
    ID2LABEL and LABEL2ID must be exact inverses of each other.
    If ID2LABEL has 7 entries, LABEL2ID must also have 7 entries
    and every key/value must map back correctly.
    """
    for id_, label in ID2LABEL.items():
        assert LABEL2ID[label] == id_, \
            f"Mismatch: ID2LABEL[{id_}]={label} but LABEL2ID[{label}]={LABEL2ID[label]}"


def test_num_labels_matches_id2label():
    """NUM_LABELS in config must match the actual number of labels defined."""
    assert NUM_LABELS == len(ID2LABEL), \
        f"NUM_LABELS={NUM_LABELS} but ID2LABEL has {len(ID2LABEL)} entries"


def test_expected_labels_exist():
    """All 7 expected BIO labels must be present in ID2LABEL."""
    expected = ["O", "B-HEADER", "I-HEADER", "B-QUESTION",
                "I-QUESTION", "B-ANSWER", "I-ANSWER"]
    for label in expected:
        assert label in LABEL2ID, f"Missing label: {label}"


def test_max_token_length_is_valid():
    """MAX_TOKEN_LENGTH must be a positive integer."""
    assert isinstance(MAX_TOKEN_LENGTH, int)
    assert MAX_TOKEN_LENGTH > 0


# ── normalize_bbox tests ───────────────────────────────────────

# Import the function directly from preprocessing
from src.preprocessing import normalize_bbox

def test_normalize_bbox_standard():
    """
    A box covering half the image width and full height
    should normalize to [500, 0, 1000, 1000] approximately.
    """
    result = normalize_bbox([500, 0, 1000, 1000], width=1000, height=1000)
    assert result == [500, 0, 1000, 1000]


def test_normalize_bbox_clamps_to_1000():
    """Values above image dimensions should be clamped to 1000."""
    result = normalize_bbox([0, 0, 1500, 1500], width=1000, height=1000)
    assert result[2] == 1000
    assert result[3] == 1000


def test_normalize_bbox_clamps_to_0():
    """Negative values should be clamped to 0."""
    result = normalize_bbox([-10, -10, 500, 500], width=1000, height=1000)
    assert result[0] == 0
    assert result[1] == 0


def test_normalize_bbox_returns_four_values():
    """Result must always be a list of exactly 4 integers."""
    result = normalize_bbox([100, 200, 300, 400], width=1000, height=1000)
    assert len(result) == 4
    assert all(isinstance(v, int) for v in result)


# ── build_structured_output tests ─────────────────────────────

from src.inference import build_structured_output

def test_structured_output_basic():
    """
    Given a simple question/answer pair,
    output should have one question and one answer.
    """
    pairs = [
        ("Name:", "B-QUESTION"),
        ("John",  "B-ANSWER"),
        ("Smith", "I-ANSWER"),
    ]
    result = build_structured_output(pairs)
    assert len(result["questions"]) == 1
    assert len(result["answers"]) == 1
    assert result["questions"][0] == "Name:"
    assert result["answers"][0] == "John Smith"


def test_structured_output_header():
    """Header entities should be collected correctly."""
    pairs = [
        ("Invoice", "B-HEADER"),
        ("Summary", "I-HEADER"),
    ]
    result = build_structured_output(pairs)
    assert len(result["headers"]) == 1
    assert result["headers"][0] == "Invoice Summary"


def test_structured_output_empty():
    """Empty input should return empty lists."""
    result = build_structured_output([])
    assert result["headers"]   == []
    assert result["questions"] == []
    assert result["answers"]   == []
    assert result["other"]     == []


def test_structured_output_only_other():
    """Words labeled O should go to other."""
    pairs = [("12345", "O"), ("noise", "O")]
    result = build_structured_output(pairs)
    assert result["questions"] == []
    assert result["answers"]   == []