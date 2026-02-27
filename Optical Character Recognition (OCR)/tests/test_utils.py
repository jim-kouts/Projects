# tests/test_utils.py
# ====================
# Unit tests for utility functions.
# pytest finds and runs these automatically.
#
# IMPORTANT: These tests are designed to run in CI without heavy ML packages.
# We do NOT import torch, transformers, or easyocr here.
# Instead we test pure logic functions that only need numpy and standard Python.

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


def test_all_label_ids_are_integers():
    """All keys in ID2LABEL must be integers."""
    for id_ in ID2LABEL.keys():
        assert isinstance(id_, int), f"Label ID {id_} is not an integer"


def test_all_label_names_are_strings():
    """All values in ID2LABEL must be strings."""
    for label in ID2LABEL.values():
        assert isinstance(label, str), f"Label name {label} is not a string"


def test_label_ids_are_sequential():
    """Label IDs must start from 0 and be sequential with no gaps."""
    ids = sorted(ID2LABEL.keys())
    assert ids == list(range(len(ids))), \
        f"Label IDs are not sequential: {ids}"


# ── normalize_bbox tests ───────────────────────────────────────
# We copy the function logic here directly to avoid importing
# preprocessing.py which imports torch at the top level

def normalize_bbox(bbox, width=1000, height=1000):
    """Local copy of normalize_bbox for testing without importing torch."""
    return [
        max(0, min(1000, int(1000 * bbox[0] / width))),
        max(0, min(1000, int(1000 * bbox[1] / height))),
        max(0, min(1000, int(1000 * bbox[2] / width))),
        max(0, min(1000, int(1000 * bbox[3] / height))),
    ]


def test_normalize_bbox_standard():
    """
    A box at half image width and full height
    should normalize to [500, 0, 1000, 1000].
    """
    result = normalize_bbox([500, 0, 1000, 1000], width=1000, height=1000)
    assert result == [500, 0, 1000, 1000]


def test_normalize_bbox_clamps_max_to_1000():
    """Values above image dimensions should be clamped to 1000."""
    result = normalize_bbox([0, 0, 1500, 1500], width=1000, height=1000)
    assert result[2] == 1000
    assert result[3] == 1000


def test_normalize_bbox_clamps_min_to_0():
    """Negative values should be clamped to 0."""
    result = normalize_bbox([-10, -10, 500, 500], width=1000, height=1000)
    assert result[0] == 0
    assert result[1] == 0


def test_normalize_bbox_returns_four_values():
    """Result must always be a list of exactly 4 integers."""
    result = normalize_bbox([100, 200, 300, 400], width=1000, height=1000)
    assert len(result) == 4
    assert all(isinstance(v, int) for v in result)


def test_normalize_bbox_small_image():
    """Test normalization with a non-square image size."""
    result = normalize_bbox([100, 50, 200, 100], width=400, height=200)
    assert result == [250, 250, 500, 500]


# ── build_structured_output tests ─────────────────────────────
# We copy the function logic here directly to avoid importing
# inference.py which imports torch, easyocr, transformers

def build_structured_output(word_label_pairs):
    """Local copy of build_structured_output for testing without importing torch."""
    result = {"headers": [], "questions": [], "answers": [], "other": []}
    current_entity = []
    current_type = None

    for word, label in word_label_pairs:
        if label.startswith("B-"):
            if current_entity and current_type:
                entity_text = " ".join(current_entity)
                if current_type == "HEADER":
                    result["headers"].append(entity_text)
                elif current_type == "QUESTION":
                    result["questions"].append(entity_text)
                elif current_type == "ANSWER":
                    result["answers"].append(entity_text)
                else:
                    result["other"].append(entity_text)
            current_entity = [word]
            current_type = label[2:]

        elif label.startswith("I-") and current_type:
            current_entity.append(word)

        else:
            if current_entity and current_type:
                entity_text = " ".join(current_entity)
                if current_type == "HEADER":
                    result["headers"].append(entity_text)
                elif current_type == "QUESTION":
                    result["questions"].append(entity_text)
                elif current_type == "ANSWER":
                    result["answers"].append(entity_text)
                else:
                    result["other"].append(entity_text)
            current_entity = []
            current_type = None

    if current_entity and current_type:
        entity_text = " ".join(current_entity)
        if current_type == "HEADER":
            result["headers"].append(entity_text)
        elif current_type == "QUESTION":
            result["questions"].append(entity_text)
        elif current_type == "ANSWER":
            result["answers"].append(entity_text)

    return result


def test_structured_output_basic():
    """Given a simple question/answer pair, output should have one of each."""
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
    """Header entities should be collected and joined correctly."""
    pairs = [
        ("Invoice", "B-HEADER"),
        ("Summary", "I-HEADER"),
    ]
    result = build_structured_output(pairs)
    assert len(result["headers"]) == 1
    assert result["headers"][0] == "Invoice Summary"


def test_structured_output_empty():
    """Empty input should return empty lists for all fields."""
    result = build_structured_output([])
    assert result["headers"]   == []
    assert result["questions"] == []
    assert result["answers"]   == []
    assert result["other"]     == []


def test_structured_output_only_other():
    """Words labeled O should not appear in any named field."""
    pairs = [("12345", "O"), ("noise", "O")]
    result = build_structured_output(pairs)
    assert result["questions"] == []
    assert result["answers"]   == []
    assert result["headers"]   == []


def test_structured_output_multiple_questions():
    """Multiple separate questions should each be a separate entry."""
    pairs = [
        ("Date:",    "B-QUESTION"),
        ("Name:",    "B-QUESTION"),
        ("Address:", "B-QUESTION"),
    ]
    result = build_structured_output(pairs)
    assert len(result["questions"]) == 3


def test_structured_output_question_answer_pairs():
    """Multiple question/answer pairs should all be captured."""
    pairs = [
        ("Date:",  "B-QUESTION"),
        ("2024",   "B-ANSWER"),
        ("Name:",  "B-QUESTION"),
        ("John",   "B-ANSWER"),
    ]
    result = build_structured_output(pairs)
    assert len(result["questions"]) == 2
    assert len(result["answers"])   == 2