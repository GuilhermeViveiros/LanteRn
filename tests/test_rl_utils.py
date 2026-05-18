"""Tests for RL utility functions (CPU, no model required)."""
from src.rl.utils import ANSWER_RE, extract_last_answer_from_text


def test_extract_answer_basic():
    text = "some reasoning <answer>A</answer>"
    assert extract_last_answer_from_text(text) == "A"


def test_extract_answer_last_only():
    text = "<answer>B</answer> then <answer>C</answer>"
    assert extract_last_answer_from_text(text) == "C"


def test_extract_answer_empty_when_missing():
    assert extract_last_answer_from_text("no tags here") == ""


def test_extract_answer_empty_string():
    assert extract_last_answer_from_text("") == ""


def test_extract_answer_strips_whitespace():
    text = "<answer>  D  </answer>"
    assert extract_last_answer_from_text(text) == "D"


def test_extract_answer_multiline():
    text = "<answer>\nsome answer\n</answer>"
    assert extract_last_answer_from_text(text) == "some answer"


def test_answer_re_case_insensitive():
    text = "<ANSWER>X</ANSWER>"
    matches = ANSWER_RE.findall(text)
    assert matches == ["X"]
