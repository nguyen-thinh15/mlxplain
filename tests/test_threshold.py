"""Tests for threshold classification logic."""

from mlxplain.core.threshold import classify


def test_classify_above_threshold():
    assert classify(0.7, 0.5) == "Positive"


def test_classify_below_threshold():
    assert classify(0.3, 0.5) == "Negative"


def test_classify_at_threshold():
    """At the threshold, should classify as positive (>= boundary)."""
    assert classify(0.5, 0.5) == "Positive"


def test_classify_custom_labels():
    assert classify(0.8, 0.5, positive_label="Declined", negative_label="Approved") == "Declined"
    assert classify(0.2, 0.5, positive_label="Declined", negative_label="Approved") == "Approved"


def test_classify_custom_threshold():
    assert classify(0.6, 0.7) == "Negative"
    assert classify(0.8, 0.7) == "Positive"
