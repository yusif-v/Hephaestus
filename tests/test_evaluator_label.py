"""Evaluator label extraction — pure regex logic, CPU-safe."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hephaestus.evaluator import _extract_label


def test_extract_critical_before_medium():
    # Ensure CRITICAL is matched preferentially over MEDIUM (length-ordered)
    out = _extract_label("This CVE is CRITICAL. The risk is medium-high.", ["CRITICAL", "HIGH", "MEDIUM", "LOW"])
    assert out == "CRITICAL"


def test_extract_low():
    out = _extract_label("Severity: LOW impact.", ["CRITICAL", "HIGH", "MEDIUM", "LOW"])
    assert out == "LOW"


def test_extract_fallback_first_class():
    out = _extract_label("no severity keyword here", ["CRITICAL", "HIGH", "MEDIUM", "LOW"])
    assert out == "CRITICAL"
