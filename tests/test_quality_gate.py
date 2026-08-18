"""Quality gate — pure logic, runs on CPU without torch."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hephaestus.quality_gate import evaluate_gate, assert_gate, GateResult


def test_pass_on_threshold():
    res = evaluate_gate({"accuracy": 0.96}, gate_accuracy=0.95)
    assert res.passed is True
    assert isinstance(res, GateResult)


def test_fail_below_threshold():
    res = evaluate_gate({"accuracy": 0.90}, gate_accuracy=0.95)
    assert res.passed is False


def test_f1_gate_stricter():
    # accuracy passes but f1 fails -> gate fails
    res = evaluate_gate({"accuracy": 0.97, "f1": 0.80}, gate_accuracy=0.95, gate_f1=0.95)
    assert res.passed is False


def test_f1_gate_both_pass():
    res = evaluate_gate({"accuracy": 0.97, "f1": 0.96}, gate_accuracy=0.95, gate_f1=0.95)
    assert res.passed is True


def test_assert_gate_raises_on_fail():
    import pytest
    with pytest.raises(AssertionError):
        assert_gate({"accuracy": 0.90}, gate_accuracy=0.95)


def test_assert_gate_ok_on_pass():
    res = assert_gate({"accuracy": 0.99}, gate_accuracy=0.95)
    assert res.passed is True
