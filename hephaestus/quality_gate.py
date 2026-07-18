"""Quality gate enforcement — pure logic, no torch dependency.

Separated from evaluator.py so CI can enforce the gate on CPU (no GPU) and
the training CLI can exit non-zero when the gate fails (instead of only
printing). The 80/95% thresholds previously lived only in narrative; now they
are real and testable.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class GateResult:
    passed: bool
    metric: str
    actual: float
    threshold: float
    detail: str = ""

    def __str__(self) -> str:
        state = "PASS" if self.passed else "FAIL"
        return (
            f"Quality Gate ({self.metric} >= {self.threshold:.0%}): {state} "
            f"(actual={self.actual:.4f})"
        )


def evaluate_gate(
    metrics: dict,
    gate_accuracy: float = 0.95,
    gate_f1: Optional[float] = None,
) -> GateResult:
    """Evaluate the primary quality gate on a metrics dict.

    `metrics` is expected to carry `accuracy` and optionally `f1` as 0..1 floats
    (the same shape produced by evaluator._evaluate_multiclass / _compile_metrics).

    Returns a GateResult. Does NOT raise — callers decide whether to fail the
    build (see `assert_gate`).
    """
    acc = float(metrics.get("accuracy", 0.0))
    result = GateResult(
        passed=acc >= gate_accuracy,
        metric="accuracy",
        actual=acc,
        threshold=gate_accuracy,
    )
    if gate_f1 is not None:
        f1 = float(metrics.get("f1", 0.0))
        result.detail = f"f1 gate (>= {gate_f1:.0%}): {'PASS' if f1 >= gate_f1 else 'FAIL'} (f1={f1:.4f})"
        if f1 < gate_f1:
            result.passed = False
    return result


def assert_gate(metrics: dict, gate_accuracy: float = 0.95, gate_f1: Optional[float] = None) -> GateResult:
    """Evaluate the gate and raise AssertionError if it fails.

    Used by the training CLI so a failed gate produces a non-zero exit code
    (fail the build / fail the release) rather than a silent print.
    """
    res = evaluate_gate(metrics, gate_accuracy=gate_accuracy, gate_f1=gate_f1)
    if not res.passed:
        raise AssertionError(f"QUALITY GATE FAILED: {res} {res.detail}".strip())
    return res
