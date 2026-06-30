"""Evaluation — inference + metrics + quality gates. Generic for any task."""

import re
from collections import defaultdict


def evaluate_model(model, tokenizer, test_data, config):
    """Evaluate model on test set. Returns metrics dict.

    Supports:
    - binary classification (ANOMALY/NORMAL, PHISHING/LEGITIMATE)
    - multiclass classification (CRITICAL/HIGH/MEDIUM/LOW)
    - Generic string matching (exact normalized match)
    """
    model.eval()
    task_type = config.evaluation.task_type

    if task_type == "binary" or task_type == "per_class":
        return _evaluate_binary(model, tokenizer, test_data, config)
    else:
        return _evaluate_multiclass(model, tokenizer, test_data, config)


def _extract_label(text: str, classes: list) -> str:
    """Extract predicted/class label from model response text.

    Looks for class names in order of priority (longest first to avoid partial matches).
    """
    text_upper = text.upper().strip()
    # Sort by length descending so CRITICAL matches before MEDIUM
    sorted_classes = sorted(classes, key=len, reverse=True)

    for cls in sorted_classes:
        # Match as standalone word/label
        pattern = r'\b' + re.escape(cls.upper()) + r'\b'
        if re.search(pattern, text_upper):
            return cls.upper()

    # Fallback: return first class as default
    return classes[0].upper()


def _evaluate_binary(model, tokenizer, test_data, config):
    """Evaluate binary classification (e.g., ANOMALY/NORMAL)."""
    max_samples = min(config.evaluation.max_test_samples, len(test_data))
    correct = tp = fp = fn = tn = 0

    # Determine the positive class labels
    classes = config.evaluation.classes or ["ANOMALY", "NORMAL"]
    positive = classes[0].upper()  # ANOMALY
    negative = classes[1].upper() if len(classes) > 1 else "NORMAL"

    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)

    for i, item in enumerate(test_data[:max_samples]):
        if "messages" in item:
            messages = item["messages"]
        else:
            messages = item

        # Find ground truth (last assistant message)
        expected = None
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                expected = msg["content"]
                break
        if not expected:
            continue

        expected_label = _extract_label(expected, classes)
        is_positive_ground = (expected_label == positive)

        # Build prompt from all messages except last
        prompt = _build_prompt(messages[:-1], tokenizer)

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        import torch
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=64, do_sample=False)

        resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        predicted_label = _extract_label(resp, classes)
        is_positive_pred = (predicted_label == positive)

        # Track per-class
        per_class_total[expected_label] += 1
        if expected_label == predicted_label:
            per_class_correct[expected_label] += 1
            correct += 1

        if is_positive_pred and is_positive_ground:
            tp += 1
        elif is_positive_pred and not is_positive_ground:
            fp += 1
        elif not is_positive_pred and is_positive_ground:
            fn += 1
        else:
            tn += 1

    return _compile_metrics(correct, tp, fp, fn, tn, max_samples, per_class_correct, per_class_total, config)


def _evaluate_multiclass(model, tokenizer, test_data, config):
    """Evaluate multiclass classification (e.g., CRITICAL/HIGH/MEDIUM/LOW)."""
    max_samples = min(config.evaluation.max_test_samples, len(test_data))
    classes = [c.upper() for c in (config.evaluation.classes or ["CRITICAL", "HIGH", "MEDIUM", "LOW"])]

    correct = 0
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)

    # Per-class TP/FP/FN for multiclass precision/recall
    per_class_tp = defaultdict(int)
    per_class_fp = defaultdict(int)
    per_class_fn = defaultdict(int)

    for i, item in enumerate(test_data[:max_samples]):
        if "messages" in item:
            messages = item["messages"]
        else:
            messages = item

        expected = None
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                expected = msg["content"]
                break
        if not expected:
            continue

        expected_label = _extract_label(expected, classes)
        per_class_total[expected_label] += 1

        prompt = _build_prompt(messages[:-1], tokenizer)

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        import torch
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=64, do_sample=False)

        resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        predicted_label = _extract_label(resp, classes)

        if expected_label == predicted_label:
            correct += 1
            per_class_correct[expected_label] += 1
            per_class_tp[predicted_label] += 1
        else:
            per_class_fn[expected_label] += 1
            per_class_fp[predicted_label] += 1

    n = max_samples

    # Macro-averaged metrics
    precisions = []
    recalls = []
    f1s = []

    for cls in classes:
        tp_c = per_class_tp[cls]
        fp_c = per_class_fp[cls]
        fn_c = per_class_fn[cls]
        p = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0
        r = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0
        f1_c = 2 * p * r / (p + r) if (p + r) > 0 else 0
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1_c)

    accuracy = correct / n if n else 0
    macro_precision = sum(precisions) / len(precisions) if precisions else 0
    macro_recall = sum(recalls) / len(recalls) if recalls else 0
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0

    results = {
        "accuracy": round(accuracy, 4),
        "precision": round(macro_precision, 4),
        "recall": round(macro_recall, 4),
        "f1": round(macro_f1, 4),
        "total_samples": n,
        "task_type": "multiclass",
    }

    # Add per-class metrics
    per_class_metrics = {}
    for cls in classes:
        acc = per_class_correct[cls] / per_class_total[cls] if per_class_total[cls] > 0 else 0
        per_class_metrics[cls] = {
            "samples": per_class_total[cls],
            "accuracy": round(acc, 4),
            "tp": per_class_tp[cls],
            "fp": per_class_fp[cls],
            "fn": per_class_fn[cls],
        }
    results["per_class"] = per_class_metrics

    _print_results(results, config)
    return results


def _compile_metrics(correct, tp, fp, fn, tn, n, per_class_correct, per_class_total, config):
    """Compile binary metrics and print results."""
    accuracy = correct / n if n else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    results = {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "total_samples": n,
        "task_type": "binary",
    }

    # Per-class breakdown
    if per_class_total:
        per_class_metrics = {}
        for cls, total in per_class_total.items():
            acc = per_class_correct[cls] / total if total > 0 else 0
            per_class_metrics[cls] = {
                "samples": total,
                "accuracy": round(acc, 4),
                "correct": per_class_correct[cls],
            }
        results["per_class"] = per_class_metrics

    _print_results(results, config)
    return results


def _print_results(results, config):
    """Print evaluation results."""
    task_type = config.evaluation.task_type
    print(f"Accuracy: {results['accuracy']*100:.1f}% | F1: {results['f1']*100:.1f}%")
    print(f"Precision: {results['precision']*100:.1f}% | Recall: {results['recall']*100:.1f}%")

    if task_type in ("binary", "per_class"):
        tp = results.get('tp', 0)
        fp = results.get('fp', 0)
        fn = results.get('fn', 0)
        tn = results.get('tn', 0)
        print(f"TP={tp} FP={fp} FN={fn} TN={tn}")

    # Per-class breakdown
    if "per_class" in results:
        print("\nPer-class breakdown:")
        for cls, metrics in results["per_class"].items():
            print(f"  {cls}: {metrics['accuracy']*100:.1f}% ({metrics.get('correct', metrics.get('tp',0))}/{metrics['samples']})")

    passed = results["accuracy"] >= config.evaluation.quality_gate
    print(f"\nQuality Gate ({config.evaluation.quality_gate*100:.0f}%): {'PASS' if passed else 'FAIL'}")
    results["quality_gate_passed"] = passed


def _build_prompt(messages, tokenizer) -> str:
    """Build prompt string from message list."""
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Manual format
    parts = []
    for m in messages:
        parts.append("<|im_start|>" + m["role"] + "\n" + m["content"] + "<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)
