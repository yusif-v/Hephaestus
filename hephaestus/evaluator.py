"""Evaluation — inference + metrics + quality gates."""

import torch


def evaluate_model(model, tokenizer, test_data, config):
    """Evaluate model on test set. Returns metrics dict."""

    model.eval()
    max_samples = min(config.evaluation.max_test_samples, len(test_data))
    correct = tp = fp = fn = tn = 0

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

        is_anomaly_ground = "ANOMALY" in expected.upper() and "NORMAL" not in expected.upper()

        # Build prompt from all messages except last
        prompt = ""
        for msg in messages[:-1]:
            prompt += "<|im_start|>" + msg["role"] + "\n" + msg["content"] + "<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=64, do_sample=False)

        resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        is_anomaly_pred = "ANOMALY" in resp.upper() and "NORMAL" not in resp.upper()

        if is_anomaly_ground == is_anomaly_pred:
            correct += 1
        if is_anomaly_pred and is_anomaly_ground:
            tp += 1
        if is_anomaly_pred and not is_anomaly_ground:
            fp += 1
        if not is_anomaly_pred and is_anomaly_ground:
            fn += 1
        if not is_anomaly_pred and not is_anomaly_ground:
            tn += 1

    n = max_samples
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
    }

    print(f"Accuracy: {accuracy*100:.1f}% | F1: {f1*100:.1f}%")
    print(f"Precision: {precision*100:.1f}% | Recall: {recall*100:.1f}%")
    print(f"TP={tp} FP={fp} FN={fn} TN={tn}")

    passed = accuracy >= config.evaluation.quality_gate
    print(f"Quality Gate ({config.evaluation.quality_gate*100:.0f}%): {'PASS' if passed else 'FAIL'}")

    results["quality_gate_passed"] = passed
    return results
