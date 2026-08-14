"""Brain — LLM client that turns draft answers into a complete DatasetSpec dict."""

import json
import os
import urllib.request
from typing import Callable, Optional


SCHEMA_HINT = (
    "Return a JSON object with these top-level keys: "
    "spec_version (int), task (name, description, task_type, classes, score_range, "
    "system_prompt, output_format), features (inputs list, field_types dict), "
    "labeling (strategy, rules list where each rule is {class, score, priority, when: "
    "{field, op, value}}, llm_gap_fill {enabled, max_fraction}, default_benign_score, "
    "default_class), sources (local [{path, label_field}], public [{dataset, split, "
    "label_field}], sampling {per_class_cap, benign_cap, seed}), augmentation {enabled}, "
    "output (train_split, max_seq_length, target_size {min, ideal}). "
    "Rules use ops: eq neq gt gte lt lte contains startswith endswith in regex. "
    "Do not invent label fields that are also input features."
)


def build_brain_prompt(draft: dict) -> str:
    return (
        "You are Hephaestus, the forge brain. Design a training dataset spec for a "
        "purpose-built LLM automaton.\n\n"
        f"Draft answers from the operator interview:\n{json.dumps(draft, indent=2)}\n\n"
        + SCHEMA_HINT
    )


def _default_llm(prompt: str) -> str:
    """Call OpenRouter (or a configured OpenAI-compatible endpoint) via stdlib urllib."""
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    model = os.environ.get("HEPHAESTUS_BRAIN_MODEL", "openai/gpt-4o-mini")
    url = os.environ.get(
        "HEPHAESTUS_BRAIN_URL", "https://openrouter.ai/api/v1/chat/completions")
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
    }).encode()
    req = urllib.request.Request(url, data=body, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}" if api_key else "",
    })
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode())
    return data["choices"][0]["message"]["content"]


def design_spec(
    draft: dict,
    llm_fn: Optional[Callable[[str], str]] = None,
) -> dict:
    """Produce a spec dict. Validates the result; retries up to 3 times."""
    if llm_fn is None:
        llm_fn = _default_llm
    prompt = build_brain_prompt(draft)
    last_error = None
    for _ in range(3):
        raw = llm_fn(prompt)
        try:
            spec = json.loads(raw)
        except json.JSONDecodeError as e:
            last_error = str(e)
            continue
        from .spec import from_dict, validate
        try:
            errors = validate(from_dict(spec))
        except Exception as e:  # malformed structure
            last_error = str(e)
            continue
        if not errors:
            return spec
        last_error = "; ".join(errors)
    raise ValueError(f"brain could not produce a valid spec after 3 tries: {last_error}")
