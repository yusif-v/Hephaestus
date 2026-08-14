"""Interactive interview — records user answers into a draft dict, no cleverness."""

from typing import Callable, List, Optional


QUESTIONS = [
    {
        "key": "task_type",
        "prompt": "What should the model output? (classification / score / score|label)",
        "options": ["classification", "score", "score|label"],
    },
    {
        "key": "inputs",
        "prompt": "Describe what the model will see. "
                  "Comma-separated field names, or path to a sample file to ingest columns.",
        "options": None,
    },
    {
        "key": "classes",
        "prompt": "What are the answer classes? (comma-separated, or 'score-only')",
        "options": None,
    },
    {
        "key": "sources",
        "prompt": "Where does the data live? Local paths and/or public dataset IDs "
                  "(comma-separated).",
        "options": None,
    },
    {
        "key": "rules",
        "prompt": "Any hard rules you already know? e.g. '.crypt -> RansomwareEncrypt'. "
                  "Leave blank for none.",
        "options": None,
    },
    {
        "key": "scale",
        "prompt": "Scale intent? (small / medium / large)",
        "options": ["small", "medium", "large"],
    },
]


def _default_ask(prompt: str, options: Optional[List[str]] = None) -> str:
    if options:
        print(f"\n{prompt}")
        for i, o in enumerate(options, 1):
            print(f"  {i}. {o}")
        return input("> ").strip()
    return input(f"\n{prompt}\n> ").strip()


def run_wizard(ask: Optional[Callable[[str, Optional[List[str]]], str]] = None) -> dict:
    if ask is None:
        ask = _default_ask
    draft = {}
    for q in QUESTIONS:
        draft[q["key"]] = ask(q["prompt"], q["options"])
    return draft
