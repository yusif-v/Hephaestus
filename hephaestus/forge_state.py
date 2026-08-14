"""Pipeline state — tracks which forge stage the current task is at."""

import json
import os
from typing import Any, Dict

STAGES = ["designed", "built", "forged", "delivered"]
STATE_FILE = "forge_state.json"


def _path(out_dir: str) -> str:
    return os.path.join(out_dir, STATE_FILE)


def load_state(out_dir: str) -> Dict[str, Any]:
    p = _path(out_dir)
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        return json.load(f)


def save_state(out_dir: str, state: Dict[str, Any]) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(_path(out_dir), "w") as f:
        json.dump(state, f, indent=2)


def update_stage(out_dir: str, stage: str) -> Dict[str, Any]:
    state = load_state(out_dir)
    state["stage"] = stage
    save_state(out_dir, state)
    return state
