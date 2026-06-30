"""Model registry — track all trained models locally."""

import json
import os
import time

REGISTRY_PATH = os.path.expanduser("~/.hephaestus/registry.json")


def list_models():
    """List all registered models."""
    if not os.path.exists(REGISTRY_PATH):
        return []
    with open(REGISTRY_PATH) as f:
        return json.load(f)


def register_model(entry):
    """Register a trained model in the local registry."""
    os.makedirs(os.path.dirname(REGISTRY_PATH), exist_ok=True)

    registry = list_models()
    entry["registered_at"] = time.strftime("%Y-%m-%d %H:%M:%S UTC")
    registry.append(entry)

    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)

    print(f"Model registered: {entry.get('name', 'unknown')} -> {entry.get('path', '?')}")


def find_model(name: str):
    """Find a registered model by name."""
    registry = list_models()
    for entry in registry:
        if entry.get("name") == name or entry.get("task") == name:
            return entry
    return None


def remove_model(name: str):
    """Remove a model from registry."""
    registry = list_models()
    registry = [e for e in registry if e.get("name") != name]
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)
    print(f"Removed: {name}")
