import json
import os

from hephaestus.forge_state import load_state, save_state, update_stage


def test_save_load_roundtrip(tmp_path):
    save_state(str(tmp_path), {"stage": "designed", "spec_hash": "abc123"})
    state = load_state(str(tmp_path))
    assert state["stage"] == "designed"
    assert state["spec_hash"] == "abc123"


def test_load_missing_returns_empty(tmp_path):
    assert load_state(str(tmp_path)) == {}


def test_update_stage_keeps_other_fields(tmp_path):
    save_state(str(tmp_path), {"stage": "designed", "spec_hash": "abc"})
    update_stage(str(tmp_path), "built")
    state = load_state(str(tmp_path))
    assert state["stage"] == "built"
    assert state["spec_hash"] == "abc"
