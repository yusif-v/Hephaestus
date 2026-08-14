import json
import os

from hephaestus.forge import run_forge, STAGES


def test_stages_order():
    assert STAGES == ["designed", "built", "forged", "delivered"]


def test_run_forge_end_to_end_with_fakes(tmp_path, monkeypatch):
    # Wizard is fully scripted; brain returns a valid spec for the fixture.
    answers = iter([
        "score|label",
        "path, event, recent_count, score_v",
        "Benign, RansomwareEncrypt, TempStaging",
        "local tests/fixtures/events.jsonl",
        ".crypt -> RansomwareEncrypt",
        "small",
    ])
    monkeypatch.setattr("hephaestus.forge.ask_input", lambda *a: next(answers))
    # Model-size choice via input too.
    size_answers = iter(["latency"])
    monkeypatch.setattr("hephaestus.forge.ask_size", lambda *a: next(size_answers))

    from hephaestus.brain import design_spec
    from tests.test_brain import GOOD_SPEC
    monkeypatch.setattr("hephaestus.forge.design_spec",
                        lambda draft: GOOD_SPEC)
    # Skip actual Kaggle push.
    monkeypatch.setattr("hephaestus.forge._push_and_wait_kernel",
                        lambda kernel_path, slug: {"status": "COMPLETE"})

    state = run_forge(
        out_dir=str(tmp_path),
        dataset_slug="yusifovtelman/forge-test",
        kernel_slug="yusifovtelman/forge-test-train",
        push_kernel=False,
    )
    assert state["stage"] == "delivered"
    assert state["spec_hash"]
    assert os.path.exists(os.path.join(str(tmp_path), state["spec_hash"], "train.jsonl"))
