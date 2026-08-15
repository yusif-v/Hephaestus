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


def test_resume_after_designed_skips_interview_and_preserves_spec(tmp_path, monkeypatch):
    from hephaestus.forge_state import save_state
    from tests.test_brain import GOOD_SPEC

    out = str(tmp_path)
    spec_path = os.path.join(out, "dataset-spec.json")
    with open(spec_path, "w") as f:
        json.dump(GOOD_SPEC, f)
    save_state(out, {"stage": "designed", "spec_hash": "oldhash"})

    calls = {"interview": 0, "design": 0}
    monkeypatch.setattr("hephaestus.forge.ask_input",
                        lambda *a, **k: calls.__setitem__("interview", calls["interview"] + 1) or "")
    monkeypatch.setattr("hephaestus.forge.design_spec",
                        lambda draft: calls.__setitem__("design", calls["design"] + 1) or GOOD_SPEC)
    monkeypatch.setattr("hephaestus.forge.ask_size", lambda *a: "latency")
    monkeypatch.setattr("hephaestus.forge.build_kernel", lambda *a, **k: {})
    monkeypatch.setattr("hephaestus.forge._push_and_wait_kernel", lambda *a: {})

    state = run_forge(
        out_dir=out,
        dataset_slug="yusifovtelman/forge-test",
        kernel_slug="yusifovtelman/forge-test-train",
        push_kernel=False,
        resume=True,
    )
    assert calls["interview"] == 0
    assert calls["design"] == 0
    with open(spec_path) as f:
        assert json.load(f) == GOOD_SPEC
    assert state["spec_hash"] == "oldhash"


def test_resume_interview_only_after_designed_stops_without_building(tmp_path, monkeypatch):
    from hephaestus.forge_state import save_state
    from tests.test_brain import GOOD_SPEC

    out = str(tmp_path)
    with open(os.path.join(out, "dataset-spec.json"), "w") as f:
        json.dump(GOOD_SPEC, f)
    save_state(out, {"stage": "designed", "spec_hash": "oldhash"})

    calls = {"interview": 0, "build": 0}
    monkeypatch.setattr("hephaestus.forge.ask_input",
                        lambda *a, **k: calls.__setitem__("interview", calls["interview"] + 1) or "")
    monkeypatch.setattr("hephaestus.forge.design_spec", lambda draft: GOOD_SPEC)
    monkeypatch.setattr("hephaestus.forge.build_dataset",
                        lambda *a, **k: calls.__setitem__("build", calls["build"] + 1) or {})

    state = run_forge(
        out_dir=out,
        dataset_slug="x/y",
        kernel_slug="x/z",
        push_kernel=False,
        resume=True,
        interview_only=True,
    )
    assert calls["interview"] == 0
    assert calls["build"] == 0
    assert state["stage"] == "designed"
