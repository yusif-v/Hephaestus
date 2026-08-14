from hephaestus.wizard import run_wizard, QUESTIONS


def test_question_keys_are_stable():
    keys = [q["key"] for q in QUESTIONS]
    assert "task_type" in keys
    assert "inputs" in keys
    assert "classes" in keys
    assert "sources" in keys
    assert "rules" in keys
    assert "scale" in keys


def test_run_wizard_with_scripted_answers():
    answers = iter([
        "score|label",
        "filesystem events: path, event, recent_count",
        "Benign, RansomwareEncrypt, TempStaging",
        "local tests/fixtures/events.jsonl",
        ".crypt -> RansomwareEncrypt",
        "small",
    ])

    def ask(prompt, options=None):
        return next(answers)

    draft = run_wizard(ask=ask)
    assert draft["task_type"] == "score|label"
    assert "RansomwareEncrypt" in draft["classes"]
    assert "events.jsonl" in draft["sources"]
    assert ".crypt" in draft["rules"]
    assert draft["scale"] == "small"
