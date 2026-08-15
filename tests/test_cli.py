import argparse
import json
import os

import pytest

from hephaestus.cli import (
    _add_forge_subparser,
    _add_dataset_subparser,
    _cmd_dataset,
    _cmd_forge,
)


def test_forge_subparser_flags():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    _add_forge_subparser(sub)
    args = parser.parse_args(["forge", "--interview-only", "--resume"])
    assert args.command == "forge"
    assert args.interview_only is True
    assert args.resume is True


def test_dataset_subparser_build_command():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    _add_dataset_subparser(sub)
    args = parser.parse_args(["dataset", "build", "spec.json"])
    assert args.dataset_action == "build"
    assert args.spec == "spec.json"


def test_dataset_subparser_requires_spec():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    _add_dataset_subparser(sub)
    with pytest.raises(SystemExit):
        parser.parse_args(["dataset", "build"])


def test_cmd_dataset_show_prints_summary(tmp_path, capsys):
    from tests.test_brain import GOOD_SPEC

    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(GOOD_SPEC))
    args = argparse.Namespace(dataset_action="show", spec=str(spec_path),
                              out_dir=str(tmp_path))
    _cmd_dataset(args)
    assert "Task: filesystem-anomaly" in capsys.readouterr().out


def test_cmd_forge_interview_only_prints_spec(tmp_path, monkeypatch, capsys):
    from tests.test_brain import GOOD_SPEC

    out = str(tmp_path)

    def fake_run_forge(**kwargs):
        with open(os.path.join(out, "dataset-spec.json"), "w") as f:
            json.dump(GOOD_SPEC, f)
        return {"stage": "designed", "spec_hash": "abc123"}

    monkeypatch.setattr("hephaestus.forge.run_forge", fake_run_forge)
    args = argparse.Namespace(dataset_slug=None, kernel_slug=None,
                              out_dir=out, interview_only=True, resume=False)
    _cmd_forge(args)
    printed = capsys.readouterr().out
    assert "Task: filesystem-anomaly" in printed
    assert "Inputs: path, event, recent_count" in printed
