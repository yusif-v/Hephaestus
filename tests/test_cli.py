from hephaestus.cli import _add_forge_subparser, _add_dataset_subparser
import argparse


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
