from hephaestus.spec import LocalSourceConfig, PublicSourceConfig, SourcesConfig
from hephaestus.sources import ingest_local, ingest_public, unified_table


def test_ingest_local_jsonl():
    src = LocalSourceConfig(path="tests/fixtures/events.jsonl", label_field=None)
    rows = ingest_local(src, {"path": "string", "recent_count": "int"})
    assert len(rows) == 5
    assert rows[0]["path"] == "/tmp/stager86"
    assert isinstance(rows[0]["recent_count"], int)


def test_ingest_local_csv():
    src = LocalSourceConfig(path="tests/fixtures/events.csv", label_field=None)
    rows = ingest_local(src, {})
    assert len(rows) == 2
    assert rows[1]["path"] == "/etc/hosts"


def test_ingest_local_missing_raises():
    src = LocalSourceConfig(path="tests/fixtures/nope.jsonl", label_field=None)
    try:
        ingest_local(src, {})
        assert False, "should raise"
    except FileNotFoundError:
        pass


def test_ingest_public_with_fake_loader():
    fake = lambda ds, split=None, trust_remote_code=False: {  # noqa: E731
        "train": [
            {"path": "/x/a.crypt", "event": "create", "recent_count": 3},
            {"path": "/y/b.txt", "event": "read", "recent_count": 1},
        ]
    }
    src = PublicSourceConfig(dataset="fake/ds", split="train", label_field=None)
    rows = ingest_public(src, {}, loader=fake)
    assert len(rows) == 2


def test_ingest_public_dict_no_split_takes_first():
    fake = lambda ds, split=None, trust_remote_code=False: {  # noqa: E731
        "train": [{"path": "/a", "event": "create", "recent_count": 1}],
        "test": [{"path": "/b", "event": "read", "recent_count": 2}],
    }
    src = PublicSourceConfig(dataset="fake/ds", split=None, label_field=None)
    rows = ingest_public(src, {}, loader=fake)
    assert rows == [{"path": "/a", "event": "create", "recent_count": 1}]


def test_unified_table_merges_local_and_public():
    fake = lambda ds, split=None, trust_remote_code=False: {  # noqa: E731
        "train": [{"path": "/pub/x", "event": "create", "recent_count": 7}]
    }
    sources = SourcesConfig(
        local=[LocalSourceConfig(path="tests/fixtures/events.jsonl")],
        public=[PublicSourceConfig(dataset="fake/ds", split="train")],
    )
    rows = unified_table(sources, {}, public_loader=fake)
    assert len(rows) == 6
