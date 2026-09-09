"""The frozen-output manifest catches any drift in an artifact directory."""

import json

import pytest

from lcsa.manifest import check_manifest, write_manifest


def test_manifest_round_trip_and_drift(tmp_path):
    (tmp_path / "a.csv").write_text("x,y\n1,2\n")
    (tmp_path / "shards").mkdir()
    (tmp_path / "shards" / "b.json").write_text("{}")
    rec = write_manifest(tmp_path)
    assert rec["n_files"] == 2 and set(rec["files"]) == {"a.csv", "shards/b.json"}
    assert check_manifest(tmp_path)["ok"]
    # A second manifest never hashes the first.
    write_manifest(tmp_path, "manifest_again.json")
    assert check_manifest(tmp_path)["ok"]
    (tmp_path / "a.csv").write_text("x,y\n1,3\n")
    (tmp_path / "shards" / "b.json").unlink()
    (tmp_path / "c.txt").write_text("new")
    rep = check_manifest(tmp_path)
    assert not rep["ok"]
    assert rep["changed"] == ["a.csv"] and rep["missing"] == ["shards/b.json"]
    assert rep["new"] == ["c.txt"]


def test_manifest_cli_exit_codes(tmp_path):
    from lcsa.cli import main

    (tmp_path / "a.json").write_text("1")
    assert main(["manifest", "--out", str(tmp_path)]) == 0
    assert main(["manifest", "--out", str(tmp_path), "--check"]) == 0
    (tmp_path / "a.json").write_text("2")
    assert main(["manifest", "--out", str(tmp_path), "--check"]) == 1
    with pytest.raises(FileNotFoundError, match="lcsa manifest"):
        check_manifest(tmp_path / "nowhere")
    assert json.loads((tmp_path / "manifest.json").read_text())["n_files"] == 1
