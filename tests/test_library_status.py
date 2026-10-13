"""A library status surface derived from the queue FSM and bundle stages."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any, cast

import pytest

from lectern import cli, records
from lectern.automation import AutomationError, open_state
from lectern.bundle import Manifest, StageName, StageState

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
SYNTHETIC_TALK = FIXTURE_DIR / "synthetic_talk.wav"
SYNTHETIC_TRANSCRIPT = FIXTURE_DIR / "synthetic_talk.transcript.txt"


def _archive(tmp_path: Path) -> tuple[Path, str, str]:
    media = tmp_path / "media" / "synthetic_talk.wav"
    media.parent.mkdir(parents=True)
    media.write_bytes(SYNTHETIC_TALK.read_bytes())
    media.with_suffix(".transcript.txt").write_text(
        SYNTHETIC_TRANSCRIPT.read_text(encoding="utf-8"), encoding="utf-8"
    )
    state_path = tmp_path / "state.sqlite"
    assert (
        cli.main(
            [
                "ingest",
                str(media),
                "--output",
                str(tmp_path / "bundles"),
                "--state",
                str(state_path),
            ]
        )
        == 0
    )
    with open_state(state_path) as state:
        bundle = state.list_library()[0]
    return state_path, bundle.bundle_id, bundle.queue_item_id


def test_status_taxonomy_maps_trustworthy_queue_and_stage_states() -> None:
    status = records.LibraryStatus
    derive = records.derive_library_status

    assert derive(records.QueueState.COMPLETED, [StageState.DONE]) is status.READY
    assert derive(records.QueueState.APPROVED, [StageState.DONE]) is status.INCOMPLETE
    assert derive(records.QueueState.COMPLETED, [StageState.RUNNING]) is status.INCOMPLETE
    assert derive(records.QueueState.FAILED, [StageState.DONE]) is status.FAILED
    assert derive(records.QueueState.COMPLETED, [StageState.FAILED]) is status.FAILED
    assert (
        derive(records.QueueState.COMPLETED, [StageState.DONE], source_changed=True)
        is status.NEEDS_REPROCESSING
    )


def test_library_status_is_projected_from_live_queue_and_manifest(tmp_path: Path) -> None:
    state_path, bundle_id, queue_item_id = _archive(tmp_path)

    with open_state(state_path) as state:
        ready = state.get_library_bundle(bundle_id)
        assert ready.status is records.LibraryStatus.READY
        assert ready.to_dict()["status"] == "ready"

        state.approve_queue_item(queue_item_id)
        assert state.get_library_bundle(bundle_id).status is records.LibraryStatus.INCOMPLETE

        state._connection.execute(  # pyright: ignore[reportPrivateUsage]
            "UPDATE queue_items SET state = ? WHERE id = ?",
            (records.QueueState.COMPLETED.value, queue_item_id),
        )
        state._connection.execute(  # pyright: ignore[reportPrivateUsage]
            "UPDATE source_items SET sha256 = ? WHERE id = ?",
            ("0" * 64, ready.source_item_id),
        )
        assert (
            state.get_library_bundle(bundle_id).status is records.LibraryStatus.NEEDS_REPROCESSING
        )

        state._connection.execute(  # pyright: ignore[reportPrivateUsage]
            "UPDATE source_items SET sha256 = "
            "(SELECT content_sha256 FROM queue_items WHERE id = ?) WHERE id = ?",
            (queue_item_id, ready.source_item_id),
        )
        manifest = Manifest.load(Path(ready.bundle_path))
        manifest.stages[StageName.SYNTHESIZE].state = StageState.FAILED
        manifest.stages[StageName.SYNTHESIZE].error = "synthetic failure"
        manifest.save(Path(ready.bundle_path))
        assert state.get_library_bundle(bundle_id).status is records.LibraryStatus.FAILED


def test_illegal_queue_transition_cannot_change_reported_status(tmp_path: Path) -> None:
    state_path, bundle_id, queue_item_id = _archive(tmp_path)

    with open_state(state_path) as state:
        assert state.get_library_bundle(bundle_id).status is records.LibraryStatus.READY
        with pytest.raises(AutomationError, match="retry is only legal from state failed"):
            state.retry_queue_item(queue_item_id)
        assert state.get_library_bundle(bundle_id).status is records.LibraryStatus.READY


def test_missing_declared_output_requires_reprocessing(tmp_path: Path) -> None:
    state_path, bundle_id, _ = _archive(tmp_path)
    with open_state(state_path) as state:
        bundle = state.get_library_bundle(bundle_id)
    (Path(bundle.bundle_path) / "transcript" / "segments.json").unlink()

    with open_state(state_path) as state:
        assert (
            state.get_library_bundle(bundle_id).status is records.LibraryStatus.NEEDS_REPROCESSING
        )


def test_modified_declared_output_requires_reprocessing(tmp_path: Path) -> None:
    state_path, bundle_id, _ = _archive(tmp_path)
    with open_state(state_path) as state:
        bundle = state.get_library_bundle(bundle_id)
    segments = Path(bundle.bundle_path) / "transcript" / "segments.json"
    segments.write_bytes(segments.read_bytes() + b" ")

    with open_state(state_path) as state:
        assert (
            state.get_library_bundle(bundle_id).status is records.LibraryStatus.NEEDS_REPROCESSING
        )


def test_symlinked_declared_output_parent_requires_reprocessing(tmp_path: Path) -> None:
    state_path, bundle_id, _ = _archive(tmp_path)
    with open_state(state_path) as state:
        bundle = state.get_library_bundle(bundle_id)
    bundle_path = Path(bundle.bundle_path)
    transcript = bundle_path / "transcript"
    external = tmp_path / "external-transcript"
    transcript.rename(external)
    transcript.symlink_to(external, target_is_directory=True)

    with open_state(state_path) as state:
        assert (
            state.get_library_bundle(bundle_id).status is records.LibraryStatus.NEEDS_REPROCESSING
        )


def test_symlinked_bundle_root_requires_reprocessing(tmp_path: Path) -> None:
    state_path, bundle_id, _ = _archive(tmp_path)
    with open_state(state_path) as state:
        bundle = state.get_library_bundle(bundle_id)
    bundle_path = Path(bundle.bundle_path)
    external = tmp_path / "external-bundle"
    bundle_path.rename(external)
    bundle_path.symlink_to(external, target_is_directory=True)

    with open_state(state_path) as state:
        assert (
            state.get_library_bundle(bundle_id).status is records.LibraryStatus.NEEDS_REPROCESSING
        )


@pytest.mark.parametrize("failure", ["failed", "unsupported", "stage"])
def test_invalid_selected_evidence_preserves_failure_status_precedence(
    tmp_path: Path, failure: str
) -> None:
    state_path, bundle_id, queue_item_id = _archive(tmp_path)
    with open_state(state_path) as state:
        bundle = Path(state.get_library_bundle(bundle_id).bundle_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for stage in manifest["stages"].values():
        stage["outputs"] = [item for item in stage["outputs"] if item["path"] != "source.json"]
    if failure == "stage":
        manifest["stages"]["synthesize"]["state"] = "failed"
    else:
        with sqlite3.connect(state_path) as connection:
            connection.execute(
                "UPDATE queue_items SET state = ? WHERE id = ?", (failure, queue_item_id)
            )
    manifest_path.write_text(json.dumps(manifest))
    with open_state(state_path) as state:
        assert state.get_library_bundle(bundle_id).status is records.LibraryStatus.FAILED


def test_status_streams_large_declared_media_and_checks_digest_and_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    state_path, bundle_id, _ = _archive(tmp_path)
    with open_state(state_path) as state:
        bundle = Path(state.get_library_bundle(bundle_id).bundle_path)
    media = bundle / "media/large-synthetic.bin"
    payload = b"synthetic" * (150_000)
    media.write_bytes(payload)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["stages"]["normalize"]["outputs"].append(
        {
            "path": "media/large-synthetic.bin",
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        }
    )
    manifest_path.write_text(json.dumps(manifest))
    real_read_bytes = Path.read_bytes
    real_open = Path.open
    sizes: list[int] = []

    class BoundedReader:
        def __init__(self, stream: Any) -> None:
            self.stream = stream

        def __enter__(self) -> BoundedReader:
            return self

        def __exit__(self, *args: Any) -> None:
            self.stream.close()

        def read(self, size: int = -1) -> bytes:
            assert 0 < size <= 1024 * 1024
            sizes.append(size)
            return self.stream.read(size)

    def guarded_open(path: Path, *args: Any, **kwargs: Any) -> Any:
        stream = cast(Any, real_open(path, *args, **kwargs))
        return BoundedReader(stream) if path == media and args and args[0] == "rb" else stream

    def guarded_read_bytes(path: Path) -> bytes:
        assert path != media, "status buffered a whole media artifact"
        return real_read_bytes(path)

    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
    with open_state(state_path) as state:
        assert state.get_library_bundle(bundle_id).status is records.LibraryStatus.READY
    capsys.readouterr()
    for command in (["list"], ["show", bundle_id]):
        assert cli.main(["library", *command, "--state", str(state_path), "--json"]) == 0
        capsys.readouterr()
    assert len(sizes) >= 2
    with real_open(media, "r+b") as stream:
        stream.write(b"X")
    with open_state(state_path) as state:
        assert (
            state.get_library_bundle(bundle_id).status is records.LibraryStatus.NEEDS_REPROCESSING
        )
    media.write_bytes(payload)
    manifest["stages"]["normalize"]["outputs"][-1]["bytes"] += 1
    manifest_path.write_text(json.dumps(manifest))
    with open_state(state_path) as state:
        assert (
            state.get_library_bundle(bundle_id).status is records.LibraryStatus.NEEDS_REPROCESSING
        )


@pytest.mark.parametrize("status", ["ready", "incomplete", "failed", "needs-reprocessing"])
@pytest.mark.parametrize("command", ["list", "show"])
def test_plain_library_rows_include_actual_status_and_preserve_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], status: str, command: str
) -> None:
    state_path, bundle_id, queue_id = _archive(tmp_path)
    with open_state(state_path) as state:
        bundle = state.get_library_bundle(bundle_id)
        if status == "incomplete":
            state.approve_queue_item(queue_id)
        elif status == "failed":
            manifest = Manifest.load(Path(bundle.bundle_path))
            manifest.stages[StageName.SYNTHESIZE].state = StageState.FAILED
            manifest.stages[StageName.SYNTHESIZE].error = "synthetic failure"
            manifest.save(Path(bundle.bundle_path))
        elif status == "needs-reprocessing":
            artifact = Path(bundle.bundle_path) / "transcript/segments.json"
            artifact.write_bytes(artifact.read_bytes() + b" ")
        observed = state.get_library_bundle(bundle_id)
        assert observed.status.value == status
        expected_bundle = observed.to_dict()
        expected_manifest = Manifest.load(Path(bundle.bundle_path)).model_dump(mode="json")
    capsys.readouterr()
    args = [
        "library",
        command,
        *([bundle_id] if command == "show" else []),
        "--state",
        str(state_path),
    ]
    assert cli.main(args) == 0
    plain = capsys.readouterr()
    assert plain.err == ""
    if command == "list":
        assert plain.out == f"{bundle_id}\t{bundle.created_at}\t{status}\tSynthetic Talk\n"
    else:
        assert f"Bundle: {bundle_id}\n" in plain.out
        assert f"Created: {bundle.created_at}\n" in plain.out
        assert f"Status: {status}\n" in plain.out
    assert cli.main([*args, "--json"]) == 0
    structured = capsys.readouterr()
    assert structured.err == ""
    payload = json.loads(structured.out)
    if command == "list":
        assert payload == {"bundles": [expected_bundle]}
    else:
        assert payload["bundle"] == expected_bundle
        assert payload["manifest"] == expected_manifest
        assert payload["status"] == status
