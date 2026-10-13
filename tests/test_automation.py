from __future__ import annotations

import hashlib
import json
import os
import socket
import sqlite3
import stat
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from pytest import MonkeyPatch

from lectern import automation
from lectern import ingest as ingest_module
from lectern import state as state_module
from lectern.automation import (
    STATE_SCHEMA_VERSION,
    AutomationError,
    QueueState,
    SourcePolicy,
    attach_provenance_to_bundle,
    open_state,
    preflight_local_folder,
)
from lectern.bundle import MANIFEST_NAME, ArtifactRef, Manifest, StageName
from lectern.ingest import IngestError

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
SYNTHETIC_TALK = FIXTURE_DIR / "synthetic_talk.wav"
SYNTHETIC_TRANSCRIPT = FIXTURE_DIR / "synthetic_talk.transcript.txt"


def copy_fixture(directory: Path, name: str = "synthetic_talk.wav") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    media = directory / name
    media.write_bytes(SYNTHETIC_TALK.read_bytes())
    media.with_suffix(".transcript.txt").write_text(
        SYNTHETIC_TRANSCRIPT.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    return media


def copy_media_without_sidecar(directory: Path, name: str = "local_talk.wav") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    media = directory / name
    media.write_bytes(SYNTHETIC_TALK.read_bytes())
    return media


def test_state_store_initializes_with_schema_version(tmp_path: Path) -> None:
    state_path = tmp_path / "state.sqlite"

    with open_state(state_path):
        pass

    with sqlite3.connect(state_path) as connection:
        version = connection.execute("PRAGMA user_version").fetchone()[0]

    assert version == STATE_SCHEMA_VERSION


def test_new_state_store_is_private_before_content(tmp_path: Path) -> None:
    state_path = tmp_path / "state.sqlite"
    previous_umask = os.umask(0o022)
    try:
        with open_state(state_path) as state:
            assert stat.S_IMODE(state_path.stat().st_mode) == 0o600
            state._connection.execute("PRAGMA journal_mode=WAL")  # pyright: ignore[reportPrivateUsage]
            state._connection.execute("CREATE TABLE private_probe(value TEXT)")  # pyright: ignore[reportPrivateUsage]
            state._connection.execute("INSERT INTO private_probe VALUES ('private transcript')")  # pyright: ignore[reportPrivateUsage]
            state._connection.commit()  # pyright: ignore[reportPrivateUsage]
            for suffix in ("-wal", "-shm"):
                sidecar = Path(f"{state_path}{suffix}")
                if sidecar.exists():
                    assert stat.S_IMODE(sidecar.stat().st_mode) & 0o077 == 0
    finally:
        os.umask(previous_umask)


def test_new_state_store_creates_a_private_parent_under_group_umask(tmp_path: Path) -> None:
    state_path = tmp_path / "missing" / "state.sqlite"
    previous_umask = os.umask(0o002)
    try:
        with open_state(state_path):
            pass
    finally:
        os.umask(previous_umask)
    assert stat.S_IMODE(state_path.parent.stat().st_mode) == 0o700


def test_existing_state_store_and_sidecars_are_restricted(tmp_path: Path) -> None:
    state_path = tmp_path / "state.sqlite"
    with open_state(state_path) as first:
        first._connection.execute("PRAGMA journal_mode=WAL")  # pyright: ignore[reportPrivateUsage]
        first._connection.execute("CREATE TABLE private_upgrade_probe(value TEXT)")  # pyright: ignore[reportPrivateUsage]
        first._connection.execute("INSERT INTO private_upgrade_probe VALUES ('private')")  # pyright: ignore[reportPrivateUsage]
        first._connection.commit()  # pyright: ignore[reportPrivateUsage]
        os.chmod(state_path, 0o644)
        for suffix in ("-wal", "-shm"):
            sidecar = Path(f"{state_path}{suffix}")
            if sidecar.exists():
                os.chmod(sidecar, 0o644)

        with open_state(state_path):
            pass

        assert stat.S_IMODE(state_path.stat().st_mode) == 0o600
        for suffix in ("-wal", "-shm"):
            sidecar = Path(f"{state_path}{suffix}")
            if sidecar.exists():
                assert stat.S_IMODE(sidecar.stat().st_mode) & 0o077 == 0


def test_state_store_refuses_path_swapped_during_connect(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    state_path = tmp_path / "state.sqlite"
    with open_state(state_path):
        pass
    victim = tmp_path / "victim.sqlite"
    with sqlite3.connect(victim) as connection:
        connection.execute("PRAGMA user_version = 91")

    parked = tmp_path / "checked.sqlite"
    real_connect = sqlite3.connect

    def swap_then_connect(path: str | Path) -> sqlite3.Connection:
        if Path(path) == state_path:
            state_path.rename(parked)
            state_path.symlink_to(victim)
        return real_connect(path)

    monkeypatch.setattr(state_module.sqlite3, "connect", swap_then_connect)
    with pytest.raises(AutomationError, match="changed while opening"):
        open_state(state_path)

    with real_connect(victim) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 91


def test_version_zero_initialization_resumes_after_v2_commit(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    state_path = tmp_path / "state.sqlite"
    real_create_v3 = automation.AutomationStateStore._create_schema_v3  # pyright: ignore[reportPrivateUsage]

    def interrupt_after_v2(self: object) -> None:
        raise RuntimeError("synthetic interruption after v2")

    monkeypatch.setattr(automation.AutomationStateStore, "_create_schema_v3", interrupt_after_v2)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        open_state(state_path)

    monkeypatch.setattr(automation.AutomationStateStore, "_create_schema_v3", real_create_v3)
    with open_state(state_path):
        pass
    with sqlite3.connect(state_path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == STATE_SCHEMA_VERSION


def test_state_store_rejects_unknown_future_schema(tmp_path: Path) -> None:
    state_path = tmp_path / "state.sqlite"
    with sqlite3.connect(state_path) as connection:
        connection.execute("PRAGMA user_version = 999")

    with pytest.raises(AutomationError, match="unsupported automation state schema"):
        open_state(state_path)


def test_source_scan_reports_delta_and_rescan_is_idempotent(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    media = copy_fixture(source_dir)
    state_path = tmp_path / "state.sqlite"

    with open_state(state_path) as state:
        source = state.add_local_folder_source("talks", source_dir)
        duplicate = state.add_local_folder_source("talks", source_dir)
        first = state.scan_source(source.id)
        second = state.scan_source(source.id)

        media.write_bytes(SYNTHETIC_TALK.read_bytes() + b"changed")
        changed = state.scan_source(source.id)

        media.unlink()
        removed = state.scan_source(source.id)

    assert duplicate == source
    assert [item.relative_path for item in first.added] == ["synthetic_talk.wav"]
    assert len(first.queued) == 1
    assert first.queued[0].state is QueueState.DISCOVERED
    assert second.added == []
    assert second.changed == []
    assert second.removed == []
    assert [item.relative_path for item in second.unchanged] == ["synthetic_talk.wav"]
    assert [item.relative_path for item in changed.changed] == ["synthetic_talk.wav"]
    assert [item.relative_path for item in removed.removed] == ["synthetic_talk.wav"]


def test_source_scan_excludes_local_state_and_bundle_output_dirs(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    copy_fixture(source_dir)
    copy_fixture(source_dir / ".lectern", "state-audio.wav")

    with open_state(tmp_path / "state.sqlite") as state:
        source = state.add_local_folder_source("talks", source_dir)
        first = state.scan_source(source.id)
        approved = state.approve_queue_item(first.queued[0].id)
        state.ingest_queue_item(approved.id, source_dir / "bundles")
        second = state.scan_source(source.id)

    assert [item.relative_path for item in first.added] == ["synthetic_talk.wav"]
    assert second.added == []
    assert second.changed == []
    assert [item.relative_path for item in second.unchanged] == ["synthetic_talk.wav"]
    assert second.queued == []


def test_local_folder_scan_ignores_in_progress_ingest_temp_dirs(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    copy_fixture(source_dir)
    temp_media = source_dir / "bundles" / ".lectern-ingest.review" / "media" / "audio.wav"
    temp_media.parent.mkdir(parents=True)
    temp_media.write_bytes(SYNTHETIC_TALK.read_bytes())

    with open_state(tmp_path / "state.sqlite") as state:
        source = state.add_local_folder_source("talks", source_dir)
        delta = state.scan_source(source.id)

    assert [item.relative_path for item in delta.added] == ["synthetic_talk.wav"]
    assert len(delta.queued) == 1


def test_source_scan_does_not_skip_user_directory_named_bundles(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    copy_fixture(source_dir / "archive" / "bundles")

    with open_state(tmp_path / "state.sqlite") as state:
        source = state.add_local_folder_source("talks", source_dir)
        delta = state.scan_source(source.id)

    assert [item.relative_path for item in delta.added] == ["archive/bundles/synthetic_talk.wav"]


def test_source_scan_discovers_common_video_container_extensions(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    copy_fixture(source_dir, "synthetic_talk.mov")

    with open_state(tmp_path / "state.sqlite") as state:
        source = state.add_local_folder_source("talks", source_dir)
        delta = state.scan_source(source.id)

    assert [item.relative_path for item in delta.added] == ["synthetic_talk.mov"]


def test_source_scan_skips_symlinks_that_escape_source_root(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    outside_media = copy_fixture(tmp_path / "outside")
    symlink = source_dir / "linked.wav"
    source_dir.mkdir()
    try:
        symlink.symlink_to(outside_media)
    except OSError as exc:
        pytest.skip(f"symlink creation is not supported here: {exc}")

    with open_state(tmp_path / "state.sqlite") as state:
        source = state.add_local_folder_source("talks", source_dir)
        delta = state.scan_source(source.id)

    assert delta.added == []
    assert delta.queued == []


def test_source_scan_skips_transcript_sidecars_that_escape_source_root(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    external_dir = tmp_path / "external"
    source_dir.mkdir()
    external_dir.mkdir()
    media = source_dir / "synthetic_talk.wav"
    media.write_bytes(SYNTHETIC_TALK.read_bytes())
    external_sidecar = external_dir / "synthetic_talk.transcript.txt"
    external_sidecar.write_text(SYNTHETIC_TRANSCRIPT.read_text(encoding="utf-8"), encoding="utf-8")
    sidecar = source_dir / "synthetic_talk.transcript.txt"
    try:
        sidecar.symlink_to(external_sidecar)
    except OSError as exc:
        pytest.skip(f"symlink creation is not supported here: {exc}")

    with open_state(tmp_path / "state.sqlite") as state:
        source = state.add_local_folder_source("talks", source_dir)
        delta = state.scan_source(source.id)

    assert delta.added == []
    assert delta.queued == []


def test_policy_states_control_queueing(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    disabled_dir = tmp_path / "disabled"
    copy_fixture(source_dir)
    copy_fixture(disabled_dir)

    with open_state(tmp_path / "state.sqlite") as state:
        scan_only = state.add_local_folder_source(
            "scan-only",
            source_dir,
            SourcePolicy.SCAN_ONLY,
        )
        disabled = state.add_local_folder_source(
            "disabled",
            disabled_dir,
            SourcePolicy.DISABLED,
        )

        scan_only_delta = state.scan_source(scan_only.id)
        disabled_delta = state.scan_source(disabled.id)

    assert len(scan_only_delta.added) == 1
    assert scan_only_delta.queued == []
    assert disabled_delta.added == []
    assert disabled_delta.queued == []


def test_queue_approval_ingests_bundle_with_provenance_and_library_record(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "source"
    copy_fixture(source_dir)
    output_root = tmp_path / "bundles"

    with open_state(tmp_path / "state.sqlite") as state:
        source = state.add_local_folder_source("talks", source_dir)
        delta = state.scan_source(source.id)
        queue_item = delta.queued[0]

        with pytest.raises(AutomationError, match="requires explicit approval"):
            state.ingest_queue_item(queue_item.id, output_root)

        approved = state.approve_queue_item(queue_item.id)
        result = state.ingest_queue_item(approved.id, output_root)
        completed = state.get_queue_item(approved.id)
        library = state.get_library_bundle(result.manifest.bundle_id)

    source_json = json.loads((result.bundle_dir / "source.json").read_text(encoding="utf-8"))
    source_json_hash = hashlib.sha256((result.bundle_dir / "source.json").read_bytes()).hexdigest()
    provenance = source_json["provenance"]
    manifest = Manifest.load(result.bundle_dir)

    assert completed.state is QueueState.COMPLETED
    assert completed.bundle_id == result.manifest.bundle_id
    assert library.queue_item_id == completed.id
    assert provenance["source_id"] == source.id
    assert provenance["source_item_id"] == queue_item.source_item_id
    assert provenance["queue_item_id"] == queue_item.id
    assert provenance["consent"] == "explicit_queue_approval"
    assert provenance["queue_state"] == QueueState.COMPLETED.value
    assert provenance["remote_services"]["allowed"] is False
    assert manifest.stages[StageName.ACQUIRE].outputs[0].path == "source.json"
    assert manifest.stages[StageName.ACQUIRE].outputs[0].sha256 == source_json_hash


def test_retried_completed_queue_ingest_returns_existing_bundle(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    copy_fixture(source_dir)
    output_root = tmp_path / "bundles"

    with open_state(tmp_path / "state.sqlite") as state:
        source = state.add_local_folder_source("talks", source_dir)
        queue_item = state.scan_source(source.id).queued[0]
        approved = state.approve_queue_item(queue_item.id)
        first = state.ingest_queue_item(approved.id, output_root)

        # Re-approval, not retry: retry is legal from FAILED only, so a rerun of
        # a completed item goes through approve.
        reapproved = state.approve_queue_item(approved.id)
        second = state.ingest_queue_item(reapproved.id, output_root)
        completed = state.get_queue_item(reapproved.id)

    assert second.bundle_dir == first.bundle_dir
    assert second.manifest.bundle_id == first.manifest.bundle_id
    assert completed.state is QueueState.COMPLETED
    assert completed.bundle_id == first.manifest.bundle_id
    assert completed.last_error is None


def test_queue_ingest_uses_local_transcriber_for_no_sidecar_source(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    copy_media_without_sidecar(source_dir)
    transcriber = _write_transcriber_script(
        tmp_path / "transcriber.py",
        json.dumps(
            {
                "segments": [
                    {
                        "start_s": 4.0,
                        "end_s": 5.0,
                        "text": "Queue command transcript.",
                    }
                ]
            }
        ),
    )
    output_root = tmp_path / "bundles"

    with open_state(tmp_path / "state.sqlite") as state:
        source = state.add_local_folder_source("talks", source_dir)
        queue_item = state.scan_source(source.id).queued[0]
        approved = state.approve_queue_item(queue_item.id)
        result = state.ingest_queue_item(
            approved.id,
            output_root,
            transcriber_command=f"{sys.executable} {transcriber}",
        )
        completed = state.get_queue_item(approved.id)
        library = state.get_library_bundle(result.manifest.bundle_id)

    source_json = json.loads((result.bundle_dir / "source.json").read_text(encoding="utf-8"))
    metadata = json.loads(
        (result.bundle_dir / "transcript" / "metadata.json").read_text(encoding="utf-8")
    )
    summary = (result.bundle_dir / "analysis" / "summary.md").read_text(encoding="utf-8")

    assert completed.state is QueueState.COMPLETED
    assert library.queue_item_id == completed.id
    assert source_json["transcript"]["method"] == "local_command_json"
    assert source_json["provenance"]["remote_services"]["allowed"] is False
    assert source_json["provenance"]["remote_services"]["scope"] == "lectern_core"
    assert source_json["provenance"]["remote_services"]["transcriber_network_posture"] == (
        "unverifiable_user_command"
    )
    assert metadata["remote_services"]["lectern_invoked"] is False
    assert "Time span: 00:04–00:05" in summary
    assert "Queue command transcript." not in summary


def test_retried_command_queue_ingest_same_output_returns_existing_bundle(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "source"
    copy_media_without_sidecar(source_dir)
    transcriber = _write_transcriber_script(
        tmp_path / "transcriber.py",
        json.dumps({"text": "Stable queue command transcript."}),
    )
    command = f"{sys.executable} {transcriber}"
    output_root = tmp_path / "bundles"

    with open_state(tmp_path / "state.sqlite") as state:
        source = state.add_local_folder_source("talks", source_dir)
        queue_item = state.scan_source(source.id).queued[0]
        approved = state.approve_queue_item(queue_item.id)
        first = state.ingest_queue_item(approved.id, output_root, transcriber_command=command)

        # Re-approval, not retry: retry is legal from FAILED only, so a rerun of
        # a completed item goes through approve.
        reapproved = state.approve_queue_item(approved.id)
        second = state.ingest_queue_item(
            reapproved.id,
            output_root,
            transcriber_command=command,
        )
        completed = state.get_queue_item(reapproved.id)

    assert second.bundle_dir == first.bundle_dir
    assert second.manifest.bundle_id == first.manifest.bundle_id
    assert completed.state is QueueState.COMPLETED
    assert completed.bundle_id == first.manifest.bundle_id
    assert completed.last_error is None


def test_queue_ingest_records_failed_local_transcriber(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    copy_media_without_sidecar(source_dir)
    transcriber = _write_transcriber_script(
        tmp_path / "transcriber.py",
        "transcriber failed",
        exit_code=7,
    )

    with open_state(tmp_path / "state.sqlite") as state:
        source = state.add_local_folder_source("talks", source_dir)
        queue_item = state.scan_source(source.id).queued[0]
        approved = state.approve_queue_item(queue_item.id)
        with pytest.raises(IngestError, match="local transcriber command failed"):
            state.ingest_queue_item(
                approved.id,
                tmp_path / "bundles",
                transcriber_command=f"{sys.executable} {transcriber}",
            )
        failed = state.get_queue_item(approved.id)

    assert failed.state is QueueState.FAILED
    assert failed.last_error is not None
    assert "local transcriber command failed" in failed.last_error


def test_queue_ingest_rejects_file_changed_after_approval(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    media = copy_fixture(source_dir)

    with open_state(tmp_path / "state.sqlite") as state:
        source = state.add_local_folder_source("talks", source_dir)
        queue_item = state.scan_source(source.id).queued[0]
        approved = state.approve_queue_item(queue_item.id)
        media.write_bytes(SYNTHETIC_TALK.read_bytes() + b"changed")

        with pytest.raises(AutomationError, match="changed since queue approval"):
            state.ingest_queue_item(approved.id, tmp_path / "bundles")
        failed = state.get_queue_item(approved.id)

    assert failed.state is QueueState.FAILED


def test_queue_ingest_rejects_transcript_sidecar_changed_after_approval(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    media = copy_fixture(source_dir)
    sidecar = media.with_suffix(".transcript.txt")

    with open_state(tmp_path / "state.sqlite") as state:
        source = state.add_local_folder_source("talks", source_dir)
        queue_item = state.scan_source(source.id).queued[0]
        approved = state.approve_queue_item(queue_item.id)
        sidecar.write_text("changed transcript\n", encoding="utf-8")

        with pytest.raises(AutomationError, match="changed since queue approval"):
            state.ingest_queue_item(approved.id, tmp_path / "bundles")


def test_one_shot_ingest_expands_user_home_before_state_setup(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    source = copy_fixture(home / "source")
    monkeypatch.setenv("HOME", str(home))

    with open_state(tmp_path / "state.sqlite") as state:
        result = state.ingest_one_shot(
            Path("~/source") / source.name,
            tmp_path / "bundles",
        )
        queue_item = state.list_queue()[0]

    assert result.bundle_dir.is_dir()
    assert queue_item.state is QueueState.COMPLETED


def test_one_shot_ingest_validates_source_before_recording_state(tmp_path: Path) -> None:
    with open_state(tmp_path / "state.sqlite") as state:
        with pytest.raises(IngestError, match="source file does not exist"):
            state.ingest_one_shot(tmp_path / "missing.wav", tmp_path / "bundles")

        assert state.list_sources() == []
        assert state.list_queue() == []


def test_one_shot_ingest_records_source_and_queue_provenance(tmp_path: Path) -> None:
    source = copy_fixture(tmp_path / "source")

    with open_state(tmp_path / "state.sqlite") as state:
        result = state.ingest_one_shot(source, tmp_path / "bundles")
        queue_items = state.list_queue()
        library = state.list_library()

    source_json = json.loads((result.bundle_dir / "source.json").read_text(encoding="utf-8"))
    provenance = source_json["provenance"]

    assert len(queue_items) == 1
    assert queue_items[0].state is QueueState.COMPLETED
    assert provenance["consent"] == "explicit_cli_invocation"
    assert provenance["queue_item_id"] == queue_items[0].id
    assert [bundle.bundle_id for bundle in library] == [result.manifest.bundle_id]


def test_one_shot_reingest_returns_existing_bundle_without_failing_queue(
    tmp_path: Path,
) -> None:
    source = copy_fixture(tmp_path / "source")

    with open_state(tmp_path / "state.sqlite") as state:
        first = state.ingest_one_shot(source, tmp_path / "bundles")
        second = state.ingest_one_shot(source, tmp_path / "bundles")
        queue_item = state.list_queue()[0]

    assert second.manifest.bundle_id == first.manifest.bundle_id
    assert second.bundle_dir == first.bundle_dir
    assert queue_item.state is QueueState.COMPLETED
    assert queue_item.last_error is None


def test_one_shot_command_reingest_reruns_transcriber(tmp_path: Path) -> None:
    source = copy_media_without_sidecar(tmp_path / "source")
    transcriber = tmp_path / "transcriber.py"
    command = f"{sys.executable} {transcriber}"
    output_root = tmp_path / "bundles"

    with open_state(tmp_path / "state.sqlite") as state:
        _write_transcriber_script(transcriber, json.dumps({"text": "First command transcript."}))
        first = state.ingest_one_shot(source, output_root, transcriber_command=command)

        _write_transcriber_script(transcriber, json.dumps({"text": "Second command transcript."}))
        second = state.ingest_one_shot(source, output_root, transcriber_command=command)
        queue_item = state.list_queue()[0]

    second_text = (second.bundle_dir / "transcript" / "transcript.md").read_text(encoding="utf-8")
    assert second.manifest.bundle_id != first.manifest.bundle_id
    assert "Second command transcript." in second_text
    assert queue_item.state is QueueState.COMPLETED
    assert queue_item.bundle_id == second.manifest.bundle_id


def test_one_shot_command_reingest_same_output_returns_existing_bundle(
    tmp_path: Path,
) -> None:
    source = copy_media_without_sidecar(tmp_path / "source")
    transcriber = _write_transcriber_script(
        tmp_path / "transcriber.py",
        json.dumps({"text": "Stable command transcript."}),
    )
    command = f"{sys.executable} {transcriber}"

    with open_state(tmp_path / "state.sqlite") as state:
        first = state.ingest_one_shot(source, tmp_path / "bundles", transcriber_command=command)
        second = state.ingest_one_shot(source, tmp_path / "bundles", transcriber_command=command)
        queue_item = state.list_queue()[0]

    assert second.manifest.bundle_id == first.manifest.bundle_id
    assert second.bundle_dir == first.bundle_dir
    assert queue_item.state is QueueState.COMPLETED
    assert queue_item.last_error is None


def test_one_shot_command_rerun_failure_preserves_completed_queue(
    tmp_path: Path,
) -> None:
    source = copy_media_without_sidecar(tmp_path / "source")
    transcriber = tmp_path / "transcriber.py"
    command = f"{sys.executable} {transcriber}"

    with open_state(tmp_path / "state.sqlite") as state:
        _write_transcriber_script(transcriber, json.dumps({"text": "Completed transcript."}))
        first = state.ingest_one_shot(source, tmp_path / "bundles", transcriber_command=command)

        _write_transcriber_script(transcriber, "rerun failed", exit_code=9)
        with pytest.raises(IngestError, match="local transcriber command failed"):
            state.ingest_one_shot(source, tmp_path / "bundles", transcriber_command=command)
        queue_item = state.list_queue()[0]

    assert queue_item.state is QueueState.COMPLETED
    assert queue_item.bundle_id == first.manifest.bundle_id
    assert queue_item.last_error is None


def test_duplicate_content_different_sources_do_not_overwrite_provenance(tmp_path: Path) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    copy_fixture(first_dir)
    copy_fixture(second_dir)

    with open_state(tmp_path / "state.sqlite") as state:
        first_source = state.add_local_folder_source("first", first_dir)
        first_queue = state.scan_source(first_source.id).queued[0]
        state.approve_queue_item(first_queue.id)
        first_result = state.ingest_queue_item(first_queue.id, tmp_path / "bundles")

        second_source = state.add_local_folder_source("second", second_dir)
        second_queue = state.scan_source(second_source.id).queued[0]
        state.approve_queue_item(second_queue.id)
        with pytest.raises(AutomationError, match="duplicate-content multi-source"):
            state.ingest_queue_item(second_queue.id, tmp_path / "bundles")
        failed = state.get_queue_item(second_queue.id)

    source_json = json.loads((first_result.bundle_dir / "source.json").read_text(encoding="utf-8"))
    assert source_json["provenance"]["queue_item_id"] == first_queue.id
    assert failed.state is QueueState.FAILED


def test_command_duplicate_content_different_sources_do_not_overwrite_provenance(
    tmp_path: Path,
) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    copy_media_without_sidecar(first_dir)
    copy_media_without_sidecar(second_dir)
    transcriber = _write_transcriber_script(
        tmp_path / "transcriber.py",
        json.dumps({"text": "Same command transcript."}),
    )
    command = f"{sys.executable} {transcriber}"

    with open_state(tmp_path / "state.sqlite") as state:
        first_source = state.add_local_folder_source("first", first_dir)
        first_queue = state.scan_source(first_source.id).queued[0]
        state.approve_queue_item(first_queue.id)
        first_result = state.ingest_queue_item(
            first_queue.id,
            tmp_path / "first-bundles",
            transcriber_command=command,
        )

        second_source = state.add_local_folder_source("second", second_dir)
        second_queue = state.scan_source(second_source.id).queued[0]
        state.approve_queue_item(second_queue.id)
        with pytest.raises(AutomationError, match="duplicate-content multi-source"):
            state.ingest_queue_item(
                second_queue.id,
                tmp_path / "second-bundles",
                transcriber_command=command,
            )
        failed = state.get_queue_item(second_queue.id)
        library = state.get_library_bundle(first_result.manifest.bundle_id)

    source_json = json.loads((first_result.bundle_dir / "source.json").read_text(encoding="utf-8"))
    duplicate_dir = tmp_path / "second-bundles" / first_result.manifest.bundle_id

    assert source_json["provenance"]["queue_item_id"] == first_queue.id
    assert library.queue_item_id == first_queue.id
    assert failed.state is QueueState.FAILED
    assert not duplicate_dir.exists()


def test_one_shot_command_duplicate_content_different_sources_does_not_complete_duplicate(
    tmp_path: Path,
) -> None:
    first = copy_media_without_sidecar(tmp_path / "first")
    second = copy_media_without_sidecar(tmp_path / "second")
    transcriber = _write_transcriber_script(
        tmp_path / "transcriber.py",
        json.dumps({"text": "Same one-shot command transcript."}),
    )
    command = f"{sys.executable} {transcriber}"

    with open_state(tmp_path / "state.sqlite") as state:
        first_result = state.ingest_one_shot(
            first,
            tmp_path / "first-bundles",
            transcriber_command=command,
        )
        first_queue = state.list_queue()[0]

        with pytest.raises(AutomationError, match="duplicate-content multi-source"):
            state.ingest_one_shot(
                second,
                tmp_path / "second-bundles",
                transcriber_command=command,
            )
        queues = state.list_queue()
        library = state.get_library_bundle(first_result.manifest.bundle_id)

    duplicate_dir = tmp_path / "second-bundles" / first_result.manifest.bundle_id
    failed = next(queue for queue in queues if queue.id != first_queue.id)

    assert library.queue_item_id == first_queue.id
    assert failed.state is QueueState.FAILED
    assert not duplicate_dir.exists()


def test_queue_ingest_rejects_existing_unindexed_bundle_directory(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    output_root = tmp_path / "bundles"
    copy_fixture(source_dir)

    with open_state(tmp_path / "first.sqlite") as state:
        source = state.add_local_folder_source("talks", source_dir)
        queue_item = state.scan_source(source.id).queued[0]
        state.approve_queue_item(queue_item.id)
        state.ingest_queue_item(queue_item.id, output_root)

    with open_state(tmp_path / "second.sqlite") as state:
        source = state.add_local_folder_source("talks", source_dir)
        queue_item = state.scan_source(source.id).queued[0]
        state.approve_queue_item(queue_item.id)
        with pytest.raises(AutomationError, match="already exists on disk"):
            state.ingest_queue_item(queue_item.id, output_root)

        failed = state.get_queue_item(queue_item.id)

    assert failed.state is QueueState.FAILED


def test_queue_skip_and_reapproval_are_inspectable(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    copy_fixture(source_dir)

    with open_state(tmp_path / "state.sqlite") as state:
        source = state.add_local_folder_source("talks", source_dir)
        queue_item = state.scan_source(source.id).queued[0]
        skipped = state.skip_queue_item(queue_item.id)
        # Retry no longer reopens a skipped item: retry is legal from FAILED
        # only, and approve is the verb that un-skips.
        with pytest.raises(AutomationError, match="retry is only legal from state failed"):
            state.retry_queue_item(queue_item.id)
        approved = state.approve_queue_item(queue_item.id)

    assert skipped.state is QueueState.SKIPPED
    assert approved.state is QueueState.APPROVED


def test_local_folder_scan_does_not_open_network_socket(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    source_dir = tmp_path / "source"
    copy_fixture(source_dir)

    def fail_socket(*args: object, **kwargs: object) -> socket.socket:
        raise AssertionError("local source scan attempted network access")

    monkeypatch.setattr(socket, "socket", fail_socket)

    with open_state(tmp_path / "state.sqlite") as state:
        source = state.add_local_folder_source("talks", source_dir)
        delta = state.scan_source(source.id)

    assert len(delta.added) == 1


def test_preflight_media_count_matches_scan_discovery(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    copy_fixture(source_dir)
    copy_fixture(source_dir / ".lectern", "state-audio.wav")

    with open_state(tmp_path / "state.sqlite") as state:
        source = state.add_local_folder_source("talks", source_dir)
        first = state.scan_source(source.id)
        approved = state.approve_queue_item(first.queued[0].id)
        state.ingest_queue_item(approved.id, source_dir / "bundles")

    preflight = preflight_local_folder(source_dir)

    assert (source_dir / "bundles").is_dir()
    assert [item.relative_path for item in first.added] == ["synthetic_talk.wav"]
    assert preflight.ok
    assert preflight.media_files == len(first.added)


def test_preflight_media_count_skips_symlinks_that_escape_source_root(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    outside_media = copy_fixture(tmp_path / "outside")
    source_dir.mkdir()
    symlink = source_dir / "linked.wav"
    try:
        symlink.symlink_to(outside_media)
    except OSError as exc:
        pytest.skip(f"symlink creation is not supported here: {exc}")

    with open_state(tmp_path / "state.sqlite") as state:
        source = state.add_local_folder_source("talks", source_dir)
        delta = state.scan_source(source.id)

    preflight = preflight_local_folder(source_dir)

    assert delta.added == []
    assert preflight.media_files == 0


def test_preflight_media_count_skips_sidecars_that_escape_source_root(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    external_dir = tmp_path / "external"
    source_dir.mkdir()
    external_dir.mkdir()
    media = source_dir / "synthetic_talk.wav"
    media.write_bytes(SYNTHETIC_TALK.read_bytes())
    external_sidecar = external_dir / "synthetic_talk.transcript.txt"
    external_sidecar.write_text(SYNTHETIC_TRANSCRIPT.read_text(encoding="utf-8"), encoding="utf-8")
    sidecar = source_dir / "synthetic_talk.transcript.txt"
    try:
        sidecar.symlink_to(external_sidecar)
    except OSError as exc:
        pytest.skip(f"symlink creation is not supported here: {exc}")

    with open_state(tmp_path / "state.sqlite") as state:
        source = state.add_local_folder_source("talks", source_dir)
        delta = state.scan_source(source.id)

    preflight = preflight_local_folder(source_dir)

    assert delta.added == []
    assert preflight.media_files == len(delta.added)


def test_provenance_records_actual_queue_state_and_bundle_remote_services(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    copy_fixture(source_dir)
    remote_services = {
        "allowed": True,
        "scope": "synthetic_probe_scope",
        "lectern_invoked": True,
        "requires_explicit_per_item_consent": False,
        "transcriber_network_posture": "synthetic_probe_posture",
    }

    with open_state(tmp_path / "state.sqlite") as state:
        source = state.add_local_folder_source("talks", source_dir)
        queue_item = state.scan_source(source.id).queued[0]
        approved = state.approve_queue_item(queue_item.id)
        result = state.ingest_queue_item(approved.id, tmp_path / "bundles")
        completed = state.get_queue_item(approved.id)
        source_item = state.get_source_item(queue_item.source_item_id)

        source_path = result.bundle_dir / "source.json"
        payload = json.loads(source_path.read_text(encoding="utf-8"))
        payload["transcript"]["remote_services"] = remote_services
        source_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

        skipped = state.skip_queue_item(completed.id)
        attach_provenance_to_bundle(
            result.bundle_dir,
            source=source,
            source_item=source_item,
            queue_item=skipped,
            consent="explicit_queue_approval",
        )

    provenance = json.loads(source_path.read_text(encoding="utf-8"))["provenance"]

    assert completed.state is QueueState.COMPLETED
    assert provenance["queue_state"] == QueueState.SKIPPED.value
    assert provenance["remote_services"] == remote_services


def test_queue_ingest_provenance_failure_cleans_up_and_retry_recovers(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    source_dir = tmp_path / "source"
    copy_fixture(source_dir)
    real_attach = automation.attach_provenance_to_bundle

    def fail_attach(*args: object, **kwargs: object) -> None:
        raise OSError("synthetic provenance write failure")

    with open_state(tmp_path / "state.sqlite") as state:
        source = state.add_local_folder_source("talks", source_dir)
        queue_item = state.scan_source(source.id).queued[0]
        approved = state.approve_queue_item(queue_item.id)
        monkeypatch.setattr(automation, "attach_provenance_to_bundle", fail_attach)

        with pytest.raises(OSError, match="synthetic provenance write failure"):
            state.ingest_queue_item(approved.id, tmp_path / "bundles")
        failed = state.get_queue_item(approved.id)

        assert failed.state is QueueState.FAILED
        assert failed.last_error is not None
        assert "synthetic provenance write failure" in failed.last_error
        # The library row inserted alongside the completion must be reversed too,
        # or the store would advertise a bundle that no longer exists on disk.
        assert state.list_library() == []
        # The half-written bundle must not survive: leaving it on disk makes the
        # advertised retry path unrecoverable (the planned bundle id collides
        # with an unrecorded directory).
        bundles_root = tmp_path / "bundles"
        leftover = list(bundles_root.iterdir()) if bundles_root.exists() else []
        assert leftover == []

        monkeypatch.setattr(automation, "attach_provenance_to_bundle", real_attach)
        retried = state.retry_queue_item(failed.id)
        state.approve_queue_item(retried.id)
        result = state.ingest_queue_item(retried.id, tmp_path / "bundles")
        recovered = state.get_queue_item(retried.id)

    assert recovered.state is QueueState.COMPLETED
    assert result.bundle_dir.is_dir()


def test_queue_ingest_library_record_failure_does_not_claim_completion(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    source_dir = tmp_path / "source"
    copy_fixture(source_dir)

    def fail_record(*args: object, **kwargs: object) -> bool:
        raise sqlite3.OperationalError("synthetic library insert failure")

    with open_state(tmp_path / "state.sqlite") as state:
        source = state.add_local_folder_source("talks", source_dir)
        queue_item = state.scan_source(source.id).queued[0]
        approved = state.approve_queue_item(queue_item.id)
        monkeypatch.setattr(
            automation.AutomationState, "_record_library_bundle", fail_record, raising=True
        )

        with pytest.raises(sqlite3.OperationalError, match="synthetic library insert failure"):
            state.ingest_queue_item(approved.id, tmp_path / "bundles")

        monkeypatch.undo()
        after = state.get_queue_item(approved.id)
        library = state.list_library()

    bundles_root = tmp_path / "bundles"
    leftover = list(bundles_root.iterdir()) if bundles_root.exists() else []

    # Completion and its library record share a transaction: a failed insert must
    # leave no COMPLETED claim, no library row, and no retry-blocking bundle dir.
    assert after.state is QueueState.FAILED
    assert library == []
    assert leftover == []


def test_one_shot_command_rerun_provenance_failure_preserves_completed_bundle(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    source = copy_media_without_sidecar(tmp_path / "source")
    first_command = f"{sys.executable} " + str(
        _write_transcriber_script(
            tmp_path / "first_transcriber.py",
            json.dumps({"text": "First command transcript."}),
        )
    )
    second_command = f"{sys.executable} " + str(
        _write_transcriber_script(
            tmp_path / "second_transcriber.py",
            json.dumps({"text": "Second command transcript."}),
        )
    )
    output_root = tmp_path / "bundles"
    real_attach = automation.attach_provenance_to_bundle

    def fail_attach(*args: object, **kwargs: object) -> None:
        raise OSError("synthetic provenance write failure")

    with open_state(tmp_path / "state.sqlite") as state:
        first = state.ingest_one_shot(source, output_root, transcriber_command=first_command)
        monkeypatch.setattr(automation, "attach_provenance_to_bundle", fail_attach)

        with pytest.raises(OSError, match="synthetic provenance write failure"):
            state.ingest_one_shot(source, output_root, transcriber_command=second_command)

        preserved = state.list_queue()[0]
        bundle_dirs = sorted(path.name for path in output_root.iterdir())
        library_ids = [bundle.bundle_id for bundle in state.list_library()]

        monkeypatch.setattr(automation, "attach_provenance_to_bundle", real_attach)
        replay = state.ingest_one_shot(source, output_root, transcriber_command=first_command)

    # A failed rerun must not demote an earlier success: the queue row keeps the
    # original completed bundle id, and only the new bundle is removed.
    assert preserved.state is QueueState.COMPLETED
    assert preserved.bundle_id == first.manifest.bundle_id
    assert preserved.last_error is None
    assert bundle_dirs == [first.manifest.bundle_id]
    assert library_ids == [first.manifest.bundle_id]
    assert replay.bundle_dir == first.bundle_dir
    assert replay.manifest.bundle_id == first.manifest.bundle_id


def test_attach_provenance_to_bundle_is_idempotent_for_identical_inputs(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    copy_fixture(source_dir)

    with open_state(tmp_path / "state.sqlite") as state:
        source = state.add_local_folder_source("talks", source_dir)
        queue_item = state.scan_source(source.id).queued[0]
        approved = state.approve_queue_item(queue_item.id)
        result = state.ingest_queue_item(approved.id, tmp_path / "bundles")
        completed = state.get_queue_item(approved.id)
        source_item = state.get_source_item(queue_item.source_item_id)

        first_source_json = (result.bundle_dir / "source.json").read_bytes()
        first_manifest = (result.bundle_dir / MANIFEST_NAME).read_bytes()
        attach_provenance_to_bundle(
            result.bundle_dir,
            source=source,
            source_item=source_item,
            queue_item=completed,
            consent="explicit_queue_approval",
        )

    # Re-attaching the same provenance must be a byte-for-byte no-op, or the
    # replay repair path would rewrite bundles on every completed replay.
    assert (result.bundle_dir / "source.json").read_bytes() == first_source_json
    assert (result.bundle_dir / MANIFEST_NAME).read_bytes() == first_manifest


def test_interrupted_provenance_attach_keeps_source_json_parseable(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    source_dir = tmp_path / "source"
    copy_fixture(source_dir)

    def fail_replace(source: object, destination: object) -> None:
        raise OSError("synthetic provenance publish failure")

    with open_state(tmp_path / "state.sqlite") as state:
        source = state.add_local_folder_source("talks", source_dir)
        queue_item = state.scan_source(source.id).queued[0]
        approved = state.approve_queue_item(queue_item.id)
        result = state.ingest_queue_item(approved.id, tmp_path / "bundles")
        completed = state.get_queue_item(approved.id)
        source_item = state.get_source_item(queue_item.source_item_id)

        source_json = result.bundle_dir / "source.json"
        before = source_json.read_bytes()
        _strip_provenance(source_json)
        stripped = source_json.read_bytes()
        monkeypatch.setattr(os, "replace", fail_replace)

        with pytest.raises(OSError, match="synthetic provenance publish failure"):
            attach_provenance_to_bundle(
                result.bundle_dir,
                source=source,
                source_item=source_item,
                queue_item=completed,
                consent="explicit_queue_approval",
            )

        monkeypatch.undo()
        leftover = sorted(path.name for path in result.bundle_dir.iterdir())

        # The committed COMPLETED row already points at this bundle: a crash
        # while republishing source.json must leave the prior file whole, not a
        # truncated one that _bundle_provenance_needs_repair refuses to repair.
        assert source_json.read_bytes() == stripped
        assert json.loads(source_json.read_text(encoding="utf-8"))["transcript"]
        assert all(not name.endswith(".tmp") for name in leftover)

        attach_provenance_to_bundle(
            result.bundle_dir,
            source=source,
            source_item=source_item,
            queue_item=completed,
            consent="explicit_queue_approval",
        )

    assert source_json.read_bytes() == before
    assert sorted(path.name for path in result.bundle_dir.iterdir()) == leftover


def test_one_shot_replay_repairs_missing_provenance(tmp_path: Path) -> None:
    source = copy_fixture(tmp_path / "source")
    output_root = tmp_path / "bundles"

    with open_state(tmp_path / "state.sqlite") as state:
        first = state.ingest_one_shot(source, output_root)
        source_json = first.bundle_dir / "source.json"
        _strip_provenance(source_json)

        replay = state.ingest_one_shot(source, output_root)

    provenance = json.loads(source_json.read_text(encoding="utf-8"))["provenance"]

    assert replay.bundle_dir == first.bundle_dir
    assert provenance["consent"] == "explicit_cli_invocation"
    assert provenance["queue_state"] == QueueState.COMPLETED.value
    assert _manifest_source_digest(replay.manifest) == _file_digest(source_json)


def test_one_shot_replay_repairs_stale_manifest_source_digest(tmp_path: Path) -> None:
    source = copy_fixture(tmp_path / "source")
    output_root = tmp_path / "bundles"

    with open_state(tmp_path / "state.sqlite") as state:
        first = state.ingest_one_shot(source, output_root)
        source_json = first.bundle_dir / "source.json"
        _staleify_manifest_source_digest(first.bundle_dir)

        replay = state.ingest_one_shot(source, output_root)

    provenance = json.loads(source_json.read_text(encoding="utf-8"))["provenance"]

    assert replay.bundle_dir == first.bundle_dir
    assert provenance["consent"] == "explicit_cli_invocation"
    assert _manifest_source_digest(replay.manifest) == _file_digest(source_json)


def test_queue_replay_repairs_missing_provenance(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    copy_fixture(source_dir)
    output_root = tmp_path / "bundles"

    with open_state(tmp_path / "state.sqlite") as state:
        source = state.add_local_folder_source("talks", source_dir)
        queue_item = state.scan_source(source.id).queued[0]
        approved = state.approve_queue_item(queue_item.id)
        first = state.ingest_queue_item(approved.id, output_root)
        source_json = first.bundle_dir / "source.json"
        _strip_provenance(source_json)

        reapproved = state.approve_queue_item(approved.id)
        replay = state.ingest_queue_item(reapproved.id, output_root)

    provenance = json.loads(source_json.read_text(encoding="utf-8"))["provenance"]

    assert replay.bundle_dir == first.bundle_dir
    assert provenance["consent"] == "explicit_queue_approval"
    assert provenance["queue_state"] == QueueState.COMPLETED.value
    assert _manifest_source_digest(replay.manifest) == _file_digest(source_json)


def test_queue_replay_repairs_stale_manifest_source_digest(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    copy_fixture(source_dir)
    output_root = tmp_path / "bundles"

    with open_state(tmp_path / "state.sqlite") as state:
        source = state.add_local_folder_source("talks", source_dir)
        queue_item = state.scan_source(source.id).queued[0]
        approved = state.approve_queue_item(queue_item.id)
        first = state.ingest_queue_item(approved.id, output_root)
        source_json = first.bundle_dir / "source.json"
        _staleify_manifest_source_digest(first.bundle_dir)

        reapproved = state.approve_queue_item(approved.id)
        replay = state.ingest_queue_item(reapproved.id, output_root)

    provenance = json.loads(source_json.read_text(encoding="utf-8"))["provenance"]

    assert replay.bundle_dir == first.bundle_dir
    assert provenance["consent"] == "explicit_queue_approval"
    assert _manifest_source_digest(replay.manifest) == _file_digest(source_json)


def test_one_shot_rerun_into_new_output_root_provenance_failure_keeps_library_row(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    source = copy_media_without_sidecar(tmp_path / "source")
    command = f"{sys.executable} " + str(
        _write_transcriber_script(
            tmp_path / "transcriber.py",
            json.dumps({"text": "Stable one-shot command transcript."}),
        )
    )
    first_root = tmp_path / "bundles"
    second_root = tmp_path / "other-bundles"

    def fail_attach(*args: object, **kwargs: object) -> None:
        raise OSError("synthetic provenance write failure")

    with open_state(tmp_path / "state.sqlite") as state:
        first = state.ingest_one_shot(source, first_root, transcriber_command=command)
        monkeypatch.setattr(automation, "attach_provenance_to_bundle", fail_attach)

        with pytest.raises(OSError, match="synthetic provenance write failure"):
            state.ingest_one_shot(source, second_root, transcriber_command=command)

        preserved = state.list_queue()[0]
        library = state.list_library()
        shown = state.get_library_bundle(first.manifest.bundle_id)

    # The rerun produced the same deterministic bundle id, so the library row was
    # updated rather than inserted. Undoing the failed rerun must put the row back
    # on the surviving original bundle instead of leaving it on the deleted copy.
    assert preserved.state is QueueState.COMPLETED
    assert preserved.bundle_id == first.manifest.bundle_id
    assert preserved.last_error is None
    assert first.bundle_dir.is_dir()
    assert not second_root.exists() or list(second_root.iterdir()) == []
    assert [bundle.bundle_path for bundle in library] == [str(first.bundle_dir.resolve())]
    assert Path(shown.bundle_path).is_dir()


def test_queue_command_rerun_provenance_failure_preserves_completed_bundle(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    source_dir = tmp_path / "source"
    copy_media_without_sidecar(source_dir)
    first_command = f"{sys.executable} " + str(
        _write_transcriber_script(
            tmp_path / "first_transcriber.py",
            json.dumps({"text": "First queue command transcript."}),
        )
    )
    second_command = f"{sys.executable} " + str(
        _write_transcriber_script(
            tmp_path / "second_transcriber.py",
            json.dumps({"text": "Second queue command transcript."}),
        )
    )
    output_root = tmp_path / "bundles"
    real_attach = automation.attach_provenance_to_bundle

    def fail_attach(*args: object, **kwargs: object) -> None:
        raise OSError("synthetic provenance write failure")

    with open_state(tmp_path / "state.sqlite") as state:
        source = state.add_local_folder_source("talks", source_dir)
        queue_item = state.scan_source(source.id).queued[0]
        approved = state.approve_queue_item(queue_item.id)
        first = state.ingest_queue_item(
            approved.id,
            output_root,
            transcriber_command=first_command,
        )

        reapproved = state.approve_queue_item(approved.id)
        monkeypatch.setattr(automation, "attach_provenance_to_bundle", fail_attach)
        with pytest.raises(OSError, match="synthetic provenance write failure"):
            state.ingest_queue_item(
                reapproved.id,
                output_root,
                transcriber_command=second_command,
            )

        preserved = state.get_queue_item(approved.id)
        bundle_dirs = sorted(path.name for path in output_root.iterdir())
        library_paths = [bundle.bundle_path for bundle in state.list_library()]

        monkeypatch.setattr(automation, "attach_provenance_to_bundle", real_attach)
        replayed = state.approve_queue_item(approved.id)
        replay = state.ingest_queue_item(
            replayed.id,
            output_root,
            transcriber_command=first_command,
        )

    # A failed rerun of an already-completed queue item must not demote the earlier
    # success: the row keeps the original bundle id, whose bundle (and library row)
    # survives, and only the new bundle is removed.
    assert preserved.state is QueueState.COMPLETED
    assert preserved.bundle_id == first.manifest.bundle_id
    assert preserved.last_error is None
    assert first.bundle_dir.is_dir()
    assert bundle_dirs == [first.manifest.bundle_id]
    assert library_paths == [str(first.bundle_dir.resolve())]
    assert replay.bundle_dir == first.bundle_dir
    assert replay.manifest.bundle_id == first.manifest.bundle_id


def _strip_provenance(source_json: Path) -> None:
    """Simulate a crash before provenance was attached to a committed bundle."""

    payload = json.loads(source_json.read_text(encoding="utf-8"))
    payload.pop("provenance", None)
    source_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _staleify_manifest_source_digest(bundle_dir: Path) -> None:
    """Simulate a crash between the source.json rewrite and the manifest patch."""

    manifest = Manifest.load(bundle_dir)
    acquire = manifest.stages[StageName.ACQUIRE]
    acquire.outputs = [
        ArtifactRef(path=output.path, sha256="0" * 64, bytes=output.bytes)
        if output.path == "source.json"
        else output
        for output in acquire.outputs
    ]
    manifest.save(bundle_dir)


def _manifest_source_digest(manifest: Manifest) -> str:
    for output in manifest.stages[StageName.ACQUIRE].outputs:
        if output.path == "source.json":
            return output.sha256
    raise AssertionError("manifest acquire stage does not record source.json")


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_transcriber_script(path: Path, stdout: str, *, exit_code: int = 0) -> Path:
    path.write_text(
        f"import sys\nsys.stdout.write({stdout!r})\nraise SystemExit({exit_code})\n",
        encoding="utf-8",
    )
    return path


def test_queue_same_command_rerun_into_new_root_provenance_failure_restores_completion(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    source_dir = tmp_path / "source"
    copy_media_without_sidecar(source_dir)
    command = f"{sys.executable} " + str(
        _write_transcriber_script(
            tmp_path / "transcriber.py",
            json.dumps({"text": "Same-command rerun transcript."}),
        )
    )
    first_root = tmp_path / "bundles"
    second_root = tmp_path / "other-bundles"

    def fail_attach(*args: object, **kwargs: object) -> None:
        raise OSError("synthetic provenance write failure")

    with open_state(tmp_path / "state.sqlite") as state:
        source = state.add_local_folder_source("talks", source_dir)
        queue_item = state.scan_source(source.id).queued[0]
        approved = state.approve_queue_item(queue_item.id)
        first = state.ingest_queue_item(
            approved.id,
            first_root,
            transcriber_command=command,
        )

        reapproved = state.approve_queue_item(approved.id)
        monkeypatch.setattr(automation, "attach_provenance_to_bundle", fail_attach)
        with pytest.raises(OSError, match="synthetic provenance write failure"):
            state.ingest_queue_item(
                reapproved.id,
                second_root,
                transcriber_command=command,
            )
        restored = state.get_queue_item(reapproved.id)
        library = state.get_library_bundle(first.manifest.bundle_id)

    # Same deterministic bundle id, different output root: the library row was
    # repointed to the new directory before provenance failed, so restoration
    # must derive from the pre-update row, not the current one.
    assert restored.state is QueueState.COMPLETED
    assert restored.bundle_id == first.manifest.bundle_id
    assert restored.last_error is None
    assert first.bundle_dir.is_dir()
    assert not (second_root / first.bundle_dir.name).exists()
    assert Path(library.bundle_path) == first.bundle_dir


# --- Queue FSM legal-transition matrix -------------------------------------
#
# The legal-source table: approve and skip stay legal from every non-terminal
# state (idempotent self-transitions included, so operator commands can be
# repeated safely); retry is legal from FAILED only, because retrying a
# discovered, approved, skipped, or completed item either loops a state it is
# already in or silently discards a recorded outcome; UNSUPPORTED remains
# terminal for all verbs.

QUEUE_VERBS = ("approve", "skip", "retry")

EXPECTED_LEGAL_SOURCES: dict[str, frozenset[QueueState]] = {
    "approve": frozenset(
        {
            QueueState.DISCOVERED,
            QueueState.APPROVED,
            QueueState.SKIPPED,
            QueueState.FAILED,
            QueueState.COMPLETED,
        }
    ),
    "skip": frozenset(
        {
            QueueState.DISCOVERED,
            QueueState.APPROVED,
            QueueState.SKIPPED,
            QueueState.FAILED,
            QueueState.COMPLETED,
        }
    ),
    "retry": frozenset({QueueState.FAILED}),
}

VERB_TARGETS: dict[str, QueueState] = {
    "approve": QueueState.APPROVED,
    "skip": QueueState.SKIPPED,
    "retry": QueueState.DISCOVERED,
}


def _force_queue_state(
    state: automation.AutomationState,
    queue_item_id: str,
    target: QueueState,
) -> None:
    """Park a queue item in `target` without going through a guarded verb."""

    connection = getattr(state, "_connection")  # noqa: B009
    connection.execute(
        "UPDATE queue_items SET state = ? WHERE id = ?",
        (target.value, queue_item_id),
    )
    connection.commit()


def _queued_item_id(state: automation.AutomationState, source_dir: Path) -> str:
    source = state.add_local_folder_source("talks", source_dir)
    return state.scan_source(source.id).queued[0].id


@pytest.mark.parametrize("verb", QUEUE_VERBS)
@pytest.mark.parametrize("source_state", list(QueueState))
def test_queue_transition_matrix(tmp_path: Path, verb: str, source_state: QueueState) -> None:
    source_dir = tmp_path / "source"
    copy_fixture(source_dir)

    with open_state(tmp_path / "state.sqlite") as state:
        queue_item_id = _queued_item_id(state, source_dir)
        _force_queue_state(state, queue_item_id, source_state)
        transition = getattr(state, f"{verb}_queue_item")

        if source_state in EXPECTED_LEGAL_SOURCES[verb]:
            item = transition(queue_item_id)
            assert item.state is VERB_TARGETS[verb]
            assert state.get_queue_item(queue_item_id).state is VERB_TARGETS[verb]
        else:
            with pytest.raises(AutomationError):
                transition(queue_item_id)
            assert state.get_queue_item(queue_item_id).state is source_state


@pytest.mark.parametrize(
    "source_state",
    [QueueState.DISCOVERED, QueueState.APPROVED, QueueState.SKIPPED, QueueState.COMPLETED],
)
def test_retry_from_non_failed_state_is_rejected(tmp_path: Path, source_state: QueueState) -> None:
    source_dir = tmp_path / "source"
    copy_fixture(source_dir)

    with open_state(tmp_path / "state.sqlite") as state:
        queue_item_id = _queued_item_id(state, source_dir)
        _force_queue_state(state, queue_item_id, source_state)

        with pytest.raises(AutomationError, match="retry is only legal from state failed"):
            state.retry_queue_item(queue_item_id)

        assert state.get_queue_item(queue_item_id).state is source_state


@pytest.mark.parametrize("verb", QUEUE_VERBS)
def test_terminal_state_rejection_message_is_unchanged(tmp_path: Path, verb: str) -> None:
    source_dir = tmp_path / "source"
    copy_fixture(source_dir)

    with open_state(tmp_path / "state.sqlite") as state:
        queue_item_id = _queued_item_id(state, source_dir)
        _force_queue_state(state, queue_item_id, QueueState.UNSUPPORTED)

        with pytest.raises(
            AutomationError,
            match=f"terminal state unsupported; {verb} is not a legal transition",
        ):
            getattr(state, f"{verb}_queue_item")(queue_item_id)


def test_illegal_retry_leaves_no_open_write_transaction(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    copy_fixture(source_dir)

    with open_state(tmp_path / "state.sqlite") as state:
        queue_item_id = _queued_item_id(state, source_dir)
        _force_queue_state(state, queue_item_id, QueueState.COMPLETED)

        # Force the guarded zero-row UPDATE to run by hiding the true state from
        # the pre-check, as a concurrent writer would.
        real_get = automation.AutomationState.get_queue_item
        stale = [replace(real_get(state, queue_item_id), state=QueueState.FAILED)]

        def stale_get(self: automation.AutomationState, item_id: str) -> automation.QueueItem:
            if stale:
                return stale.pop(0)
            return real_get(self, item_id)

        with MonkeyPatch.context() as patch:
            patch.setattr(automation.AutomationState, "get_queue_item", stale_get)
            with pytest.raises(AutomationError, match="retry is only legal from state failed"):
                state.retry_queue_item(queue_item_id)

        connection = getattr(state, "_connection")  # noqa: B009
        assert connection.in_transaction is False
        assert state.get_queue_item(queue_item_id).state is QueueState.COMPLETED


def test_queue_uses_approved_sidecar_when_original_changes_at_processing(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    media = copy_fixture(tmp_path / "source")
    sidecar = media.with_suffix(".transcript.txt")
    original = sidecar.read_bytes()
    original_ingest = ingest_module.PreparedLocalIngest.ingest
    injections: list[bool] = []

    def change_then_ingest(
        prepared: ingest_module.PreparedLocalIngest, output_root: Path
    ) -> ingest_module.IngestResult:
        injections.append(True)
        sidecar.write_bytes(b"unapproved processing-boundary evidence\n")
        return original_ingest(prepared, output_root)

    with open_state(tmp_path / "state.sqlite") as state:
        source = state.add_local_folder_source("talks", media.parent)
        state.scan_source(source.id)
        queued = state.list_queue()[0]
        state.approve_queue_item(queued.id)
        monkeypatch.setattr(ingest_module.PreparedLocalIngest, "ingest", change_then_ingest)
        result = state.ingest_queue_item(queued.id, tmp_path / "bundles")
        assert injections == [True]
        assert (result.bundle_dir / "transcript/transcript.md").read_bytes() == original
        assert state.get_queue_item(queued.id).content_sha256 == queued.content_sha256


@pytest.mark.parametrize("change", ["appears", "disappears", "changes-during-capture"])
def test_queue_snapshot_mismatch_refuses_before_processing(
    tmp_path: Path, monkeypatch: MonkeyPatch, change: str
) -> None:
    media = copy_fixture(tmp_path / "source")
    sidecar = media.with_suffix(".transcript.txt")
    if change == "appears":
        sidecar.unlink()
    capture_read = Path.read_bytes
    injections: list[bool] = []

    def change_and_read(path: Path) -> bytes:
        if path == sidecar:
            injections.append(True)
            sidecar.write_bytes(b"changed while capturing synthetic sidecar\n")
        return capture_read(path)

    def forbidden_normalize(input_path: Path, output: Path) -> None:
        raise AssertionError("processing began before approval comparison")

    with open_state(tmp_path / "state.sqlite") as state:
        source = state.add_local_folder_source("talks", media.parent)
        queued = state.scan_source(source.id).queued[0]
        state.approve_queue_item(queued.id)
        if change == "appears":
            sidecar.write_bytes(b"new sidecar\n")
        elif change == "disappears":
            sidecar.unlink()
        else:
            monkeypatch.setattr(Path, "read_bytes", change_and_read)
        monkeypatch.setattr(ingest_module, "_normalize_to_canonical_wav", forbidden_normalize)
        with pytest.raises(AutomationError, match="changed since queue approval"):
            state.ingest_queue_item(queued.id, tmp_path / "bundles", transcriber_command="unused")
        if change == "changes-during-capture":
            assert injections == [True]
        assert state.get_queue_item(queued.id).state is QueueState.FAILED
        assert not (tmp_path / "bundles").exists()


def test_one_shot_registry_uses_captured_identity_after_planning_mutation(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    media = copy_fixture(tmp_path / "source")
    original_sidecar = media.with_suffix(".transcript.txt").read_bytes()
    expected_digest, expected_size = automation.approval_digest_and_media_size(media)
    original_planner = ingest_module.PreparedLocalIngest.planned_bundle_id
    injections: list[bool] = []

    def mutate_after_plan(prepared: ingest_module.PreparedLocalIngest) -> str | None:
        result = original_planner(prepared)
        injections.append(True)
        media.with_suffix(".transcript.txt").write_bytes(b"new after planned snapshot\n")
        return result

    monkeypatch.setattr(ingest_module.PreparedLocalIngest, "planned_bundle_id", mutate_after_plan)
    with open_state(tmp_path / "state.sqlite") as state:
        result = state.ingest_one_shot(media, tmp_path / "bundles")
        queue_item = state.list_queue()[0]
        item = state.get_source_item(queue_item.source_item_id)
        assert injections == [True]
        assert item.sha256 == queue_item.content_sha256 == expected_digest
        assert item.size_bytes == expected_size
        assert item.absolute_path == str(media.resolve())
        assert (result.bundle_dir / "transcript/transcript.md").read_bytes() == original_sidecar


@pytest.mark.parametrize("sidecar_present", [False, True])
def test_command_approval_includes_even_an_unused_sidecar(
    tmp_path: Path, sidecar_present: bool
) -> None:
    media = copy_media_without_sidecar(tmp_path / "source")
    if sidecar_present:
        media.with_suffix(".transcript.txt").write_bytes(b"\xff unused sidecar bytes")
    expected_digest, _ = automation.approval_digest_and_media_size(media)
    transcriber = _write_transcriber_script(
        tmp_path / "transcriber.py", '{"text":"command evidence"}'
    )
    with open_state(tmp_path / "state.sqlite") as state:
        source = state.add_local_folder_source("talks", media.parent)
        queued = state.scan_source(source.id).queued[0]
        state.approve_queue_item(queued.id)
        result = state.ingest_queue_item(
            queued.id, tmp_path / "bundles", transcriber_command=f"{sys.executable} {transcriber}"
        )
        assert state.get_queue_item(queued.id).content_sha256 == expected_digest
        assert state.get_queue_item(queued.id).state is QueueState.COMPLETED
        assert (
            json.loads((result.bundle_dir / "source.json").read_text())["transcript_sidecar"]
            is None
        )
        assert (result.bundle_dir / "transcript/transcript.md").read_text() == "command evidence\n"


@pytest.mark.parametrize("pivot", [False, True])
def test_one_shot_alias_keeps_captured_registry_and_lexical_evidence(
    tmp_path: Path, monkeypatch: MonkeyPatch, pivot: bool
) -> None:
    root = tmp_path.resolve()
    target_a, target_b = root / "A.wav", root / "B.wav"
    original = SYNTHETIC_TALK.read_bytes()
    target_a.write_bytes(original)
    target_b.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
    alias = root / "Selected.wav"
    alias.symlink_to(target_a)
    selected_text = b"synthetic Selected sidecar\n"
    alias.with_suffix(".transcript.txt").write_bytes(selected_text)
    target_a.with_suffix(".transcript.txt").write_bytes(b"different target A sidecar\n")
    planner = ingest_module.PreparedLocalIngest.planned_bundle_id
    normalizer = ingest_module._normalize_to_canonical_wav  # pyright: ignore[reportPrivateUsage]
    events: list[str] = []

    def pivot_after_capture(prepared: ingest_module.PreparedLocalIngest) -> str | None:
        planned = planner(prepared)
        if pivot:
            alias.unlink()
            alias.symlink_to(target_b)
            events.append("pivoted to B")
        return planned

    def restore_at_processing(captured: Path, output: Path) -> None:
        assert captured.name == "Selected.wav"
        assert captured.read_bytes() == original
        if pivot:
            assert alias.resolve() == target_b
            alias.unlink()
            alias.symlink_to(target_a)
        events.append("consumed A")
        normalizer(captured, output)

    monkeypatch.setattr(ingest_module.PreparedLocalIngest, "planned_bundle_id", pivot_after_capture)
    monkeypatch.setattr(ingest_module, "_normalize_to_canonical_wav", restore_at_processing)
    with open_state(root / "state.sqlite") as state:
        result = state.ingest_one_shot(alias, root / "bundles")
        queued = state.list_queue()[0]
        item = state.get_source_item(queued.source_item_id)
        source = state.get_source(queued.source_id)
        assert events == (["pivoted to B", "consumed A"] if pivot else ["consumed A"])
        assert (result.bundle_dir / "media/audio.wav").read_bytes() == original
        assert (result.bundle_dir / "transcript/transcript.md").read_bytes() == selected_text
        assert result.manifest.bundle_id.startswith("selected-")
        assert result.manifest.source.title == "Selected"
        assert Path(source.root_path) == target_a
        assert Path(item.absolute_path) == target_a
        assert source.id == automation.make_source_id("one-shot", str(target_a))
        assert source.name.startswith("one-shot:A.wav:")
        assert item.id == automation.make_source_item_id(source.id, "A.wav")
        assert item.relative_path == "A.wav"
        assert queued.state is QueueState.COMPLETED
        assert alias.resolve() == target_a


@pytest.mark.parametrize("replay", [False, True], ids=["fresh", "early-replay"])
@pytest.mark.parametrize("changed_input", ["media", "sidecar"])
def test_approved_capture_refreshes_latest_source_observation(
    tmp_path: Path, monkeypatch: MonkeyPatch, replay: bool, changed_input: str
) -> None:
    media = copy_fixture(tmp_path / "source")
    sidecar = media.with_suffix(".transcript.txt")
    original_media = media.read_bytes()
    original_sidecar = sidecar.read_bytes()
    with open_state(tmp_path / "state.sqlite") as state:
        source = state.add_local_folder_source("talks", media.parent)
        approved = state.approve_queue_item(state.scan_source(source.id).queued[0].id)
        first = None
        if replay:
            first = state.ingest_queue_item(approved.id, tmp_path / "bundles")
            state.approve_queue_item(approved.id)
        changed = media if changed_input == "media" else sidecar
        changed.write_bytes(changed.read_bytes() + b"changed B")
        other = state.scan_source(source.id).queued[0]
        stale = replace(
            state.get_source_item(approved.source_item_id),
            metadata={"synthetic": "preserve"},
            present=False,
        )
        state._upsert_source_item(stale, created_at=stale.created_at)  # pyright: ignore[reportPrivateUsage]
        state._connection.commit()  # pyright: ignore[reportPrivateUsage]
        media.write_bytes(original_media)
        sidecar.write_bytes(original_sidecar)
        os.utime(media, ns=(1_700_000_000_000_000_000, 1_700_000_000_000_000_000))
        captured_mtime = media.stat().st_mtime_ns

        normalizer = ingest_module._normalize_to_canonical_wav  # pyright: ignore[reportPrivateUsage]
        normalization_calls = 0

        def consume_a(captured: Path, output: Path) -> None:
            nonlocal normalization_calls
            normalization_calls += 1
            assert captured.read_bytes() == original_media
            normalizer(captured, output)

        monkeypatch.setattr(ingest_module, "_normalize_to_canonical_wav", consume_a)
        result = state.ingest_queue_item(approved.id, tmp_path / "bundles")

        assert normalization_calls == (0 if replay else 1)
        assert (result.bundle_dir / "media/audio.wav").read_bytes() == original_media
        assert (result.bundle_dir / "transcript/transcript.md").read_bytes() == original_sidecar
        if replay:
            assert first is not None
            assert result.bundle_dir == first.bundle_dir
        refreshed = state.get_source_item(stale.id)
        assert refreshed.sha256 == approved.content_sha256 != stale.sha256
        assert refreshed.size_bytes == len(original_media)
        assert refreshed.mtime_ns == captured_mtime
        assert refreshed.present is True
        assert refreshed.updated_at != stale.updated_at
        assert (
            replace(
                refreshed,
                sha256=stale.sha256,
                size_bytes=stale.size_bytes,
                mtime_ns=stale.mtime_ns,
                present=stale.present,
                updated_at=stale.updated_at,
            )
            == stale
        )
        assert state.get_queue_item(other.id) == other
        assert state.get_queue_item(approved.id).content_sha256 == approved.content_sha256
        assert state.get_library_bundle(result.manifest.bundle_id).status.value == "ready"


@pytest.mark.parametrize("mismatch", [False, True], ids=["processing-failure", "mismatch"])
def test_capture_observation_refresh_boundary(
    tmp_path: Path, monkeypatch: MonkeyPatch, mismatch: bool
) -> None:
    media = copy_fixture(tmp_path / "source")
    original = media.read_bytes()
    calls = 0

    def fail_normalization(*args: object, **kwargs: object) -> None:
        nonlocal calls
        calls += 1
        raise IngestError("synthetic processing failure")

    monkeypatch.setattr(ingest_module, "_normalize_to_canonical_wav", fail_normalization)
    with open_state(tmp_path / "state.sqlite") as state:
        source = state.add_local_folder_source("talks", media.parent)
        approved = state.approve_queue_item(state.scan_source(source.id).queued[0].id)
        media.write_bytes(original + b"changed B")
        other = state.scan_source(source.id).queued[0]
        before = state.get_source_item(approved.source_item_id)
        if not mismatch:
            media.write_bytes(original)
        with pytest.raises(AutomationError if mismatch else IngestError):
            state.ingest_queue_item(approved.id, tmp_path / "bundles")
        after = state.get_source_item(before.id)
        assert calls == (0 if mismatch else 1)
        if mismatch:
            assert after == before
        else:
            assert after.sha256 == approved.content_sha256 != before.sha256
            assert after.size_bytes == len(original)
        assert state.get_queue_item(other.id) == other


def _bundle_tree(path: Path) -> dict[str, bytes]:
    return {
        str(item.relative_to(path)): item.read_bytes() for item in path.rglob("*") if item.is_file()
    }


@pytest.mark.parametrize("one_shot", [False, True], ids=["queue", "one-shot"])
@pytest.mark.parametrize("planned_a", [False, True], ids=["command-a", "planned-a"])
def test_replay_selects_retained_a_after_b(tmp_path: Path, one_shot: bool, planned_a: bool) -> None:
    media = copy_fixture(tmp_path / "source")
    script = tmp_path / "transcriber.py"
    counter = tmp_path / "calls"
    command = f"{sys.executable} {script}"

    def configure(text: str) -> None:
        script.write_text(
            "from pathlib import Path\n"
            f"with Path({str(counter)!r}).open('a') as calls: calls.write('called\\n')\n"
            f"print({json.dumps({'text': text})!r})\n",
            encoding="utf-8",
        )

    with open_state(tmp_path / "state.sqlite") as state:
        queue_id: str | None = None
        if not one_shot:
            source = state.add_local_folder_source("talks", media.parent)
            queue_id = state.scan_source(source.id).queued[0].id

        def run(command_arg: str | None) -> ingest_module.IngestResult:
            if one_shot:
                return state.ingest_one_shot(
                    media, tmp_path / "bundles", transcriber_command=command_arg
                )
            assert queue_id is not None
            state.approve_queue_item(queue_id)
            return state.ingest_queue_item(
                queue_id, tmp_path / "bundles", transcriber_command=command_arg
            )

        configure("Command A transcript.")
        first = run(None if planned_a else command)
        first_tree = _bundle_tree(first.bundle_dir)
        configure("Command B transcript.")
        second = run(command)
        assert first.manifest.bundle_id != second.manifest.bundle_id
        assert "Command B transcript." in (
            second.bundle_dir / "transcript/transcript.md"
        ).read_text(encoding="utf-8")
        second_tree = _bundle_tree(second.bundle_dir)
        rows = state._connection.execute(  # pyright: ignore[reportPrivateUsage]
            "SELECT * FROM library_bundles ORDER BY bundle_id"
        ).fetchall()
        configure("Command A transcript.")
        replay = run(None if planned_a else command)

        assert replay.bundle_dir == first.bundle_dir
        selected = state.list_queue()[0]
        assert selected.bundle_id == first.manifest.bundle_id
        assert selected.state is QueueState.COMPLETED
        assert selected.last_error is None
        assert _bundle_tree(first.bundle_dir) == first_tree
        assert _bundle_tree(second.bundle_dir) == second_tree
        assert (
            state._connection.execute(  # pyright: ignore[reportPrivateUsage]
                "SELECT * FROM library_bundles ORDER BY bundle_id"
            ).fetchall()
            == rows
        )
        assert counter.read_text().splitlines() == ["called"] * (1 if planned_a else 3)


@pytest.mark.parametrize("one_shot", [False, True], ids=["queue", "one-shot"])
def test_generic_collision_text_does_not_authorize_replay(
    tmp_path: Path, monkeypatch: MonkeyPatch, one_shot: bool
) -> None:
    media = copy_fixture(tmp_path / "source")
    calls = 0
    with open_state(tmp_path / "state.sqlite") as state:
        first = state.ingest_one_shot(media, tmp_path / "bundles")
        queue = state.list_queue()[0]
        before = _bundle_tree(first.bundle_dir)

        def spoof(
            prepared: ingest_module.PreparedLocalIngest, output: Path
        ) -> ingest_module.IngestResult:
            nonlocal calls
            calls += 1
            raise IngestError(f"bundle already exists: {first.bundle_dir}")

        monkeypatch.setattr(ingest_module.PreparedLocalIngest, "ingest", spoof)
        with pytest.raises(IngestError, match="bundle already exists"):
            if one_shot:
                state.ingest_one_shot(media, tmp_path / "bundles", transcriber_command="unused")
            else:
                state.approve_queue_item(queue.id)
                state.ingest_queue_item(
                    queue.id, tmp_path / "bundles", transcriber_command="unused"
                )
        assert calls == 1
        assert _bundle_tree(first.bundle_dir) == before


@pytest.mark.parametrize("one_shot", [False, True], ids=["queue", "one-shot"])
@pytest.mark.parametrize(
    "damage",
    [
        "unregistered",
        "queue",
        "source",
        "item",
        "missing",
        "manifest-id",
        "malformed",
        "recursive",
        "injected-recursion",
        "missing-manifest",
        "symlink-manifest",
    ],
)
def test_typed_candidate_refusal_never_falls_back_to_selected_b(
    tmp_path: Path, monkeypatch: MonkeyPatch, one_shot: bool, damage: str
) -> None:
    media = copy_fixture(tmp_path / "source")
    script = _write_transcriber_script(
        tmp_path / "transcriber.py", json.dumps({"text": "B output"})
    )
    calls = 0
    with open_state(tmp_path / "state.sqlite") as state:
        first = state.ingest_one_shot(media, tmp_path / "bundles")
        selected = state.ingest_one_shot(
            media, tmp_path / "bundles", transcriber_command=f"{sys.executable} {script}"
        )
        queue = state.list_queue()[0]
        foreign = state.ingest_one_shot(
            copy_fixture(tmp_path / "foreign", "other.wav"), tmp_path / "foreign-bundles"
        )
        foreign_row = state.get_library_bundle(foreign.manifest.bundle_id)
        connection = state._connection  # pyright: ignore[reportPrivateUsage]
        if damage == "unregistered":
            connection.execute(
                "DELETE FROM library_bundles WHERE bundle_id = ?", (first.manifest.bundle_id,)
            )
        elif damage in {"queue", "source", "item"}:
            field, value = {
                "queue": ("queue_item_id", foreign_row.queue_item_id),
                "source": ("source_id", foreign_row.source_id),
                "item": ("source_item_id", foreign_row.source_item_id),
            }[damage]
            connection.execute(
                f"UPDATE library_bundles SET {field} = ? WHERE bundle_id = ?",
                (value, first.manifest.bundle_id),
            )
        elif damage == "missing":
            first.bundle_dir.rename(tmp_path / "held-a")
        else:
            manifest_path = first.bundle_dir / MANIFEST_NAME
            if damage == "recursive":
                manifest_path.write_text("[" * 100_000 + "0" + "]" * 100_000)
                with pytest.raises((ValueError, RecursionError)):
                    Manifest.load(first.bundle_dir)
            elif damage == "injected-recursion":
                pass
            elif damage == "malformed":
                manifest_path.write_text("{invalid JSON")
            elif damage == "missing-manifest":
                manifest_path.unlink()
            elif damage == "symlink-manifest":
                external = tmp_path / "external-manifest.json"
                manifest_path.rename(external)
                manifest_path.symlink_to(external)
            else:
                payload = json.loads(manifest_path.read_text())
                payload["bundle_id"] = "different-manifest-id"
                manifest_path.write_text(json.dumps(payload))
        connection.commit()
        before = _bundle_tree(tmp_path / "bundles")
        manifest_calls = 0
        real_load = Manifest.load

        def recursive_load(cls: type[Manifest], bundle_dir: Path) -> Manifest:
            nonlocal manifest_calls
            if bundle_dir == first.bundle_dir:
                manifest_calls += 1
                raise RecursionError("synthetic candidate manifest recursion")
            return real_load(bundle_dir)

        if damage == "injected-recursion":
            monkeypatch.setattr(Manifest, "load", classmethod(recursive_load))

        def collide(
            prepared: ingest_module.PreparedLocalIngest, output: Path
        ) -> ingest_module.IngestResult:
            nonlocal calls
            calls += 1
            raise ingest_module.BundleExistsError(first.manifest.bundle_id, first.bundle_dir)

        monkeypatch.setattr(ingest_module.PreparedLocalIngest, "ingest", collide)
        with pytest.raises(ingest_module.BundleExistsError):
            if one_shot:
                state.ingest_one_shot(media, tmp_path / "bundles", transcriber_command="unused")
            else:
                state.approve_queue_item(queue.id)
                state.ingest_queue_item(
                    queue.id, tmp_path / "bundles", transcriber_command="unused"
                )
        assert calls == 1
        assert manifest_calls == (1 if damage == "injected-recursion" else 0)
        assert state.get_queue_item(queue.id).bundle_id == selected.manifest.bundle_id
        assert _bundle_tree(tmp_path / "bundles") == before


@pytest.mark.parametrize("one_shot", [False, True], ids=["queue", "one-shot"])
def test_collision_returns_registered_root_without_adopting_collision_root(
    tmp_path: Path, one_shot: bool
) -> None:
    media = copy_fixture(tmp_path / "source")
    script = _write_transcriber_script(
        tmp_path / "transcriber.py", json.dumps({"text": "Same output"})
    )
    command = f"{sys.executable} {script}"
    with open_state(tmp_path / "state.sqlite") as state:
        source = state.add_local_folder_source("talks", media.parent)
        queue = state.scan_source(source.id).queued[0]

        def run(root: Path) -> ingest_module.IngestResult:
            if one_shot:
                return state.ingest_one_shot(media, root, transcriber_command=command)
            state.approve_queue_item(queue.id)
            return state.ingest_queue_item(queue.id, root, transcriber_command=command)

        first = run(tmp_path / "root1")
        second = run(tmp_path / "root2")
        before_first = _bundle_tree(first.bundle_dir)
        before_second = _bundle_tree(second.bundle_dir)
        replay = run(tmp_path / "root1")
        assert replay.bundle_dir == second.bundle_dir != first.bundle_dir
        assert state.get_library_bundle(first.manifest.bundle_id).bundle_path == str(
            second.bundle_dir
        )
        assert _bundle_tree(first.bundle_dir) == before_first
        assert _bundle_tree(second.bundle_dir) == before_second


@pytest.mark.parametrize("one_shot", [False, True], ids=["queue", "one-shot"])
@pytest.mark.parametrize(
    "prior", ["valid", "first", "deleted", "unregistered", "queue", "source", "item"]
)
def test_failed_invocation_preserves_only_surviving_owned_completion(
    tmp_path: Path, one_shot: bool, prior: str
) -> None:
    import shutil

    media = copy_fixture(tmp_path / "source")
    script = _write_transcriber_script(tmp_path / "fail.py", "", exit_code=9)
    command = f"{sys.executable} {script}"
    with open_state(tmp_path / "state.sqlite") as state:
        if prior == "first":
            source = state.add_local_folder_source("talks", media.parent)
            queue = state.scan_source(source.id).queued[0]
        else:
            result = state.ingest_one_shot(media, tmp_path / "bundles")
            queue = state.list_queue()[0]
            if prior == "deleted":
                shutil.rmtree(result.bundle_dir)
            elif prior == "unregistered":
                state._connection.execute(  # pyright: ignore[reportPrivateUsage]
                    "DELETE FROM library_bundles WHERE bundle_id = ?", (result.manifest.bundle_id,)
                )
            elif prior in {"queue", "source", "item"}:
                foreign = state.ingest_one_shot(
                    copy_fixture(tmp_path / "foreign", "other.wav"), tmp_path / "other-bundles"
                )
                row = state.get_library_bundle(foreign.manifest.bundle_id)
                field, value = {
                    "queue": ("queue_item_id", row.queue_item_id),
                    "source": ("source_id", row.source_id),
                    "item": ("source_item_id", row.source_item_id),
                }[prior]
                state._connection.execute(  # pyright: ignore[reportPrivateUsage]
                    f"UPDATE library_bundles SET {field} = ? WHERE bundle_id = ?",
                    (value, result.manifest.bundle_id),
                )
            state._connection.commit()  # pyright: ignore[reportPrivateUsage]
        trees = _bundle_tree(tmp_path / "bundles")
        rows = state._connection.execute(  # pyright: ignore[reportPrivateUsage]
            "SELECT * FROM library_bundles ORDER BY bundle_id"
        ).fetchall()  # pyright: ignore[reportPrivateUsage]
        with pytest.raises(IngestError):
            if one_shot:
                state.ingest_one_shot(media, tmp_path / "bundles", transcriber_command=command)
            else:
                state.approve_queue_item(queue.id)
                state.ingest_queue_item(queue.id, tmp_path / "bundles", transcriber_command=command)
        after = (
            next(item for item in state.list_queue() if item.source_item_id == queue.source_item_id)
            if not (one_shot and prior == "first")
            else next(item for item in state.list_queue() if item.state is QueueState.FAILED)
        )
        assert after.state is (QueueState.COMPLETED if prior == "valid" else QueueState.FAILED)
        assert after.content_sha256 == queue.content_sha256
        assert after.attempts == (0 if prior == "valid" else 1)
        assert (after.last_error is None) == (prior == "valid")
        assert _bundle_tree(tmp_path / "bundles") == trees
        assert (
            state._connection.execute("SELECT * FROM library_bundles ORDER BY bundle_id").fetchall()  # pyright: ignore[reportPrivateUsage]
            == rows
        )  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize(
    "kind", ["json", "utf8", "integer", "deep", "source-recursion", "manifest-recursion"]
)
def test_repair_probe_contains_unreadable_input(
    tmp_path: Path, monkeypatch: MonkeyPatch, kind: str
) -> None:
    from lectern import provenance

    with open_state(tmp_path / "state.sqlite") as state:
        result = state.ingest_one_shot(copy_fixture(tmp_path / "source"), tmp_path / "bundles")
    source = result.bundle_dir / "source.json"
    cases = {
        "json": b"{bad",
        "utf8": b"\xff",
        "integer": b'{"number":' + b"1" * 5000 + b"}",
        "deep": b"[" * 100_000 + b"0" + b"]" * 100_000,
    }
    calls = 0
    if kind in cases:
        source.write_bytes(cases[kind])
    elif kind == "source-recursion":
        real_read = Path.read_text

        def recursive_read(path: Path, *args: object, **kwargs: Any) -> str:
            nonlocal calls
            if path == source:
                calls += 1
                raise RecursionError("synthetic source recursion")
            return real_read(path, **kwargs)

        monkeypatch.setattr(Path, "read_text", recursive_read)
    else:

        def recursive_manifest(cls: type[Manifest], path: Path) -> Manifest:
            nonlocal calls
            calls += 1
            assert path == result.bundle_dir
            raise RecursionError("synthetic manifest recursion")

        monkeypatch.setattr(Manifest, "load", classmethod(recursive_manifest))
    before = _bundle_tree(result.bundle_dir)
    assert provenance.bundle_provenance_needs_repair(result.bundle_dir) is False
    assert calls == (1 if kind.endswith("recursion") else 0)
    assert _bundle_tree(result.bundle_dir) == before


@pytest.mark.parametrize("one_shot", [False, True])
@pytest.mark.parametrize(
    "payload",
    [b"{bad", b"\xff", b'{"number":' + b"1" * 5000 + b"}", b"[" * 100_000 + b"0" + b"]" * 100_000],
    ids=["json", "utf8", "integer", "deep"],
)
def test_unreadable_source_replay_remains_unchanged_and_nonready(
    tmp_path: Path, one_shot: bool, payload: bytes
) -> None:
    media = copy_fixture(tmp_path / "source")
    with open_state(tmp_path / "state.sqlite") as state:
        first = state.ingest_one_shot(media, tmp_path / "bundles")
        queue = state.list_queue()[0]
        (first.bundle_dir / "source.json").write_bytes(payload)
        before = _bundle_tree(first.bundle_dir)
        if one_shot:
            result = state.ingest_one_shot(media, tmp_path / "bundles")
        else:
            state.approve_queue_item(queue.id)
            result = state.ingest_queue_item(queue.id, tmp_path / "bundles")
        assert result.bundle_dir == first.bundle_dir
        assert _bundle_tree(first.bundle_dir) == before
        assert (
            state.get_library_bundle(first.manifest.bundle_id).status.value == "needs-reprocessing"
        )
        with pytest.raises((AutomationError, ValueError)):
            state.cite_segment(first.manifest.bundle_id, 0)
    with open_state(tmp_path / "state.sqlite") as state:
        assert state.search_segments("knowledge") == []


@pytest.mark.parametrize("ownership", ["valid", "queue", "source", "item"])
def test_prior_registration_restoration_requires_all_ownership_fields(
    tmp_path: Path, ownership: str
) -> None:
    with open_state(tmp_path / "state.sqlite") as state:
        result = state.ingest_one_shot(copy_fixture(tmp_path / "source"), tmp_path / "bundles")
        queue = state.list_queue()[0]
        previous = state.get_library_bundle(result.manifest.bundle_id)
        if ownership != "valid":
            field = {"queue": "queue_item_id", "source": "source_id", "item": "source_item_id"}[
                ownership
            ]
            previous = replace(previous, **{field: "different-owner"})
        state._connection.execute(  # pyright: ignore[reportPrivateUsage]
            "DELETE FROM library_bundles WHERE bundle_id = ?", (result.manifest.bundle_id,)
        )  # pyright: ignore[reportPrivateUsage]
        actual = state._restorable_queue_bundle_id(queue, previous)  # pyright: ignore[reportPrivateUsage]
        assert actual == (result.manifest.bundle_id if ownership == "valid" else None)
