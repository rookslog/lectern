from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import sys
from pathlib import Path
from typing import Any, cast

import pytest

from lectern import cli, migrations
from lectern.automation import open_state
from lectern.bundle import MANIFEST_NAME, Manifest
from lectern.ingest import ingest_local
from lectern.migrations import (
    BACKUP_SUFFIX,
    MARKER_NAME,
    STAGING_SUFFIX,
    MigrationError,
    prepare_bundle_migration,
)
from lectern.search import Anchor, AnchorResolution

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
SYNTHETIC_TALK = FIXTURE_DIR / "synthetic_talk.wav"
SYNTHETIC_TRANSCRIPT = FIXTURE_DIR / "synthetic_talk.transcript.txt"


def _digest(path: Path) -> tuple[str, int]:
    data = path.read_bytes()
    return hashlib.sha256(data).hexdigest(), len(data)


def _rewrite_manifest_hashes(bundle: Path, manifest: dict[str, Any]) -> None:
    for stage in manifest["stages"].values():
        for output in stage["outputs"]:
            digest, size = _digest(bundle / output["path"])
            output["sha256"] = digest
            output["bytes"] = size


def _tree_bytes(root: Path) -> dict[Path, bytes]:
    return {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def _tree_state(root: Path) -> dict[Path, tuple[str, bytes | str | None]]:
    state: dict[Path, tuple[str, bytes | str | None]] = {}
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if path.is_symlink():
            state[relative] = ("symlink", os.readlink(path))
        elif path.is_dir():
            state[relative] = ("directory", None)
        else:
            state[relative] = ("file", path.read_bytes())
    return state


def _write_legacy_shape(bundle: Path, media: Path) -> None:
    manifest_path = bundle / MANIFEST_NAME
    source_path = bundle / "source.json"
    metadata_path = bundle / "transcript" / "metadata.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = json.loads(source_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = "0.1.0"
    manifest["source"]["ref"] = str(media)
    manifest["source"].pop("bytes", None)
    source["source"]["ref"] = str(media)
    source["source"].pop("bytes", None)
    sidecar = source["transcript_sidecar"]
    assert isinstance(sidecar, dict)
    sidecar["path"] = str(media.with_suffix(".transcript.txt"))
    metadata["source_media"]["path"] = str(media)
    metadata["backend"]["path"] = str(media.with_suffix(".transcript.txt"))
    source_path.write_text(json.dumps(source, indent=2) + "\n", encoding="utf-8")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    _rewrite_manifest_hashes(bundle, manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def legacy_bundle(tmp_path: Path) -> Path:
    media = tmp_path / "Private Recordings" / "synthetic_talk.wav"
    media.parent.mkdir(parents=True)
    media.write_bytes(SYNTHETIC_TALK.read_bytes())
    media.with_suffix(".transcript.txt").write_text(
        SYNTHETIC_TRANSCRIPT.read_text(encoding="utf-8"), encoding="utf-8"
    )
    bundle = ingest_local(media, tmp_path / "bundles").bundle_dir
    _write_legacy_shape(bundle, media)
    return bundle


def registered_legacy_bundle(tmp_path: Path) -> tuple[Path, Path, str, int, Anchor]:
    media = tmp_path / "Registered Recordings" / "synthetic_talk.wav"
    media.parent.mkdir(parents=True)
    media.write_bytes(SYNTHETIC_TALK.read_bytes())
    media.with_suffix(".transcript.txt").write_text(
        SYNTHETIC_TRANSCRIPT.read_text(encoding="utf-8"), encoding="utf-8"
    )
    state_path = tmp_path / "state.sqlite"
    with open_state(state_path) as state:
        result = state.ingest_one_shot(media, tmp_path / "registered-bundles")
        hits = state.search_segments("inspectable knowledge")
        assert hits
        hit = hits[0]
        assert hit.segment_id is not None
        bundle_id = hit.bundle_id
        segment_id = hit.segment_id
        before_anchor, resolved = state.cite_segment(bundle_id, segment_id)
        assert resolved.outcome is AnchorResolution.EXACT
        assert state.get_library_bundle(bundle_id).bundle_path == str(result.bundle_dir)
    _write_legacy_shape(result.bundle_dir, media)
    return state_path, result.bundle_dir, bundle_id, segment_id, before_anchor


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return cast(dict[str, Any], value)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _repair_manifest_after_artifact_change(bundle: Path) -> None:
    manifest_path = bundle / MANIFEST_NAME
    manifest = _read_json(manifest_path)
    _rewrite_manifest_hashes(bundle, manifest)
    _write_json(manifest_path, manifest)


def _staging_role(bundle: Path) -> Path:
    return bundle.with_name(bundle.name + STAGING_SUFFIX)


def _nested_json_object(depth: int) -> dict[str, Any]:
    value: object = "leaf"
    for _ in range(depth):
        value = {"nested": value}
    assert isinstance(value, dict)
    return cast(dict[str, Any], value)


def _read_json_array(path: Path) -> list[dict[str, Any]]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, list)
    items = cast(list[object], value)
    assert all(isinstance(item, dict) for item in items)
    return cast(list[dict[str, Any]], value)


def _assert_preflight_refusal(bundle: Path, message: str) -> None:
    before = _tree_state(bundle)
    with pytest.raises(MigrationError, match=message) as error:
        prepare_bundle_migration(bundle)
    assert str(bundle) not in str(error.value)
    assert _tree_state(bundle) == before
    assert not os.path.lexists(_staging_role(bundle))


def test_prepare_builds_valid_target_without_mutating_source(tmp_path: Path) -> None:
    bundle = legacy_bundle(tmp_path)
    before = _tree_bytes(bundle)
    prepared = prepare_bundle_migration(bundle)
    assert prepared.source_version == "0.1.0"
    assert prepared.target_version == "1.0.0"
    assert Manifest.load(prepared.staging_dir).schema_version == "1.0.0"
    assert _tree_bytes(bundle) == before


def test_migrate_restarts_marker_owned_partial_staging(tmp_path: Path) -> None:
    bundle = legacy_bundle(tmp_path)
    staging = _staging_role(bundle)
    staging.mkdir(mode=0o700)
    manifest = _read_json(bundle / MANIFEST_NAME)
    _write_json(
        staging / MARKER_NAME,
        migrations._marker_payload(str(manifest["bundle_id"])),  # pyright: ignore[reportPrivateUsage]
    )

    result = migrations.migrate_bundle(bundle)

    assert result.outcome == "recovered"
    assert Manifest.load(bundle).schema_version == "1.0.0"
    assert bundle.with_name(bundle.name + BACKUP_SUFFIX).is_dir()
    assert not staging.exists()


def test_prepare_rejects_hash_mismatch_before_staging(tmp_path: Path) -> None:
    bundle = legacy_bundle(tmp_path)
    (bundle / "source.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(MigrationError, match="declared artifact integrity"):
        prepare_bundle_migration(bundle)


def test_source_path_normalizes_lexical_parent_without_following_links(
    tmp_path: Path,
) -> None:
    bundle = legacy_bundle(tmp_path)
    (bundle / "child").mkdir()
    before = _tree_state(bundle)
    normalized = migrations._source_path(  # pyright: ignore[reportPrivateUsage]
        bundle / "child" / ".."
    )
    assert normalized == bundle
    assert normalized.with_name(normalized.name + STAGING_SUFFIX).parent == bundle.parent
    assert normalized.with_name(normalized.name + BACKUP_SUFFIX).parent == bundle.parent
    assert _tree_state(bundle) == before
    assert not os.path.lexists(_staging_role(bundle))


def test_prepare_rejects_root_symlink_without_following_it(tmp_path: Path) -> None:
    bundle = legacy_bundle(tmp_path)
    alias = tmp_path / "bundle-alias"
    alias.symlink_to(bundle, target_is_directory=True)
    before = _tree_state(bundle)
    with pytest.raises(
        MigrationError, match="source role must be a real bundle directory"
    ) as error:
        prepare_bundle_migration(alias)
    assert str(alias) not in str(error.value)
    assert _tree_state(bundle) == before
    assert alias.is_symlink()
    assert not os.path.lexists(_staging_role(alias))


def test_prepare_rejects_nested_symlink_before_staging(tmp_path: Path) -> None:
    bundle = legacy_bundle(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("outside\n", encoding="utf-8")
    (bundle / "nested-link").symlink_to(outside)
    _assert_preflight_refusal(bundle, "source role contains a symlink")


@pytest.mark.parametrize("suffix", [BACKUP_SUFFIX, STAGING_SUFFIX])
@pytest.mark.parametrize("dangling", [False, True])
def test_prepare_preserves_occupied_sibling_roles(
    tmp_path: Path, suffix: str, dangling: bool
) -> None:
    bundle = legacy_bundle(tmp_path)
    collision = bundle.with_name(bundle.name + suffix)
    if dangling:
        collision.symlink_to(tmp_path / "missing-role-target")
    else:
        collision.mkdir()
        (collision / "owner.txt").write_text("unrelated\n", encoding="utf-8")
    bundle_before = _tree_state(bundle)
    collision_before = (
        ("symlink", os.readlink(collision))
        if collision.is_symlink()
        else ("directory", _tree_state(collision))
    )
    role = "backup" if suffix == BACKUP_SUFFIX else "staging"
    with pytest.raises(MigrationError, match=f"{role} role is already occupied") as error:
        prepare_bundle_migration(bundle)
    assert str(bundle) not in str(error.value)
    assert _tree_state(bundle) == bundle_before
    collision_after = (
        ("symlink", os.readlink(collision))
        if collision.is_symlink()
        else ("directory", _tree_state(collision))
    )
    assert collision_after == collision_before


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (b"{", "cannot read valid source manifest"),
        (b"[]\n", "source manifest must contain a JSON object"),
        (b"\xff", "cannot read valid source manifest"),
    ],
)
def test_prepare_rejects_unreadable_manifest_before_staging(
    tmp_path: Path, raw: bytes, message: str
) -> None:
    bundle = legacy_bundle(tmp_path)
    (bundle / MANIFEST_NAME).write_bytes(raw)
    _assert_preflight_refusal(bundle, message)


@pytest.mark.parametrize("version", [None, 1, "0.2.0", "2.0.0"])
def test_prepare_checks_version_before_other_manifest_fields(
    tmp_path: Path, version: object
) -> None:
    bundle = legacy_bundle(tmp_path)
    manifest = _read_json(bundle / MANIFEST_NAME)
    if version is None:
        manifest.pop("schema_version")
    else:
        manifest["schema_version"] = version
    manifest["bundle_id"] = None
    manifest["source"] = None
    manifest["stages"] = None
    _write_json(bundle / MANIFEST_NAME, manifest)
    _assert_preflight_refusal(bundle, "source manifest does not declare schema 0.1.0")


@pytest.mark.parametrize("bundle_id", [None, "", 3, True])
def test_prepare_rejects_malformed_bundle_id_before_staging(
    tmp_path: Path, bundle_id: object
) -> None:
    bundle = legacy_bundle(tmp_path)
    manifest = _read_json(bundle / MANIFEST_NAME)
    manifest["bundle_id"] = bundle_id
    _write_json(bundle / MANIFEST_NAME, manifest)
    _assert_preflight_refusal(bundle, "manifest bundle ID is malformed")


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (None, "manifest source is malformed"),
        ({}, "manifest source kind is malformed"),
        ({"kind": "unknown", "ref": "identity"}, "manifest source kind is malformed"),
        ({"kind": "local"}, "manifest source reference is malformed"),
        ({"kind": "local", "ref": ""}, "manifest source reference is malformed"),
        ({"kind": "local", "ref": 4}, "manifest source reference is malformed"),
    ],
)
def test_prepare_rejects_malformed_manifest_source_before_staging(
    tmp_path: Path, source: object, message: str
) -> None:
    bundle = legacy_bundle(tmp_path)
    manifest = _read_json(bundle / MANIFEST_NAME)
    manifest["source"] = source
    _write_json(bundle / MANIFEST_NAME, manifest)
    _assert_preflight_refusal(bundle, message)


@pytest.mark.parametrize("stages", [None, [], "stages"])
def test_prepare_rejects_malformed_stages_before_staging(tmp_path: Path, stages: object) -> None:
    bundle = legacy_bundle(tmp_path)
    manifest = _read_json(bundle / MANIFEST_NAME)
    manifest["stages"] = stages
    _write_json(bundle / MANIFEST_NAME, manifest)
    _assert_preflight_refusal(bundle, "manifest stages are malformed")


@pytest.mark.parametrize(
    ("stage", "message"),
    [
        (None, "manifest stage outputs are malformed"),
        ({}, "manifest stage outputs are malformed"),
        ({"outputs": {}}, "manifest stage outputs are malformed"),
        ({"outputs": [None]}, "manifest artifact record is malformed"),
    ],
)
def test_prepare_rejects_malformed_stage_output_structure(
    tmp_path: Path, stage: object, message: str
) -> None:
    bundle = legacy_bundle(tmp_path)
    manifest = _read_json(bundle / MANIFEST_NAME)
    first_name = next(iter(manifest["stages"]))
    manifest["stages"][first_name] = stage
    _write_json(bundle / MANIFEST_NAME, manifest)
    _assert_preflight_refusal(bundle, message)


@pytest.mark.parametrize("digest", [None, "bad", "A" * 64, 4])
def test_prepare_rejects_malformed_artifact_digest_before_staging(
    tmp_path: Path, digest: object
) -> None:
    bundle = legacy_bundle(tmp_path)
    manifest = _read_json(bundle / MANIFEST_NAME)
    output = next(output for stage in manifest["stages"].values() for output in stage["outputs"])
    output["sha256"] = digest
    _write_json(bundle / MANIFEST_NAME, manifest)
    _assert_preflight_refusal(bundle, "manifest artifact digest is malformed")


@pytest.mark.parametrize("size", [None, -1, True, "1"])
def test_prepare_rejects_malformed_artifact_size_before_staging(
    tmp_path: Path, size: object
) -> None:
    bundle = legacy_bundle(tmp_path)
    manifest = _read_json(bundle / MANIFEST_NAME)
    output = next(output for stage in manifest["stages"].values() for output in stage["outputs"])
    output["bytes"] = size
    _write_json(bundle / MANIFEST_NAME, manifest)
    _assert_preflight_refusal(bundle, "manifest artifact size is malformed")


@pytest.mark.parametrize(
    ("artifact_path", "message"),
    [
        (None, "declared artifact path must be text"),
        ("", "declared artifact path escapes the bundle"),
        ("../outside", "declared artifact path escapes the bundle"),
        ("/tmp/outside", "declared artifact path escapes the bundle"),
        ("missing.txt", "declared bundle artifact is missing or not a regular file"),
        ("transcript", "declared bundle artifact is missing or not a regular file"),
    ],
)
def test_prepare_rejects_invalid_artifact_paths_before_staging(
    tmp_path: Path, artifact_path: object, message: str
) -> None:
    bundle = legacy_bundle(tmp_path)
    manifest = _read_json(bundle / MANIFEST_NAME)
    output = next(output for stage in manifest["stages"].values() for output in stage["outputs"])
    output.update({"path": artifact_path, "sha256": "a" * 64, "bytes": 1})
    _write_json(bundle / MANIFEST_NAME, manifest)
    _assert_preflight_refusal(bundle, message)


def test_prepare_rejects_hash_invalid_artifact_before_staging(tmp_path: Path) -> None:
    bundle = legacy_bundle(tmp_path)
    manifest = _read_json(bundle / MANIFEST_NAME)
    output = next(output for stage in manifest["stages"].values() for output in stage["outputs"])
    output["sha256"] = "0" * 64
    _write_json(bundle / MANIFEST_NAME, manifest)
    _assert_preflight_refusal(bundle, "declared artifact integrity")


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (b"{", "cannot read valid source metadata"),
        (b"[]\n", "source metadata must contain a JSON object"),
        (b"\xff", "cannot read valid source metadata"),
    ],
)
def test_prepare_rejects_unreadable_source_document_before_staging(
    tmp_path: Path, raw: bytes, message: str
) -> None:
    bundle = legacy_bundle(tmp_path)
    (bundle / "source.json").write_bytes(raw)
    _repair_manifest_after_artifact_change(bundle)
    _assert_preflight_refusal(bundle, message)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("sha256", None, "source metadata lacks a valid sha256 identity"),
        ("sha256", "bad", "source metadata lacks a valid sha256 identity"),
        ("sha256", "A" * 64, "source metadata lacks a valid sha256 identity"),
        ("bytes", None, "source metadata lacks a valid byte size"),
        ("bytes", -1, "source metadata lacks a valid byte size"),
        ("bytes", True, "source metadata lacks a valid byte size"),
        ("bytes", "1", "source metadata lacks a valid byte size"),
    ],
)
def test_prepare_rejects_malformed_source_identity_before_staging(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    bundle = legacy_bundle(tmp_path)
    source_path = bundle / "source.json"
    source = _read_json(source_path)
    if value is None:
        source.pop(field, None)
    else:
        source[field] = value
    _write_json(source_path, source)
    _repair_manifest_after_artifact_change(bundle)
    _assert_preflight_refusal(bundle, message)


@pytest.mark.parametrize(
    ("source_fields", "message"),
    [
        (None, "source metadata source is malformed"),
        ({}, "source metadata kind is malformed"),
        ({"kind": "unknown", "ref": "identity"}, "source metadata kind is malformed"),
        ({"kind": "local"}, "source metadata reference is malformed"),
        ({"kind": "local", "ref": 9}, "source metadata reference is malformed"),
    ],
)
def test_prepare_rejects_malformed_source_document_source_before_staging(
    tmp_path: Path, source_fields: object, message: str
) -> None:
    bundle = legacy_bundle(tmp_path)
    source_path = bundle / "source.json"
    source = _read_json(source_path)
    source["source"] = source_fields
    _write_json(source_path, source)
    _repair_manifest_after_artifact_change(bundle)
    _assert_preflight_refusal(bundle, message)


def test_prepare_rejects_cross_document_source_disagreement_before_staging(
    tmp_path: Path,
) -> None:
    bundle = legacy_bundle(tmp_path)
    manifest_path = bundle / MANIFEST_NAME
    source_path = bundle / "source.json"
    manifest = _read_json(manifest_path)
    source = _read_json(source_path)
    manifest["source"].update({"kind": "url", "ref": "https://example.invalid/a"})
    source["source"].update({"kind": "url", "ref": "https://example.invalid/b"})
    _write_json(source_path, source)
    _rewrite_manifest_hashes(bundle, manifest)
    _write_json(manifest_path, manifest)
    _assert_preflight_refusal(bundle, "manifest and source metadata identities disagree")


def test_prepare_rejects_source_marker_before_staging(tmp_path: Path) -> None:
    bundle = legacy_bundle(tmp_path)
    (bundle / MARKER_NAME).write_text("{}\n", encoding="utf-8")
    _assert_preflight_refusal(bundle, "source role contains a migration marker")


def _assert_source_unchanged_and_staging_removed(
    bundle: Path, before: dict[Path, tuple[str, bytes | str | None]]
) -> None:
    assert _tree_state(bundle) == before
    assert not os.path.lexists(_staging_role(bundle))


def test_prepare_rewrites_only_approved_local_identity_fields(tmp_path: Path) -> None:
    bundle = legacy_bundle(tmp_path)
    before = _tree_state(bundle)
    source_before = _read_json(bundle / "source.json")
    prepared = prepare_bundle_migration(bundle)
    staging = prepared.staging_dir
    manifest = _read_json(staging / MANIFEST_NAME)
    source = _read_json(staging / "source.json")
    metadata = _read_json(staging / "transcript" / "metadata.json")
    expected_ref = f"sha256:{source_before['sha256']}"
    assert manifest["schema_version"] == "1.0.0"
    assert manifest["source"]["ref"] == source["source"]["ref"] == expected_ref
    assert manifest["source"]["bytes"] == source["source"]["bytes"] == source_before["bytes"]
    assert "path" not in source["transcript_sidecar"]
    assert "path" not in metadata["source_media"]
    assert "path" not in metadata["backend"]
    assert _tree_state(bundle) == before


def test_prepare_preserves_remote_source_url(tmp_path: Path) -> None:
    bundle = legacy_bundle(tmp_path)
    manifest_path = bundle / MANIFEST_NAME
    source_path = bundle / "source.json"
    manifest = _read_json(manifest_path)
    source = _read_json(source_path)
    url = "https://example.invalid/talk"
    manifest["source"].update({"kind": "url", "ref": url})
    source["source"].update({"kind": "url", "ref": url})
    _write_json(source_path, source)
    _rewrite_manifest_hashes(bundle, manifest)
    _write_json(manifest_path, manifest)
    before = _tree_state(bundle)
    staging = prepare_bundle_migration(bundle).staging_dir
    migrated_manifest = _read_json(staging / MANIFEST_NAME)
    migrated_source = _read_json(staging / "source.json")
    assert migrated_manifest["source"] == manifest["source"]
    assert migrated_source["source"] == source["source"]
    assert migrated_manifest["source"]["ref"] == url
    assert _tree_state(bundle) == before


def test_prepare_refreshes_every_declared_output_identity(tmp_path: Path) -> None:
    bundle = legacy_bundle(tmp_path)
    before = _tree_state(bundle)
    staging = prepare_bundle_migration(bundle).staging_dir
    manifest = _read_json(staging / MANIFEST_NAME)
    records = [output for stage in manifest["stages"].values() for output in stage["outputs"]]
    assert records
    for output in records:
        assert (output["sha256"], output["bytes"]) == _digest(staging / output["path"])
    assert _tree_state(bundle) == before


def test_prepare_follows_declared_transcript_metadata_path(tmp_path: Path) -> None:
    bundle = legacy_bundle(tmp_path)
    manifest_path = bundle / MANIFEST_NAME
    source_path = bundle / "source.json"
    default_metadata_path = bundle / "transcript" / "metadata.json"
    declared_relative = Path("transcript/alternate/metadata.json")
    declared_metadata_path = bundle / declared_relative
    declared_metadata_path.parent.mkdir()

    source = _read_json(source_path)
    source["transcript"]["metadata"] = declared_relative.as_posix()
    _write_json(source_path, source)

    metadata = _read_json(default_metadata_path)
    metadata["backend"]["command"] = "/Users/alice/bin/transcribe --private"
    metadata["backend"]["argv0"] = "/Users/alice/bin/transcribe"
    _write_json(declared_metadata_path, metadata)
    default_before = default_metadata_path.read_bytes()

    manifest = _read_json(manifest_path)
    metadata_output = next(
        output
        for stage in manifest["stages"].values()
        for output in stage["outputs"]
        if output["path"] == "transcript/metadata.json"
    )
    metadata_output["path"] = declared_relative.as_posix()
    _rewrite_manifest_hashes(bundle, manifest)
    _write_json(manifest_path, manifest)
    before = _tree_state(bundle)

    staging = prepare_bundle_migration(bundle).staging_dir

    migrated_source = _read_json(staging / "source.json")
    migrated_metadata = _read_json(staging / declared_relative)
    migrated_manifest = _read_json(staging / MANIFEST_NAME)
    assert migrated_source["transcript"]["metadata"] == declared_relative.as_posix()
    assert "path" not in migrated_metadata["source_media"]
    assert "path" not in migrated_metadata["backend"]
    assert "command" not in migrated_metadata["backend"]
    assert migrated_metadata["backend"]["argv0"] == "transcribe"
    assert (staging / "transcript" / "metadata.json").read_bytes() == default_before
    migrated_output = next(
        output
        for stage in migrated_manifest["stages"].values()
        for output in stage["outputs"]
        if output["path"] == declared_relative.as_posix()
    )
    assert (migrated_output["sha256"], migrated_output["bytes"]) == _digest(
        staging / declared_relative
    )
    assert _tree_state(bundle) == before


def test_prepare_preserves_every_non_target_entry_and_records_snapshots(
    tmp_path: Path,
) -> None:
    bundle = legacy_bundle(tmp_path)
    (bundle / "empty-extra").mkdir()
    (bundle / "analysis" / "user-note.txt").write_bytes(b"user bytes\x00\xff")
    before = _tree_state(bundle)
    prepared = prepare_bundle_migration(bundle)
    after = _tree_state(prepared.staging_dir)
    target_paths = {
        Path(MANIFEST_NAME),
        Path("source.json"),
        Path("transcript/metadata.json"),
        Path(MARKER_NAME),
    }
    assert {path: value for path, value in after.items() if path not in target_paths} == {
        path: value for path, value in before.items() if path not in target_paths
    }
    assert {entry.path.as_posix() for entry in prepared.source_snapshot} == {
        path.as_posix() for path in before
    }
    assert {entry.path.as_posix() for entry in prepared.target_snapshot} == {
        path.as_posix() for path in after
    }
    assert _tree_state(bundle) == before


def test_prepare_rejects_copy_drift_and_removes_owned_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = legacy_bundle(tmp_path)
    before = _tree_state(bundle)
    real_copytree = migrations.shutil.copytree

    def copy_then_drift(source: Any, destination: Any, *args: Any, **kwargs: Any) -> Any:
        copied = real_copytree(source, destination, *args, **kwargs)
        if Path(source) == bundle:
            (Path(copied) / "transcript" / "transcript.md").write_text(
                "copy drift\n", encoding="utf-8"
            )
        return copied

    monkeypatch.setattr(migrations.shutil, "copytree", copy_then_drift)
    with pytest.raises(MigrationError, match="copied staging tree differs") as error:
        prepare_bundle_migration(bundle)
    assert str(bundle) not in str(error.value)
    _assert_source_unchanged_and_staging_removed(bundle, before)


def test_prepare_rejects_transformation_drift_and_removes_owned_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = legacy_bundle(tmp_path)
    before = _tree_state(bundle)
    real_refresh = getattr(migrations, "_refresh_output_records", None)

    def drift_then_refresh(staging: Path, manifest: dict[str, Any]) -> None:
        (staging / "transcript" / "transcript.md").write_text(
            "transformation drift\n", encoding="utf-8"
        )
        assert real_refresh is not None
        real_refresh(staging, manifest)

    monkeypatch.setattr(migrations, "_refresh_output_records", drift_then_refresh, raising=False)
    with pytest.raises(MigrationError, match="changed a non-target source entry") as error:
        prepare_bundle_migration(bundle)
    assert str(bundle) not in str(error.value)
    _assert_source_unchanged_and_staging_removed(bundle, before)


def test_prepare_detects_source_mutation_and_removes_owned_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = legacy_bundle(tmp_path)
    before = _tree_state(bundle)
    real_copytree = migrations.shutil.copytree

    def copy_then_mutate_source(source: Any, destination: Any, *args: Any, **kwargs: Any) -> Any:
        copied = real_copytree(source, destination, *args, **kwargs)
        if Path(source) == bundle:
            (Path(source) / "transcript" / "transcript.md").write_text(
                "concurrent source change\n", encoding="utf-8"
            )
        return copied

    monkeypatch.setattr(migrations.shutil, "copytree", copy_then_mutate_source)
    with pytest.raises(MigrationError, match="source tree changed") as error:
        prepare_bundle_migration(bundle)
    assert str(bundle) not in str(error.value)
    expected = dict(before)
    expected[Path("transcript/transcript.md")] = (
        "file",
        b"concurrent source change\n",
    )
    assert _tree_state(bundle) == expected
    assert not os.path.lexists(_staging_role(bundle))


def test_marker_write_failure_removes_only_empty_reserved_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = legacy_bundle(tmp_path)
    before = _tree_state(bundle)
    real_write_json = getattr(migrations, "_write_json", None)

    def fail_marker(path: Path, value: object, role: str) -> None:
        if path.name == MARKER_NAME:
            raise MigrationError("cannot write migration marker")
        assert real_write_json is not None
        real_write_json(path, value, role)

    monkeypatch.setattr(migrations, "_write_json", fail_marker, raising=False)
    with pytest.raises(MigrationError, match="initialize the staging role") as error:
        prepare_bundle_migration(bundle)
    assert str(bundle) not in str(error.value)
    _assert_source_unchanged_and_staging_removed(bundle, before)


def test_partial_copy_failure_removes_exactly_marked_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = legacy_bundle(tmp_path)
    before = _tree_state(bundle)

    def fail_copy(source: Path, destination: Path, **kwargs: Any) -> Path:
        del source, kwargs
        (destination / "partial.txt").write_text("partial\n", encoding="utf-8")
        raise OSError("synthetic copy failure")

    monkeypatch.setattr(migrations.shutil, "copytree", fail_copy)
    with pytest.raises(MigrationError, match="copy the source role") as error:
        prepare_bundle_migration(bundle)
    assert str(bundle) not in str(error.value)
    _assert_source_unchanged_and_staging_removed(bundle, before)


def test_mismatched_marker_cleanup_refuses_and_preserves_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = legacy_bundle(tmp_path)
    before = _tree_state(bundle)

    def replace_marker_then_fail(source: Path, destination: Path, **kwargs: Any) -> Path:
        del source, kwargs
        _write_json(
            destination / MARKER_NAME,
            {
                "bundle_id": "different-owner",
                "source_version": "0.1.0",
                "target_version": "1.0.0",
            },
        )
        (destination / "owner.txt").write_text("retain\n", encoding="utf-8")
        raise OSError("synthetic copy failure")

    monkeypatch.setattr(migrations.shutil, "copytree", replace_marker_then_fail)
    with pytest.raises(
        MigrationError, match="cannot copy the source role and cannot remove owned staging"
    ) as error:
        prepare_bundle_migration(bundle)
    assert str(bundle) not in str(error.value)
    staging = _staging_role(bundle)
    assert _tree_state(bundle) == before
    assert _read_json(staging / MARKER_NAME)["bundle_id"] == "different-owner"
    assert (staging / "owner.txt").read_text(encoding="utf-8") == "retain\n"


def test_exact_payload_marker_symlink_never_authorizes_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = legacy_bundle(tmp_path)
    before = _tree_state(bundle)
    bundle_id = _read_json(bundle / MANIFEST_NAME)["bundle_id"]
    external_marker = tmp_path / "external-marker.json"
    _write_json(
        external_marker,
        {
            "bundle_id": bundle_id,
            "source_version": "0.1.0",
            "target_version": "1.0.0",
        },
    )

    def replace_marker_with_symlink_then_fail(
        source: Path, destination: Path, **kwargs: Any
    ) -> Path:
        del source, kwargs
        marker = destination / MARKER_NAME
        marker.unlink()
        marker.symlink_to(external_marker)
        (destination / "foreign.txt").write_text("retain\n", encoding="utf-8")
        raise OSError("synthetic copy failure")

    monkeypatch.setattr(
        migrations.shutil,
        "copytree",
        replace_marker_with_symlink_then_fail,
    )
    with pytest.raises(
        MigrationError, match="cannot copy the source role and cannot remove owned staging"
    ) as error:
        prepare_bundle_migration(bundle)
    assert str(bundle) not in str(error.value)
    staging = _staging_role(bundle)
    assert _tree_state(bundle) == before
    assert staging.is_dir()
    assert (staging / MARKER_NAME).is_symlink()
    assert (staging / "foreign.txt").read_text(encoding="utf-8") == "retain\n"
    assert external_marker.is_file()


def test_post_copy_marker_drift_is_not_blessed_or_cleaned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = legacy_bundle(tmp_path)
    before = _tree_state(bundle)
    real_derive = migrations._derive_target_documents  # pyright: ignore[reportPrivateUsage]

    def derive_then_drift_marker(
        source: Path,
        manifest: dict[str, Any],
        source_sha256: str,
        source_bytes: int,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        documents = real_derive(source, manifest, source_sha256, source_bytes)
        marker_path = _staging_role(bundle) / MARKER_NAME
        marker = _read_json(marker_path)
        marker["bundle_id"] = "different-owner"
        _write_json(marker_path, marker)
        return documents

    monkeypatch.setattr(migrations, "_derive_target_documents", derive_then_drift_marker)
    with pytest.raises(
        MigrationError, match="target preparation failed and owned staging cleanup also failed"
    ) as error:
        prepare_bundle_migration(bundle)
    assert str(bundle) not in str(error.value)
    staging = _staging_role(bundle)
    assert _tree_state(bundle) == before
    assert staging.is_dir()
    assert _read_json(staging / MARKER_NAME)["bundle_id"] == "different-owner"


def test_target_model_failure_removes_only_owned_staging(tmp_path: Path) -> None:
    bundle = legacy_bundle(tmp_path)
    manifest_path = bundle / MANIFEST_NAME
    manifest = _read_json(manifest_path)
    manifest["created"] = "not-a-datetime"
    _write_json(manifest_path, manifest)
    before = _tree_state(bundle)
    with pytest.raises(MigrationError, match="current artifact models") as error:
        prepare_bundle_migration(bundle)
    assert str(bundle) not in str(error.value)
    _assert_source_unchanged_and_staging_removed(bundle, before)


def test_target_cross_document_failure_is_detected_standalone(tmp_path: Path) -> None:
    bundle = legacy_bundle(tmp_path)
    before = _tree_state(bundle)
    prepared = prepare_bundle_migration(bundle)
    source_path = prepared.staging_dir / "source.json"
    source = _read_json(source_path)
    source["source"]["ref"] = f"sha256:{'f' * 64}"
    _write_json(source_path, source)
    manifest = _read_json(prepared.staging_dir / MANIFEST_NAME)
    _rewrite_manifest_hashes(prepared.staging_dir, manifest)
    _write_json(prepared.staging_dir / MANIFEST_NAME, manifest)
    with pytest.raises(MigrationError, match="manifest and source metadata disagree") as error:
        migrations._validate_target(  # pyright: ignore[reportPrivateUsage]
            prepared.staging_dir,
            expected_bundle_id=prepared.bundle_id,
            source_sha256=prepared.source_sha256,
            source_bytes=prepared.source_bytes,
        )
    assert str(bundle) not in str(error.value)
    assert _tree_state(bundle) == before
    assert prepared.staging_dir.is_dir()


def test_late_target_document_mutation_cannot_enter_the_final_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = legacy_bundle(tmp_path)
    before = _tree_state(bundle)
    real_assert_preserved = (
        migrations._assert_preserved_tree  # pyright: ignore[reportPrivateUsage]
    )
    calls = 0

    def mutate_after_second_preservation_check(
        target: Path,
        source_snapshot: migrations.TreeSnapshot,
        target_documents: set[Any],
    ) -> None:
        nonlocal calls
        real_assert_preserved(target, source_snapshot, target_documents)
        calls += 1
        if calls == 2:
            (target / "source.json").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(
        migrations,
        "_assert_preserved_tree",
        mutate_after_second_preservation_check,
    )
    with pytest.raises(
        MigrationError, match="target source metadata differs from approved transformation"
    ) as error:
        prepare_bundle_migration(bundle)
    assert str(bundle) not in str(error.value)
    assert calls >= 2
    _assert_source_unchanged_and_staging_removed(bundle, before)


def test_malformed_transcript_metadata_removes_owned_staging(tmp_path: Path) -> None:
    bundle = legacy_bundle(tmp_path)
    metadata_path = bundle / "transcript" / "metadata.json"
    metadata_path.write_text("{", encoding="utf-8")
    _repair_manifest_after_artifact_change(bundle)
    before = _tree_state(bundle)
    with pytest.raises(MigrationError, match="cannot read valid transcript metadata") as error:
        prepare_bundle_migration(bundle)
    assert str(bundle) not in str(error.value)
    _assert_source_unchanged_and_staging_removed(bundle, before)


def test_invalid_utf8_target_document_removes_owned_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = legacy_bundle(tmp_path)
    before = _tree_state(bundle)
    real_write_json = getattr(migrations, "_write_json", None)

    def write_then_corrupt(path: Path, value: object, role: str) -> None:
        assert real_write_json is not None
        real_write_json(path, value, role)
        if path.name == "source.json":
            path.write_bytes(b"\xff")

    monkeypatch.setattr(migrations, "_write_json", write_then_corrupt, raising=False)
    with pytest.raises(MigrationError, match="cannot read target source metadata") as error:
        prepare_bundle_migration(bundle)
    assert str(bundle) not in str(error.value)
    _assert_source_unchanged_and_staging_removed(bundle, before)


def test_cleanup_failure_retains_exactly_marked_private_staging_mode_and_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = legacy_bundle(tmp_path)
    bundle.chmod(0o700)
    before = _tree_state(bundle)

    def fail_copy(source: Path, destination: Path, **kwargs: Any) -> Path:
        del source, destination, kwargs
        raise OSError("synthetic copy failure")

    def fail_cleanup(path: Path) -> None:
        del path
        raise OSError("synthetic cleanup failure")

    monkeypatch.setattr(migrations.shutil, "copytree", fail_copy)
    monkeypatch.setattr(migrations.shutil, "rmtree", fail_cleanup)
    previous_umask = os.umask(0o022)
    try:
        with pytest.raises(MigrationError, match="cannot remove owned staging") as error:
            prepare_bundle_migration(bundle)
    finally:
        os.umask(previous_umask)
    assert str(bundle) not in str(error.value)
    staging = _staging_role(bundle)
    assert _tree_state(bundle) == before
    assert staging.is_dir()
    assert _read_json(staging / MARKER_NAME)["source_version"] == "0.1.0"
    source_mode = stat.S_IMODE(bundle.stat().st_mode)
    staging_mode = stat.S_IMODE(staging.stat().st_mode)
    assert staging_mode == 0o700
    assert staging_mode & 0o077 == 0
    assert staging_mode & ~source_mode == 0


def test_prepare_reserves_private_staging_mode_at_copy_entry_under_umask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = legacy_bundle(tmp_path)
    bundle.chmod(0o700)
    before = _tree_state(bundle)
    observed_modes: list[tuple[int, int]] = []
    real_copytree = migrations.shutil.copytree

    def record_modes(source: Any, destination: Any, *args: Any, **kwargs: Any) -> Any:
        if Path(source) == bundle:
            observed_modes.append(
                (
                    stat.S_IMODE(Path(source).stat().st_mode),
                    stat.S_IMODE(Path(destination).stat().st_mode),
                )
            )
        return real_copytree(source, destination, *args, **kwargs)

    monkeypatch.setattr(migrations.shutil, "copytree", record_modes)
    previous_umask = os.umask(0o022)
    try:
        prepared = prepare_bundle_migration(bundle)
    finally:
        os.umask(previous_umask)

    assert observed_modes == [(0o700, 0o700)]
    assert stat.S_IMODE(prepared.staging_dir.stat().st_mode) == 0o700
    assert _tree_state(bundle) == before


def test_prepare_rejects_deep_json_manifest_before_staging(tmp_path: Path) -> None:
    bundle = legacy_bundle(tmp_path)
    manifest_path = bundle / MANIFEST_NAME
    manifest = _read_json(manifest_path)
    manifest["deep_json_probe"] = _nested_json_object(129)
    _write_json(manifest_path, manifest)
    _assert_preflight_refusal(bundle, "cannot read valid source manifest")


def test_declared_artifact_status_error_is_path_free_before_staging(tmp_path: Path) -> None:
    bundle = legacy_bundle(tmp_path)
    before = _tree_state(bundle)
    restricted = bundle / "analysis"
    original_mode = stat.S_IMODE(restricted.stat().st_mode)
    restricted.chmod(0o000)
    try:
        with pytest.raises(
            MigrationError, match="cannot inspect a declared bundle artifact"
        ) as error:
            prepare_bundle_migration(bundle)
    finally:
        restricted.chmod(original_mode)
    assert str(bundle) not in str(error.value)
    assert _tree_state(bundle) == before
    assert not os.path.lexists(_staging_role(bundle))


@pytest.mark.parametrize(
    ("document", "message"),
    [
        pytest.param(
            "transcript-metadata",
            "cannot read valid transcript metadata",
            id="transcript-metadata",
        ),
        pytest.param(
            "transcript-segments",
            "cannot read valid target transcript segments",
            id="transcript-segments",
        ),
    ],
)
def test_prepare_rejects_deep_json_document_and_removes_owned_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    document: str,
    message: str,
) -> None:
    bundle = legacy_bundle(tmp_path)
    if document == "transcript-metadata":
        path = bundle / "transcript" / "metadata.json"
        value = _read_json(path)
        value["deep_json_probe"] = _nested_json_object(129)
        _write_json(path, value)
    else:
        path = bundle / "transcript" / "segments.json"
        segments = _read_json_array(path)
        segments[0]["deep_json_probe"] = _nested_json_object(129)
        _write_json(path, segments)
    _repair_manifest_after_artifact_change(bundle)
    before = _tree_state(bundle)
    copy_entries = 0
    real_copytree = migrations.shutil.copytree

    def record_copy(source: Any, destination: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal copy_entries
        if Path(source) == bundle:
            copy_entries += 1
        return real_copytree(source, destination, *args, **kwargs)

    monkeypatch.setattr(migrations.shutil, "copytree", record_copy)
    with pytest.raises(MigrationError, match=message) as error:
        prepare_bundle_migration(bundle)
    assert str(bundle) not in str(error.value)
    assert copy_entries == 1
    _assert_source_unchanged_and_staging_removed(bundle, before)


@pytest.mark.parametrize(
    ("document", "message", "post_marker"),
    [
        pytest.param("manifest", "cannot read valid source manifest", False, id="manifest"),
        pytest.param(
            "source-metadata",
            "cannot read valid source metadata",
            False,
            id="source-metadata",
        ),
        pytest.param(
            "transcript-metadata",
            "cannot read valid transcript metadata",
            True,
            id="transcript-metadata",
        ),
        pytest.param(
            "transcript-segments",
            "cannot read valid target transcript segments",
            True,
            id="transcript-segments",
        ),
    ],
)
@pytest.mark.parametrize(
    "constant",
    [
        pytest.param(float("nan"), id="NaN"),
        pytest.param(float("inf"), id="Infinity"),
        pytest.param(float("-inf"), id="minus-Infinity"),
    ],
)
def test_prepare_rejects_non_finite_json_across_real_documents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    document: str,
    message: str,
    post_marker: bool,
    constant: float,
) -> None:
    bundle = legacy_bundle(tmp_path)
    if document == "manifest":
        path = bundle / MANIFEST_NAME
        value = _read_json(path)
        value["non_finite_probe"] = constant
        _write_json(path, value)
    elif document == "source-metadata":
        path = bundle / "source.json"
        value = _read_json(path)
        value["non_finite_probe"] = constant
        _write_json(path, value)
        _repair_manifest_after_artifact_change(bundle)
    elif document == "transcript-metadata":
        path = bundle / "transcript" / "metadata.json"
        value = _read_json(path)
        value["non_finite_probe"] = constant
        _write_json(path, value)
        _repair_manifest_after_artifact_change(bundle)
    else:
        path = bundle / "transcript" / "segments.json"
        segments = _read_json_array(path)
        segments[0]["non_finite_probe"] = constant
        _write_json(path, segments)
        _repair_manifest_after_artifact_change(bundle)
    before = _tree_state(bundle)
    copy_entries = 0
    real_copytree = migrations.shutil.copytree

    def record_copy(source: Any, destination: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal copy_entries
        if Path(source) == bundle:
            copy_entries += 1
        return real_copytree(source, destination, *args, **kwargs)

    monkeypatch.setattr(migrations.shutil, "copytree", record_copy)
    with pytest.raises(MigrationError, match=message) as error:
        prepare_bundle_migration(bundle)
    assert str(bundle) not in str(error.value)
    assert bool(copy_entries) is post_marker
    _assert_source_unchanged_and_staging_removed(bundle, before)


def test_prepare_rejects_non_finite_json_created_during_transformation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = legacy_bundle(tmp_path)
    before = _tree_state(bundle)
    real_derive = migrations._derive_target_documents  # pyright: ignore[reportPrivateUsage]

    def inject_non_finite(
        source: Path,
        manifest: dict[str, Any],
        source_sha256: str,
        source_bytes: int,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        documents = real_derive(source, manifest, source_sha256, source_bytes)
        documents[1]["non_finite_probe"] = float("nan")
        return documents

    monkeypatch.setattr(migrations, "_derive_target_documents", inject_non_finite)
    with pytest.raises(MigrationError, match="serialize migration JSON") as error:
        prepare_bundle_migration(bundle)
    assert str(bundle) not in str(error.value)
    _assert_source_unchanged_and_staging_removed(bundle, before)


def test_prepare_rejects_matching_strict_raw_duration_strings(
    tmp_path: Path,
) -> None:
    bundle = legacy_bundle(tmp_path)
    manifest_path = bundle / MANIFEST_NAME
    source_path = bundle / "source.json"
    manifest = _read_json(manifest_path)
    source = _read_json(source_path)
    manifest["source"]["duration_s"] = "1.25"
    source["source"]["duration_s"] = "1.25"
    _write_json(source_path, source)
    _rewrite_manifest_hashes(bundle, manifest)
    _write_json(manifest_path, manifest)
    before = _tree_state(bundle)
    with pytest.raises(MigrationError, match="current artifact models") as error:
        prepare_bundle_migration(bundle)
    assert str(bundle) not in str(error.value)
    _assert_source_unchanged_and_staging_removed(bundle, before)


def test_prepare_rejects_strict_raw_transcript_metadata_numeric_string(
    tmp_path: Path,
) -> None:
    bundle = legacy_bundle(tmp_path)
    metadata_path = bundle / "transcript" / "metadata.json"
    metadata = _read_json(metadata_path)
    metadata["normalized_audio"]["bytes"] = str(metadata["normalized_audio"]["bytes"])
    _write_json(metadata_path, metadata)
    _repair_manifest_after_artifact_change(bundle)
    before = _tree_state(bundle)
    with pytest.raises(MigrationError, match="current artifact models") as error:
        prepare_bundle_migration(bundle)
    assert str(bundle) not in str(error.value)
    _assert_source_unchanged_and_staging_removed(bundle, before)


def test_prepare_rejects_strict_raw_transcript_segment_numeric_string(
    tmp_path: Path,
) -> None:
    bundle = legacy_bundle(tmp_path)
    segments_path = bundle / "transcript" / "segments.json"
    segments = _read_json_array(segments_path)
    segments[0]["start_s"] = str(segments[0]["start_s"])
    _write_json(segments_path, segments)
    _repair_manifest_after_artifact_change(bundle)
    before = _tree_state(bundle)
    with pytest.raises(MigrationError, match="current artifact models") as error:
        prepare_bundle_migration(bundle)
    assert str(bundle) not in str(error.value)
    _assert_source_unchanged_and_staging_removed(bundle, before)


def test_migrate_swaps_in_target_and_retains_original_backup(tmp_path: Path) -> None:
    bundle = legacy_bundle(tmp_path)
    before = _tree_bytes(bundle)
    result = migrations.migrate_bundle(bundle)
    assert result.outcome == "migrated"
    assert result.backup_retained is True
    assert Manifest.load(bundle).schema_version == "1.0.0"
    backup = bundle.with_name(bundle.name + BACKUP_SUFFIX)
    assert _tree_bytes(backup) == before


def test_migration_redacts_stage_error_paths_in_written_target(tmp_path: Path) -> None:
    bundle = legacy_bundle(tmp_path)
    manifest_path = bundle / MANIFEST_NAME
    manifest = _read_json(manifest_path)
    private_path = "/Users/alice/Private Recordings/secret.wav"
    manifest["stages"]["transcribe"]["error"] = f"decoder failed at {private_path}"
    _write_json(manifest_path, manifest)
    before = _tree_bytes(bundle)

    result = migrations.migrate_bundle(bundle)
    assert result.outcome == "migrated"
    assert private_path not in manifest_path.read_text(encoding="utf-8")
    assert "<path>" in manifest_path.read_text(encoding="utf-8")
    assert _tree_bytes(bundle.with_name(bundle.name + BACKUP_SUFFIX)) == before


def test_migration_refuses_undeclared_legacy_extensions_without_data_loss(tmp_path: Path) -> None:
    bundle = legacy_bundle(tmp_path)
    manifest_path = bundle / MANIFEST_NAME
    manifest = _read_json(manifest_path)
    manifest["stages"]["transcribe"]["legacy_extension"] = {"operator_note": "retain me"}
    _write_json(manifest_path, manifest)
    before = _tree_bytes(bundle)

    with pytest.raises(MigrationError, match="current artifact models") as error:
        migrations.migrate_bundle(bundle)
    assert str(bundle) not in str(error.value)
    assert _tree_bytes(bundle) == before


def test_second_rename_failure_restores_legacy_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = legacy_bundle(tmp_path)
    real_rename = migrations._rename  # pyright: ignore[reportPrivateUsage]
    calls = 0

    def fail_second(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic publish failure")
        real_rename(source, destination)

    monkeypatch.setattr(migrations, "_rename", fail_second)
    with pytest.raises(MigrationError, match="publish migrated bundle"):
        migrations.migrate_bundle(bundle)
    assert _read_json(bundle / MANIFEST_NAME)["schema_version"] == "0.1.0"
    assert not os.path.lexists(bundle.with_name(bundle.name + BACKUP_SUFFIX))
    assert bundle.with_name(bundle.name + STAGING_SUFFIX).is_dir()


def test_restart_reuses_marked_target_after_in_process_rollback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = legacy_bundle(tmp_path)
    real_rename = migrations._rename  # pyright: ignore[reportPrivateUsage]
    calls = 0

    def fail_second_once(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic publish failure")
        real_rename(source, destination)

    monkeypatch.setattr(migrations, "_rename", fail_second_once)
    with pytest.raises(MigrationError, match="original was restored"):
        migrations.migrate_bundle(bundle)
    monkeypatch.setattr(migrations, "_rename", real_rename)
    result = migrations.migrate_bundle(bundle)
    assert result.outcome == "recovered"
    assert Manifest.load(bundle).schema_version == "1.0.0"


def test_restart_finishes_forward_from_backup_and_marked_staging(tmp_path: Path) -> None:
    bundle = legacy_bundle(tmp_path)
    prepared = prepare_bundle_migration(bundle)
    prepared.source_dir.rename(prepared.backup_dir)
    result = migrations.migrate_bundle(bundle)
    assert result.outcome == "recovered"
    assert result.backup_retained is True
    assert Manifest.load(bundle).schema_version == "1.0.0"


def test_restart_restores_backup_when_no_usable_staging_exists(tmp_path: Path) -> None:
    bundle = legacy_bundle(tmp_path)
    prepared = prepare_bundle_migration(bundle)
    shutil.rmtree(prepared.staging_dir)
    prepared.source_dir.rename(prepared.backup_dir)
    with pytest.raises(MigrationError, match="restored the original"):
        migrations.migrate_bundle(bundle)
    assert _read_json(bundle / MANIFEST_NAME)["schema_version"] == "0.1.0"
    assert not os.path.lexists(prepared.backup_dir)


@pytest.mark.parametrize("role", ["backup", "staging"])
def test_migrate_preserves_occupied_sibling_role(tmp_path: Path, role: str) -> None:
    bundle = legacy_bundle(tmp_path)
    sibling = bundle.with_name(
        bundle.name + (BACKUP_SUFFIX if role == "backup" else STAGING_SUFFIX)
    )
    sibling.mkdir()
    (sibling / "owner.txt").write_text("unrelated\n", encoding="utf-8")
    before = _tree_state(bundle)
    with pytest.raises(MigrationError, match=f"{role} role"):
        migrations.migrate_bundle(bundle)
    assert _tree_state(bundle) == before
    assert (sibling / "owner.txt").read_text(encoding="utf-8") == "unrelated\n"


def test_already_current_is_a_validated_no_op(tmp_path: Path) -> None:
    media = tmp_path / "current" / "synthetic_talk.wav"
    media.parent.mkdir(parents=True)
    media.write_bytes(SYNTHETIC_TALK.read_bytes())
    media.with_suffix(".transcript.txt").write_text(
        SYNTHETIC_TRANSCRIPT.read_text(encoding="utf-8"), encoding="utf-8"
    )
    bundle = ingest_local(media, tmp_path / "current-bundles").bundle_dir
    result = migrations.migrate_bundle(bundle)
    assert result.outcome == "already_current"
    assert result.source_version == result.target_version == "1.1.0"
    assert result.backup_retained is False


def test_stored_one_zero_bundle_stays_ready_but_later_minor_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    media = tmp_path / "synthetic_talk.wav"
    media.write_bytes(SYNTHETIC_TALK.read_bytes())
    media.with_suffix(".transcript.txt").write_text(SYNTHETIC_TRANSCRIPT.read_text())
    state_path = tmp_path / "state.sqlite"
    with open_state(state_path) as state:
        result = state.ingest_one_shot(media, tmp_path / "bundles")
    assert result.manifest.schema_version == "1.1.0"
    bundle = result.bundle_dir
    manifest = _read_json(bundle / MANIFEST_NAME)
    manifest["schema_version"] = "1.0.0"
    manifest["source"].pop("external")
    for stage in manifest["stages"].values():
        stage.pop("warnings")
    source = _read_json(bundle / "source.json")
    source["source"].pop("external")
    _write_json(bundle / "source.json", source)
    metadata = _read_json(bundle / "transcript/metadata.json")
    metadata.pop("language")
    for field in (
        "model",
        "library_version",
        "compute_type",
        "detected_language",
        "language_probability",
    ):
        metadata["backend"].pop(field, None)
    _write_json(bundle / "transcript/metadata.json", metadata)
    _rewrite_manifest_hashes(bundle, manifest)
    _write_json(bundle / MANIFEST_NAME, manifest)
    before = _tree_bytes(bundle)

    assert cli.main(["library", "list", "--state", str(state_path), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["bundles"][0]["status"] == "ready"
    with open_state(state_path) as state:
        assert state.get_library_manifest(result.manifest.bundle_id).schema_version == "1.0.0"
        assert state.get_registered_evidence(result.manifest.bundle_id).segments.root
    assert cli.main(["migrate", str(bundle)]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["outcome"] == "already_current"
    assert "already readable; 1.0.0 is a subset of 1.1.0" in captured.err
    assert _tree_bytes(bundle) == before

    manifest["schema_version"] = "1.2.0"
    _write_json(bundle / MANIFEST_NAME, manifest)
    with pytest.raises(ValueError, match="unsupported bundle schema version '1.2.0'"):
        Manifest.load(bundle)
    assert cli.main(["library", "list", "--state", str(state_path), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["bundles"][0]["status"] != "ready"


def test_already_current_refuses_raw_path_bearing_stage_error(tmp_path: Path) -> None:
    media = tmp_path / "current-path-error" / "synthetic_talk.wav"
    media.parent.mkdir(parents=True)
    media.write_bytes(SYNTHETIC_TALK.read_bytes())
    media.with_suffix(".transcript.txt").write_text(
        SYNTHETIC_TRANSCRIPT.read_text(encoding="utf-8"), encoding="utf-8"
    )
    bundle = ingest_local(media, tmp_path / "current-path-error-bundles").bundle_dir
    manifest_path = bundle / MANIFEST_NAME
    payload = _read_json(manifest_path)
    private_path = "/Users/alice/Private Recordings/secret.wav"
    payload["stages"]["transcribe"]["error"] = f"failed at {private_path}"
    _write_json(manifest_path, payload)
    before = _tree_bytes(bundle)

    with pytest.raises(MigrationError, match="path-bearing stage error") as error:
        migrations.migrate_bundle(bundle)
    assert str(bundle) not in str(error.value)
    assert _tree_bytes(bundle) == before


def test_already_current_rejects_hash_invalid_target(tmp_path: Path) -> None:
    media = tmp_path / "invalid-current" / "synthetic_talk.wav"
    media.parent.mkdir(parents=True)
    media.write_bytes(SYNTHETIC_TALK.read_bytes())
    media.with_suffix(".transcript.txt").write_text(
        SYNTHETIC_TRANSCRIPT.read_text(encoding="utf-8"), encoding="utf-8"
    )
    bundle = ingest_local(media, tmp_path / "invalid-current-bundles").bundle_dir
    (bundle / "source.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(MigrationError, match="declared artifact integrity"):
        migrations.migrate_bundle(bundle)


def test_rerun_after_migration_validates_the_retained_backup(tmp_path: Path) -> None:
    bundle = legacy_bundle(tmp_path)
    migrations.migrate_bundle(bundle)
    result = migrations.migrate_bundle(bundle)
    assert result.outcome == "already_current"
    assert result.backup_retained is True


@pytest.mark.parametrize("role", ["source", "target"])
def test_migrate_rechecks_prepared_snapshots_before_first_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    role: str,
) -> None:
    bundle = legacy_bundle(tmp_path)
    real_prepare = migrations.prepare_bundle_migration

    def prepare_then_mutate(path: Path) -> migrations.PreparedBundleMigration:
        prepared = real_prepare(path)
        changed = prepared.source_dir if role == "source" else prepared.staging_dir
        (changed / "transcript" / "transcript.md").write_text(
            f"late {role} mutation\n", encoding="utf-8"
        )
        return prepared

    monkeypatch.setattr(migrations, "prepare_bundle_migration", prepare_then_mutate)
    with pytest.raises(MigrationError, match=f"{role} tree changed"):
        migrations.migrate_bundle(bundle)
    assert not os.path.lexists(bundle.with_name(bundle.name + BACKUP_SUFFIX))
    assert _read_json(bundle / MANIFEST_NAME)["schema_version"] == "0.1.0"


def test_backup_role_appearing_before_first_rename_is_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = legacy_bundle(tmp_path)
    backup = bundle.with_name(bundle.name + BACKUP_SUFFIX)
    real_assert = migrations._assert_snapshot  # pyright: ignore[reportPrivateUsage]

    def assert_then_occupy(path: Path, expected: Any, role: str) -> None:
        real_assert(path, expected, role)
        if role == "target" and not backup.exists():
            backup.mkdir(mode=0o711)
            (backup / "owner.txt").write_text("unrelated\n", encoding="utf-8")

    monkeypatch.setattr(migrations, "_assert_snapshot", assert_then_occupy)
    with pytest.raises(MigrationError, match="backup role became occupied") as error:
        migrations.migrate_bundle(bundle)
    assert str(bundle) not in str(error.value)
    assert (backup / "owner.txt").read_text(encoding="utf-8") == "unrelated\n"
    assert _read_json(bundle / MANIFEST_NAME)["schema_version"] == "0.1.0"


def test_source_role_appearing_before_forward_recovery_is_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = legacy_bundle(tmp_path)
    prepared = prepare_bundle_migration(bundle)
    prepared.source_dir.rename(prepared.backup_dir)
    real_assert = migrations._assert_snapshot  # pyright: ignore[reportPrivateUsage]

    def assert_then_occupy(path: Path, expected: Any, role: str) -> None:
        real_assert(path, expected, role)
        if role == "target" and not prepared.source_dir.exists():
            prepared.source_dir.mkdir(mode=0o711)
            (prepared.source_dir / "owner.txt").write_text("unrelated\n", encoding="utf-8")

    monkeypatch.setattr(migrations, "_assert_snapshot", assert_then_occupy)
    with pytest.raises(MigrationError, match="finish publication") as error:
        migrations.migrate_bundle(bundle)
    assert str(bundle) not in str(error.value)
    assert (prepared.source_dir / "owner.txt").read_text(encoding="utf-8") == "unrelated\n"
    assert prepared.backup_dir.is_dir()
    assert prepared.staging_dir.is_dir()


def test_restart_rejects_self_consistent_staging_drift_and_restores_backup(
    tmp_path: Path,
) -> None:
    bundle = legacy_bundle(tmp_path)
    prepared = prepare_bundle_migration(bundle)
    prepared.source_dir.rename(prepared.backup_dir)
    (prepared.staging_dir / "transcript" / "transcript.md").write_text(
        "self-consistent drift\n", encoding="utf-8"
    )
    _repair_manifest_after_artifact_change(prepared.staging_dir)

    with pytest.raises(MigrationError, match="staging role remains unusable"):
        migrations.migrate_bundle(bundle)

    assert _read_json(bundle / MANIFEST_NAME)["schema_version"] == "0.1.0"
    assert not os.path.lexists(prepared.backup_dir)
    assert prepared.staging_dir.is_dir()


def test_rerun_after_migration_rejects_invalid_retained_backup(tmp_path: Path) -> None:
    bundle = legacy_bundle(tmp_path)
    migrations.migrate_bundle(bundle)
    backup = bundle.with_name(bundle.name + BACKUP_SUFFIX)
    (backup / "source.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(MigrationError, match="declared artifact integrity"):
        migrations.migrate_bundle(bundle)

    assert Manifest.load(bundle).schema_version == "1.0.0"
    assert backup.is_dir()


def test_rerun_rejects_valid_backup_with_different_source_identity(tmp_path: Path) -> None:
    bundle = legacy_bundle(tmp_path)
    migrations.migrate_bundle(bundle)
    backup = bundle.with_name(bundle.name + BACKUP_SUFFIX)
    source_path = backup / "source.json"
    source = _read_json(source_path)
    source["sha256"] = "f" * 64
    _write_json(source_path, source)
    _repair_manifest_after_artifact_change(backup)

    with pytest.raises(MigrationError, match="backup source identity"):
        migrations.migrate_bundle(bundle)

    assert Manifest.load(bundle).schema_version == "1.0.0"
    assert backup.is_dir()


def _fail_lstat_once(
    monkeypatch: pytest.MonkeyPatch,
    target: Path,
) -> None:
    real_lstat = Path.lstat
    failed = False

    def fail_target_once(path: Path) -> os.stat_result:
        nonlocal failed
        caller = sys._getframe(1).f_code.co_name  # pyright: ignore[reportPrivateUsage]
        if not failed and path == target and caller == "_role_exists":
            failed = True
            raise PermissionError("synthetic role status failure")
        return real_lstat(path)

    monkeypatch.setattr(Path, "lstat", fail_target_once)


def test_source_role_status_error_does_not_overwrite_occupied_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = legacy_bundle(tmp_path)
    prepared = prepare_bundle_migration(bundle)
    shutil.rmtree(prepared.staging_dir)
    prepared.source_dir.rename(prepared.backup_dir)
    prepared.source_dir.mkdir()
    (prepared.source_dir / "owner.txt").write_text("unrelated\n", encoding="utf-8")
    _fail_lstat_once(monkeypatch, prepared.source_dir)

    with pytest.raises(MigrationError, match="cannot inspect the source role") as error:
        migrations.migrate_bundle(bundle)

    assert str(bundle) not in str(error.value)
    assert (prepared.source_dir / "owner.txt").read_text(encoding="utf-8") == "unrelated\n"
    assert prepared.backup_dir.is_dir()


@pytest.mark.parametrize("role", ["backup", "staging"])
def test_legacy_sibling_role_status_error_fails_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    role: str,
) -> None:
    bundle = legacy_bundle(tmp_path)
    before = _tree_state(bundle)
    sibling = bundle.with_name(
        bundle.name + (BACKUP_SUFFIX if role == "backup" else STAGING_SUFFIX)
    )
    _fail_lstat_once(monkeypatch, sibling)

    with pytest.raises(MigrationError, match=f"cannot inspect the {role} role") as error:
        migrations.migrate_bundle(bundle)

    assert str(bundle) not in str(error.value)
    assert _tree_state(bundle) == before
    assert not os.path.lexists(sibling)


def test_current_marker_status_error_does_not_settle_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = legacy_bundle(tmp_path)
    migrations.migrate_bundle(bundle)
    marker = bundle / MARKER_NAME
    _write_json(
        marker,
        migrations._marker_payload(  # pyright: ignore[reportPrivateUsage]
            Manifest.load(bundle).bundle_id
        ),
    )
    _fail_lstat_once(monkeypatch, marker)

    with pytest.raises(MigrationError, match="cannot inspect the source migration marker") as error:
        migrations.migrate_bundle(bundle)

    assert str(bundle) not in str(error.value)
    assert marker.is_file()
    assert bundle.with_name(bundle.name + BACKUP_SUFFIX).is_dir()


def test_current_staging_status_error_preserves_marker_and_collision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = legacy_bundle(tmp_path)
    migrations.migrate_bundle(bundle)
    marker = bundle / MARKER_NAME
    _write_json(
        marker,
        migrations._marker_payload(  # pyright: ignore[reportPrivateUsage]
            Manifest.load(bundle).bundle_id
        ),
    )
    staging = bundle.with_name(bundle.name + STAGING_SUFFIX)
    staging.mkdir()
    (staging / "owner.txt").write_text("unrelated\n", encoding="utf-8")
    _fail_lstat_once(monkeypatch, staging)

    with pytest.raises(MigrationError, match="cannot inspect the staging role") as error:
        migrations.migrate_bundle(bundle)

    assert str(bundle) not in str(error.value)
    assert marker.is_file()
    assert (staging / "owner.txt").read_text(encoding="utf-8") == "unrelated\n"


def test_first_rename_failure_retains_retryable_legacy_and_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = legacy_bundle(tmp_path)
    real_rename = migrations._rename  # pyright: ignore[reportPrivateUsage]

    def fail_first(source: Path, destination: Path) -> None:
        del source, destination
        raise OSError("synthetic first rename failure")

    monkeypatch.setattr(migrations, "_rename", fail_first)
    with pytest.raises(MigrationError, match="move the source role") as error:
        migrations.migrate_bundle(bundle)
    assert str(bundle) not in str(error.value)
    assert _read_json(bundle / MANIFEST_NAME)["schema_version"] == "0.1.0"
    assert not os.path.lexists(bundle.with_name(bundle.name + BACKUP_SUFFIX))
    assert _staging_role(bundle).is_dir()

    monkeypatch.setattr(migrations, "_rename", real_rename)
    assert migrations.migrate_bundle(bundle).outcome == "recovered"


def test_rollback_failure_retains_forward_recoverable_roles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = legacy_bundle(tmp_path)
    real_rename = migrations._rename  # pyright: ignore[reportPrivateUsage]
    calls = 0

    def fail_publish_and_restore(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls in {2, 3}:
            raise OSError("synthetic rename failure")
        real_rename(source, destination)

    monkeypatch.setattr(migrations, "_rename", fail_publish_and_restore)
    with pytest.raises(MigrationError, match="restoration also failed") as error:
        migrations.migrate_bundle(bundle)
    assert str(bundle) not in str(error.value)
    assert not os.path.lexists(bundle)
    assert bundle.with_name(bundle.name + BACKUP_SUFFIX).is_dir()
    assert _staging_role(bundle).is_dir()

    monkeypatch.setattr(migrations, "_rename", real_rename)
    assert migrations.migrate_bundle(bundle).outcome == "recovered"


def test_post_publication_validation_failure_is_restart_recoverable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = legacy_bundle(tmp_path)
    real_validate = migrations._validate_prepared_target  # pyright: ignore[reportPrivateUsage]

    def fail_only_after_publication(
        target: Path,
        **kwargs: Any,
    ) -> tuple[Manifest, migrations.TreeSnapshot]:
        result = real_validate(target, **kwargs)
        if target == bundle:
            raise MigrationError("synthetic post-publication validation failure")
        return result

    monkeypatch.setattr(
        migrations,
        "_validate_prepared_target",
        fail_only_after_publication,
    )
    with pytest.raises(MigrationError, match="post-publication validation") as error:
        migrations.migrate_bundle(bundle)
    assert str(bundle) not in str(error.value)
    assert Manifest.load(bundle).schema_version == "1.0.0"
    assert (bundle / MARKER_NAME).is_file()
    assert bundle.with_name(bundle.name + BACKUP_SUFFIX).is_dir()

    monkeypatch.setattr(migrations, "_validate_prepared_target", real_validate)
    assert migrations.migrate_bundle(bundle).outcome == "recovered"
    assert not os.path.lexists(bundle / MARKER_NAME)


def test_marker_clear_failure_is_restart_recoverable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = legacy_bundle(tmp_path)
    real_clear = migrations._clear_marker  # pyright: ignore[reportPrivateUsage]

    def fail_clear(target: Path) -> None:
        del target
        raise MigrationError("synthetic marker clear failure")

    monkeypatch.setattr(migrations, "_clear_marker", fail_clear)
    with pytest.raises(MigrationError, match="marker clear failure") as error:
        migrations.migrate_bundle(bundle)
    assert str(bundle) not in str(error.value)
    assert Manifest.load(bundle).schema_version == "1.0.0"
    assert (bundle / MARKER_NAME).is_file()
    assert bundle.with_name(bundle.name + BACKUP_SUFFIX).is_dir()

    monkeypatch.setattr(migrations, "_clear_marker", real_clear)
    assert migrations.migrate_bundle(bundle).outcome == "recovered"
    assert not os.path.lexists(bundle / MARKER_NAME)


def test_cli_migrates_in_place_with_path_free_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bundle = legacy_bundle(tmp_path)
    assert cli.main(["migrate", str(bundle)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "backup_retained": True,
        "bundle_id": Manifest.load(bundle).bundle_id,
        "outcome": "migrated",
        "source_version": "0.1.0",
        "target_version": "1.0.0",
    }
    assert str(bundle) not in json.dumps(payload)


def test_library_search_and_cite_survive_same_path_migration(tmp_path: Path) -> None:
    state_path, bundle, bundle_id, segment_id, before_anchor = registered_legacy_bundle(tmp_path)
    migrations.migrate_bundle(bundle)
    with open_state(state_path) as state:
        assert state.get_library_bundle(bundle_id).bundle_path == str(bundle)
        after = state.search_segments("inspectable knowledge")
        assert any(hit.bundle_id == bundle_id and hit.segment_id == segment_id for hit in after)
        after_anchor, resolved = state.cite_segment(bundle_id, segment_id)
        assert after_anchor == before_anchor
        assert resolved.outcome is AnchorResolution.EXACT


@pytest.mark.parametrize("text", ["", " \t\r\n", "\u001c\u0085\u2003\u3000"])
@pytest.mark.parametrize("operation", ["prepare", "migrate"])
def test_blank_legacy_segments_are_refused_without_changing_roles(
    tmp_path: Path, text: str, operation: str
) -> None:
    bundle = legacy_bundle(tmp_path)
    segments_path = bundle / "transcript" / "segments.json"
    segments = _read_json_array(segments_path)
    segments[0]["text"] = text
    _write_json(segments_path, segments)
    _repair_manifest_after_artifact_change(bundle)
    manifest = _read_json(bundle / MANIFEST_NAME)
    for stage in manifest["stages"].values():
        for output in stage["outputs"]:
            assert _digest(bundle / output["path"]) == (output["sha256"], output["bytes"])
    before = _tree_state(bundle.parent)
    with pytest.raises(MigrationError, match="artifact models"):
        if operation == "prepare":
            prepare_bundle_migration(bundle)
        else:
            migrations.migrate_bundle(bundle)
    assert _tree_state(bundle.parent) == before


def test_valid_legacy_segment_text_and_bytes_survive_migration(tmp_path: Path) -> None:
    bundle = legacy_bundle(tmp_path)
    segments_path = bundle / "transcript" / "segments.json"
    segments = _read_json_array(segments_path)
    segments[0]["text"] = " \tעברית\nالعربية 日本語 🙂\u3000"
    segments_path.write_text(json.dumps(segments, ensure_ascii=False, indent=3) + "\n")
    _repair_manifest_after_artifact_change(bundle)
    before = segments_path.read_bytes()
    ids = [segment["id"] for segment in segments]
    migrations.migrate_bundle(bundle)
    assert segments_path.read_bytes() == before
    backup = bundle.with_name(bundle.name + BACKUP_SUFFIX)
    assert (backup / "transcript" / "segments.json").read_bytes() == before
    assert [segment["id"] for segment in _read_json_array(segments_path)] == ids


def test_blank_legacy_retry_preserves_source_while_discarding_owned_staging(
    tmp_path: Path,
) -> None:
    bundle = legacy_bundle(tmp_path)
    prepared = prepare_bundle_migration(bundle)
    assert prepared.staging_dir.is_dir()
    segments_path = bundle / "transcript" / "segments.json"
    segments = _read_json_array(segments_path)
    segments[0]["text"] = " \u2003\n"
    _write_json(segments_path, segments)
    _repair_manifest_after_artifact_change(bundle)
    unrelated = bundle.parent / "unrelated-evidence"
    unrelated.mkdir()
    (unrelated / "note.txt").write_text("synthetic retained evidence")
    before_source = _tree_state(bundle)
    before_unrelated = _tree_state(unrelated)

    with pytest.raises(MigrationError, match="artifact models"):
        migrations.migrate_bundle(bundle)

    assert _tree_state(bundle) == before_source
    assert _tree_state(unrelated) == before_unrelated
    assert not prepared.staging_dir.exists()
    assert not bundle.with_name(bundle.name + BACKUP_SUFFIX).exists()


@pytest.mark.parametrize("artifact", ["source.json", "transcript/segments.json"])
@pytest.mark.parametrize("operation", ["prepare", "migrate", "current"])
def test_migration_requires_selected_evidence_declarations_without_rewriting_trees(
    tmp_path: Path, artifact: str, operation: str
) -> None:
    bundle = legacy_bundle(tmp_path)
    if operation == "current":
        migrations.migrate_bundle(bundle)
    manifest = _read_json(bundle / MANIFEST_NAME)
    for stage in manifest["stages"].values():
        stage["outputs"] = [item for item in stage["outputs"] if item["path"] != artifact]
    _write_json(bundle / MANIFEST_NAME, manifest)
    unrelated = bundle.parent / "unrelated-evidence"
    unrelated.mkdir()
    (unrelated / "note.txt").write_text("synthetic retained evidence")
    before = _tree_state(bundle.parent)
    with pytest.raises(MigrationError):
        if operation == "prepare":
            prepare_bundle_migration(bundle)
        else:
            migrations.migrate_bundle(bundle)
    assert _tree_state(bundle.parent) == before


@pytest.mark.parametrize("artifact", ["source.json", "transcript/segments.json"])
@pytest.mark.parametrize("current", [False, True])
def test_migration_refuses_contradictory_evidence_declarations(
    tmp_path: Path, artifact: str, current: bool
) -> None:
    bundle = legacy_bundle(tmp_path)
    if current:
        migrations.migrate_bundle(bundle)
    manifest = _read_json(bundle / MANIFEST_NAME)
    _, size = _digest(bundle / artifact)
    manifest["stages"]["acquire"]["outputs"].append(
        {"path": artifact, "sha256": "0" * 64, "bytes": size}
    )
    _write_json(bundle / MANIFEST_NAME, manifest)
    before = _tree_state(bundle.parent)
    with pytest.raises(MigrationError, match="integrity"):
        migrations.migrate_bundle(bundle)
    assert _tree_state(bundle.parent) == before


def test_migration_preserves_valid_alternate_selected_evidence_and_registered_reads(
    tmp_path: Path,
) -> None:
    state_path, bundle, bundle_id, segment_id, _ = registered_legacy_bundle(tmp_path)
    source = _read_json(bundle / "source.json")
    original_path = bundle / "transcript/segments.json"
    segments = _read_json_array(original_path)
    segments[0]["text"] = " \tעברית synthetic alternate evidence\n日本語 \u3000"
    alternate = bundle / "transcript/alternate.json"
    _write_json(alternate, segments)
    source["transcript"]["segments"] = "transcript/alternate.json"
    _write_json(bundle / "source.json", source)
    manifest = _read_json(bundle / MANIFEST_NAME)
    digest, size = _digest(alternate)
    manifest["stages"]["transcribe"]["outputs"].append(
        {"path": "transcript/alternate.json", "sha256": digest, "bytes": size}
    )
    _rewrite_manifest_hashes(bundle, manifest)
    _write_json(bundle / MANIFEST_NAME, manifest)
    before_alternate = alternate.read_bytes()
    before_original = original_path.read_bytes()
    migrations.migrate_bundle(bundle)
    assert migrations.migrate_bundle(bundle).outcome == "already_current"
    assert alternate.read_bytes() == before_alternate
    assert original_path.read_bytes() == before_original
    backup = bundle.with_name(bundle.name + BACKUP_SUFFIX)
    assert (backup / "transcript/alternate.json").read_bytes() == before_alternate
    with open_state(state_path) as state:
        assert state.search_segments("synthetic alternate evidence")[0].bundle_id == bundle_id
        _, resolved = state.cite_segment(bundle_id, segment_id)
        assert resolved.current_text == segments[0]["text"]


@pytest.mark.parametrize("label", ["/Users/alice/Research Talks", r"C:\Users\Alice\Research Talks"])
def test_migration_preserves_user_authored_path_shaped_source_labels(
    tmp_path: Path, label: str
) -> None:
    _, bundle, _, _, _ = registered_legacy_bundle(tmp_path)
    source_path = bundle / "source.json"
    source = _read_json(source_path)
    source["provenance"]["source_name"] = label
    _write_json(source_path, source)
    _repair_manifest_after_artifact_change(bundle)
    original = _tree_state(bundle)
    prepared = prepare_bundle_migration(bundle)
    assert _read_json(prepared.staging_dir / "source.json")["provenance"]["source_name"] == label
    assert _tree_state(bundle) == original
    migrations.migrate_bundle(bundle)
    assert _read_json(source_path)["provenance"]["source_name"] == label
    backup = bundle.with_name(bundle.name + BACKUP_SUFFIX)
    assert _tree_state(backup) == original


def test_current_noop_refuses_integral_float_representation_without_rewriting_original(
    tmp_path: Path,
) -> None:
    bundle = legacy_bundle(tmp_path)
    migrations.migrate_bundle(bundle)
    path = bundle / MANIFEST_NAME
    manifest = _read_json(path)
    integer_size = manifest["source"]["bytes"]
    manifest["source"]["bytes"] = float(integer_size)
    _write_json(path, manifest)
    assert Manifest.load(bundle).source.bytes == integer_size
    before = _tree_state(bundle.parent)
    with pytest.raises(MigrationError, match="current artifact models"):
        migrations.migrate_bundle(bundle)
    assert _tree_state(bundle.parent) == before


@pytest.mark.parametrize("current", [False, True])
def test_migration_refuses_nul_artifact_path_without_changing_evidence(
    tmp_path: Path, current: bool
) -> None:
    bundle = legacy_bundle(tmp_path)
    if current:
        migrations.migrate_bundle(bundle)
    manifest = _read_json(bundle / MANIFEST_NAME)
    manifest["stages"]["normalize"]["outputs"].append(
        {"path": "media/null\x00artifact", "sha256": "0" * 64, "bytes": 0}
    )
    _write_json(bundle / MANIFEST_NAME, manifest)
    before = _tree_state(bundle.parent)
    with pytest.raises(MigrationError):
        migrations.migrate_bundle(bundle)
    assert _tree_state(bundle.parent) == before


def _contradict_metadata_identity(bundle: Path, contradiction: str) -> None:
    path = bundle / "transcript/metadata.json"
    metadata = _read_json(path)
    source = _read_json(bundle / "source.json")
    if contradiction == "source-sha":
        metadata["source_media"]["sha256"] = "0" * 64
    elif contradiction == "source-size":
        metadata["source_media"]["bytes"] = source["bytes"] + 1
    elif contradiction == "source-zero":
        metadata["source_media"]["bytes"] = 0
    elif contradiction == "normalized-sha":
        metadata["normalized_audio"]["sha256"] = "0" * 64
    elif contradiction == "normalized-size":
        metadata["normalized_audio"]["bytes"] += 1
    else:
        metadata["normalized_audio"]["path"] = "media/absent.wav"
    _write_json(path, metadata)
    manifest = _read_json(bundle / MANIFEST_NAME)
    _rewrite_manifest_hashes(bundle, manifest)
    _write_json(bundle / MANIFEST_NAME, manifest)


@pytest.mark.parametrize(
    "contradiction",
    [
        "source-sha",
        "source-size",
        "source-zero",
        "normalized-sha",
        "normalized-size",
        "normalized-missing",
    ],
)
@pytest.mark.parametrize(
    "mode", ["prepare", "direct", "current", "staging", "forward", "current-marker"]
)
def test_migration_refuses_metadata_identity_contradictions(
    tmp_path: Path, contradiction: str, mode: str
) -> None:
    bundle = legacy_bundle(tmp_path)
    unrelated = bundle.parent / "unrelated"
    unrelated.mkdir()
    (unrelated / "keep").write_bytes(b"unrelated evidence")
    backup = bundle.with_name(bundle.name + BACKUP_SUFFIX)
    staging = bundle.with_name(bundle.name + STAGING_SUFFIX)
    if mode == "current":
        migrations.migrate_bundle(bundle)
    if mode in {"staging", "forward", "current-marker"}:
        prepared = prepare_bundle_migration(bundle)
        _contradict_metadata_identity(prepared.staging_dir, contradiction)
    _contradict_metadata_identity(bundle, contradiction)
    legacy_before = _tree_state(bundle)
    if mode in {"forward", "current-marker"}:
        bundle.rename(backup)
        if mode == "current-marker":
            staging.rename(bundle)
    before = _tree_state(bundle.parent)
    with pytest.raises(MigrationError) as error:
        (prepare_bundle_migration if mode == "prepare" else migrations.migrate_bundle)(bundle)
    assert str(tmp_path) not in str(error.value)
    assert (unrelated / "keep").read_bytes() == b"unrelated evidence"
    if mode == "forward":
        assert _tree_state(bundle) == legacy_before
        assert not backup.exists()
    elif mode in {"prepare", "direct", "staging"}:
        assert _tree_state(bundle) == legacy_before
        assert not staging.exists()
        assert not backup.exists()
    else:
        assert _tree_state(bundle.parent) == before


@pytest.mark.parametrize("source_size", ["absent", "null", "matching"])
@pytest.mark.parametrize("alternate", [False, True])
def test_migration_metadata_optional_size_and_named_paths(
    tmp_path: Path, source_size: str, alternate: bool
) -> None:
    bundle = legacy_bundle(tmp_path)
    source = _read_json(bundle / "source.json")
    metadata = _read_json(bundle / "transcript/metadata.json")
    if source_size == "absent":
        metadata["source_media"].pop("bytes", None)
    else:
        metadata["source_media"]["bytes"] = None if source_size == "null" else source["bytes"]
    metadata_path = bundle / "transcript/metadata.json"
    manifest = _read_json(bundle / MANIFEST_NAME)
    if alternate:
        metadata_path = bundle / "transcript/alternate-metadata.json"
        source["transcript"]["metadata"] = "transcript/alternate-metadata.json"
        _write_json(bundle / "source.json", source)
        named_audio = bundle / "media/alternate.wav"
        named_audio.write_bytes((bundle / metadata["normalized_audio"]["path"]).read_bytes())
        metadata["normalized_audio"]["path"] = "media/alternate.wav"
        # Named normalized identity does not add a declaration-membership requirement.
        manifest["stages"]["transcribe"]["outputs"].append(
            {"path": "transcript/alternate-metadata.json", "sha256": "0" * 64, "bytes": 0}
        )
    _write_json(metadata_path, metadata)
    _rewrite_manifest_hashes(bundle, manifest)
    _write_json(bundle / MANIFEST_NAME, manifest)
    original = _tree_state(bundle)
    assert migrations.migrate_bundle(bundle).outcome == "migrated"
    assert _tree_state(bundle.with_name(bundle.name + BACKUP_SUFFIX)) == original
    current = _read_json(metadata_path)
    assert current["source_media"].get("bytes") == metadata["source_media"].get("bytes")
    assert ("bytes" in current["source_media"]) == ("bytes" in metadata["source_media"])
    assert current["normalized_audio"] == metadata["normalized_audio"]
    before = _tree_state(bundle)
    assert migrations.migrate_bundle(bundle).outcome == "already_current"
    assert _tree_state(bundle) == before


@pytest.mark.parametrize("command", [False, True], ids=["sidecar", "command"])
def test_migration_preserves_distinct_original_and_normalized_identity(
    tmp_path: Path, command: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    import wave

    from lectern import ingest as ingest_module

    real_which = shutil.which

    def without_ffmpeg(name: str, *args: Any, **kwargs: Any) -> str | None:
        return None if name == "ffmpeg" else real_which(name, *args, **kwargs)

    monkeypatch.setattr(shutil, "which", without_ffmpeg)

    media = tmp_path / "original.wav"
    with wave.open(str(media), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(8000)
        audio.writeframes(b"\x00\x00" * 8000)
    original_bytes = media.read_bytes()
    normalization_calls = 0

    def normalize_synthetic(captured: Path, output: Path) -> None:
        nonlocal normalization_calls
        normalization_calls += 1
        assert captured != media
        assert captured.read_bytes() == original_bytes
        assert shutil.which("ffmpeg") is None
        with wave.open(str(output), "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(16000)
            audio.writeframes(b"\x00\x00" * 16000)

    monkeypatch.setattr(ingest_module, "_normalize_to_canonical_wav", normalize_synthetic)
    media.with_suffix(".transcript.txt").write_text("Synthetic transcript.")
    script = tmp_path / "transcriber.py"
    script.write_text('print(\'{"text": "Synthetic command transcript."}\')\n')
    bundle = ingest_local(
        media,
        tmp_path / "bundles",
        transcriber_command=f"{sys.executable} {script}" if command else None,
    ).bundle_dir
    assert normalization_calls == 1
    metadata = _read_json(bundle / "transcript/metadata.json")
    assert metadata["normalized_audio"]["sha256"] == _digest(bundle / "media/audio.wav")[0]
    assert metadata["normalized_audio"]["bytes"] == _digest(bundle / "media/audio.wav")[1]
    assert metadata["source_media"]["sha256"] == _digest(media)[0]
    assert metadata["source_media"]["sha256"] != metadata["normalized_audio"]["sha256"]
    assert "bytes" not in metadata["source_media"]
    assert migrations.migrate_bundle(bundle).outcome == "already_current"
    if not command:
        _write_legacy_shape(bundle, media)
        assert migrations.migrate_bundle(bundle).outcome == "migrated"
    assert (
        _read_json(bundle / "transcript/metadata.json")["normalized_audio"]
        == metadata["normalized_audio"]
    )


@pytest.mark.parametrize(
    "raw_id",
    ["/Users/synthetic/private-talk", r"C:\Users\synthetic\private-talk", "ordinary-bundle-id"],
)
@pytest.mark.parametrize("mode", ["migrated", "already_current", "recovered"])
def test_cli_migration_projects_path_bearing_id_without_rewriting_identity(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], raw_id: str, mode: str
) -> None:
    from lectern.records import PATH_REDACTED

    bundle = legacy_bundle(tmp_path)
    manifest = _read_json(bundle / MANIFEST_NAME)
    manifest["bundle_id"] = raw_id
    _write_json(bundle / MANIFEST_NAME, manifest)
    legacy_before = _tree_state(bundle)
    if mode == "already_current":
        assert migrations.migrate_bundle(bundle).bundle_id == raw_id
    elif mode == "recovered":
        prepared = prepare_bundle_migration(bundle)
        bundle.rename(prepared.backup_dir)
    capsys.readouterr()
    assert cli.main(["migrate", str(bundle)]) == 0
    captured = capsys.readouterr()
    assert captured.err == (
        "migrate: already readable; 1.0.0 is a subset of 1.1.0\n"
        if mode == "already_current"
        else ""
    )
    expected_id = raw_id if raw_id == "ordinary-bundle-id" else PATH_REDACTED
    assert json.loads(captured.out) == {
        "bundle_id": expected_id,
        "source_version": "1.0.0" if mode == "already_current" else "0.1.0",
        "target_version": "1.0.0",
        "outcome": mode,
        "backup_retained": True,
    }
    assert Manifest.load(bundle).bundle_id == raw_id
    backup = bundle.with_name(bundle.name + BACKUP_SUFFIX)
    assert _tree_state(backup) == legacy_before
    assert not bundle.with_name(bundle.name + STAGING_SUFFIX).exists()
    before = _tree_state(bundle.parent)
    assert migrations.migrate_bundle(bundle).bundle_id == raw_id
    assert _tree_state(bundle.parent) == before


@pytest.mark.parametrize("phase", ["backup", "publication", "forward", "restoration"])
def test_late_empty_destination_never_replaces_unrelated_role(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    bundle = legacy_bundle(tmp_path)
    original = _tree_state(bundle)
    prepared = prepare_bundle_migration(bundle)
    if phase in {"forward", "restoration"}:
        bundle.rename(prepared.backup_dir)
        if phase == "restoration":
            shutil.rmtree(prepared.staging_dir)
    target = prepared.backup_dir if phase == "backup" else bundle
    real_rename = migrations._rename  # pyright: ignore[reportPrivateUsage]
    inserted: list[tuple[int, int, int]] = []

    def race(source: Path, destination: Path) -> None:
        if destination == target and not inserted:
            destination.mkdir(mode=0o750)
            record = destination.lstat()
            inserted.append((record.st_dev, record.st_ino, record.st_mode))
        real_rename(source, destination)

    monkeypatch.setattr(migrations, "_rename", race)
    with pytest.raises(MigrationError):
        migrations.migrate_bundle(bundle)
    assert len(inserted) == 1
    record = target.lstat()
    assert (record.st_dev, record.st_ino, record.st_mode) == inserted[0]
    assert list(target.iterdir()) == []
    assert _tree_state(bundle if phase == "backup" else prepared.backup_dir) == original
    if phase != "restoration":
        assert (prepared.staging_dir / MARKER_NAME).is_file()


@pytest.mark.parametrize("destination_kind", ["absent", "directory", "file", "symlink"])
def test_native_rename_excludes_existing_unicode_destination(
    tmp_path: Path, destination_kind: str
) -> None:
    source = tmp_path / "原本"
    destination = tmp_path / "目標"
    source.mkdir(mode=0o750)
    (source / "evidence").write_bytes(b"original")
    if destination_kind == "directory":
        destination.mkdir(mode=0o700)
    elif destination_kind == "file":
        destination.write_bytes(b"unrelated")
    elif destination_kind == "symlink":
        destination.symlink_to(tmp_path / "absent")
    before = _tree_state(tmp_path)
    identity = source.lstat()
    if destination_kind == "absent":
        migrations._rename(source, destination)  # pyright: ignore[reportPrivateUsage]
        assert not source.exists()
        after = destination.lstat()
        assert (after.st_dev, after.st_ino, after.st_mode) == (
            identity.st_dev,
            identity.st_ino,
            identity.st_mode,
        )
        assert (destination / "evidence").read_bytes() == b"original"
    else:
        occupied = destination.lstat()
        with pytest.raises(FileExistsError):
            migrations._rename(source, destination)  # pyright: ignore[reportPrivateUsage]
        assert _tree_state(tmp_path) == before
        after = destination.lstat()
        assert (after.st_dev, after.st_ino, after.st_mode) == (
            occupied.st_dev,
            occupied.st_ino,
            occupied.st_mode,
        )
        assert source.lstat().st_ino == identity.st_ino


@pytest.mark.parametrize(
    "platform,symbol,flag", [("darwin", "renameatx_np", 4), ("linux", "renameat2", 1)]
)
@pytest.mark.parametrize("error", [0, 17, 13, 38, 22, 45])
def test_exclusive_rename_abi_errno_and_descriptor_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    platform: str,
    symbol: str,
    flag: int,
    error: int,
) -> None:
    import ctypes
    import errno
    from types import SimpleNamespace

    # Use host errno values even when simulating the other platform's ABI.
    error = {38: errno.ENOSYS, 45: errno.ENOTSUP}.get(error, error)
    source = tmp_path / "原本"
    destination = tmp_path / "目標"
    source.mkdir()
    calls: list[int] = []

    class Native:
        argtypes: list[Any] = []
        restype: Any = None

        def __call__(self, from_fd: int, old: bytes, to_fd: int, new: bytes, flags: int) -> int:
            assert from_fd == to_fd
            assert stat.S_ISDIR(os.fstat(from_fd).st_mode)
            assert (old, new, flags) == (
                os.fsencode(source.name),
                os.fsencode(destination.name),
                flag,
            )
            calls.append(from_fd)
            ctypes.set_errno(error)
            return -1 if error else 0

    native = Native()

    def load(name: object, *, use_errno: bool) -> Any:
        assert name is None and use_errno is True
        return SimpleNamespace(**{symbol: native})

    def no_fallback(*args: Any, **kwargs: Any) -> None:
        pytest.fail("nonexclusive fallback was invoked")

    monkeypatch.setattr(migrations.sys, "platform", platform)
    monkeypatch.setattr(ctypes, "CDLL", load)
    monkeypatch.setattr(os, "rename", no_fallback)
    monkeypatch.setattr(os, "replace", no_fallback)
    monkeypatch.setattr(shutil, "copytree", no_fallback)
    monkeypatch.setattr(shutil, "rmtree", no_fallback)
    if error:
        expected = (
            MigrationError if error in {errno.ENOSYS, errno.EINVAL, errno.ENOTSUP} else OSError
        )
        with pytest.raises(expected) as raised:
            migrations._rename(source, destination)  # pyright: ignore[reportPrivateUsage]
        assert str(tmp_path) not in str(raised.value)
        if isinstance(raised.value, OSError):
            assert raised.value.errno == error
    else:
        migrations._rename(source, destination)  # pyright: ignore[reportPrivateUsage]
    assert native.argtypes == [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    assert native.restype is ctypes.c_int
    assert len(calls) == 1
    with pytest.raises(OSError) as closed:
        os.fstat(calls[0])
    assert closed.value.errno == errno.EBADF
    assert source.is_dir() and not destination.exists()


@pytest.mark.parametrize("failure", ["platform", "symbol", "load", "nul", "nonsibling"])
def test_exclusive_rename_unavailable_or_invalid_never_opens_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    import ctypes
    from types import SimpleNamespace

    source = tmp_path / "source"
    destination = tmp_path / "destination"
    if failure == "nul":
        destination = tmp_path / "destination\x00truncated"
    elif failure == "nonsibling":
        destination = tmp_path / "other" / "destination"
    loads = 0

    def load(*args: Any, **kwargs: Any) -> Any:
        nonlocal loads
        loads += 1
        if failure == "load":
            raise OSError("synthetic loader failure")
        return SimpleNamespace()

    def unexpected(*args: Any, **kwargs: Any) -> None:
        pytest.fail("invalid or unsupported operation reached filesystem mutation")

    monkeypatch.setattr(
        migrations.sys, "platform", "unsupported" if failure == "platform" else "linux"
    )
    monkeypatch.setattr(ctypes, "CDLL", load)
    monkeypatch.setattr(os, "open", unexpected)
    monkeypatch.setattr(os, "rename", unexpected)
    with pytest.raises(MigrationError) as raised:
        migrations._rename(source, destination)  # pyright: ignore[reportPrivateUsage]
    assert str(tmp_path) not in str(raised.value)
    assert loads == (1 if failure in {"symbol", "load"} else 0)


def test_current_migration_does_not_load_publication_primitive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import ctypes

    bundle = legacy_bundle(tmp_path)
    migrations.migrate_bundle(bundle)

    def unavailable(*args: Any, **kwargs: Any) -> None:
        pytest.fail("no-op loaded a publication primitive")

    monkeypatch.setattr(ctypes, "CDLL", unavailable)
    assert migrations.migrate_bundle(bundle).outcome == "already_current"
