"""Fail-closed migration support for legacy Lectern bundles."""

from __future__ import annotations

import copy
import ctypes
import errno
import hashlib
import json
import os
import re
import shutil
import stat
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal, cast

from lectern.bundle import (
    MANIFEST_NAME,
    Manifest,
    SourceKind,
    TranscriptMetadataDocument,
    atomic_write_text,
    schema_version_is_compatible,
)
from lectern.evidence import read_selected_evidence
from lectern.records import redact_paths

LEGACY_SCHEMA_VERSION = "0.1.0"
TARGET_SCHEMA_VERSION = "1.0.0"
BACKUP_SUFFIX = ".v0.1.0.bak"
STAGING_SUFFIX = ".migrating-0.1.0-to-1.0.0"
MARKER_NAME = ".lectern-migration.json"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_MAX_JSON_DEPTH = 128


class MigrationError(RuntimeError):
    """A path-projected refusal to migrate one bundle."""


@dataclass(frozen=True)
class TreeEntry:
    path: PurePosixPath
    kind: Literal["directory", "file"]
    sha256: str | None = None
    bytes: int | None = None


TreeSnapshot = tuple[TreeEntry, ...]


@dataclass(frozen=True)
class PreparedBundleMigration:
    bundle_id: str
    source_version: str
    target_version: str
    source_sha256: str
    source_bytes: int
    source_dir: Path
    staging_dir: Path
    backup_dir: Path
    source_snapshot: TreeSnapshot
    target_snapshot: TreeSnapshot


@dataclass(frozen=True)
class BundleMigrationResult:
    bundle_id: str
    source_version: str
    target_version: str
    outcome: Literal["migrated", "recovered", "already_current"]
    backup_retained: bool

    def to_public_dict(self) -> dict[str, str | bool]:
        return {
            "bundle_id": redact_paths(self.bundle_id),
            "source_version": self.source_version,
            "target_version": self.target_version,
            "outcome": self.outcome,
            "backup_retained": self.backup_retained,
        }


def _reject_non_finite_json(constant: str) -> None:
    raise ValueError(f"non-finite JSON constant {constant!r}")


def _assert_json_depth(value: object) -> None:
    stack: list[tuple[object, int]] = [(value, 0)]
    while stack:
        current, parent_depth = stack.pop()
        if isinstance(current, dict):
            depth = parent_depth + 1
            if depth > _MAX_JSON_DEPTH:
                raise ValueError("JSON nesting exceeds the migration limit")
            mapping = cast(dict[object, object], current)
            stack.extend((item, depth) for item in mapping.values())
        elif isinstance(current, list):
            depth = parent_depth + 1
            if depth > _MAX_JSON_DEPTH:
                raise ValueError("JSON nesting exceeds the migration limit")
            sequence = cast(list[object], current)
            stack.extend((item, depth) for item in sequence)


def _read_json_value(path: Path, role: str) -> tuple[str, object]:
    try:
        raw = path.read_text(encoding="utf-8")
        value: object = json.loads(raw, parse_constant=_reject_non_finite_json)
        _assert_json_depth(value)
    except (OSError, UnicodeError, ValueError, RecursionError) as exc:
        raise MigrationError(f"cannot read valid {role}") from exc
    return raw, value


def _read_object_with_text(path: Path, role: str) -> tuple[str, dict[str, Any]]:
    raw, value = _read_json_value(path, role)
    if not isinstance(value, dict):
        raise MigrationError(f"{role} must contain a JSON object")
    return raw, cast(dict[str, Any], value)


def _read_object(path: Path, role: str) -> dict[str, Any]:
    return _read_object_with_text(path, role)[1]


def _object_field(value: dict[str, Any], key: str, role: str) -> dict[str, Any]:
    field = value.get(key)
    if not isinstance(field, dict):
        raise MigrationError(f"{role} is malformed")
    return cast(dict[str, Any], field)


def _source_path(path: Path) -> Path:
    try:
        source = Path(os.path.abspath(path.expanduser()))
    except (OSError, RuntimeError) as exc:
        raise MigrationError("cannot normalize the source role") from exc
    if not source.name:
        raise MigrationError("source role must name one bundle directory")
    return source


def _role_exists(path: Path, role: str) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise MigrationError(f"cannot inspect the {role}") from exc
    return True


def _digest(path: Path) -> tuple[str, int]:
    hasher = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(chunk)
                size += len(chunk)
    except OSError as exc:
        raise MigrationError("cannot read a declared bundle artifact") from exc
    return hasher.hexdigest(), size


def _tree_snapshot(bundle: Path, role: str) -> TreeSnapshot:
    entries: list[TreeEntry] = []
    try:
        paths = sorted(
            bundle.rglob("*"),
            key=lambda item: item.relative_to(bundle).as_posix(),
        )
        for path in paths:
            relative = PurePosixPath(path.relative_to(bundle).as_posix())
            if path.is_symlink():
                raise MigrationError(f"{role} role contains a symlink")
            if path.is_dir():
                entries.append(TreeEntry(relative, "directory"))
            elif path.is_file():
                digest, size = _digest(path)
                entries.append(TreeEntry(relative, "file", digest, size))
            else:
                raise MigrationError(f"{role} role contains an unsupported entry")
    except OSError as exc:
        raise MigrationError(f"cannot snapshot the {role} role") from exc
    return tuple(entries)


def _assert_snapshot(bundle: Path, expected: TreeSnapshot, role: str) -> None:
    if _tree_snapshot(bundle, role) != expected:
        raise MigrationError(f"{role} tree changed during migration preparation")


def _artifact_path(bundle: Path, raw: object) -> Path:
    if not isinstance(raw, str):
        raise MigrationError("declared artifact path must be text")
    relative = PurePosixPath(raw)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise MigrationError("declared artifact path escapes the bundle")
    path = bundle.joinpath(*relative.parts)
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise MigrationError("declared bundle artifact is missing or not a regular file") from exc
    except (OSError, ValueError) as exc:
        raise MigrationError("cannot inspect a declared bundle artifact") from exc
    if not stat.S_ISREG(mode):
        raise MigrationError("declared bundle artifact is missing or not a regular file")
    return path


def _transcript_artifact_path(
    bundle: Path,
    source_document: dict[str, Any],
    field: str,
) -> tuple[PurePosixPath, Path]:
    transcript = _object_field(source_document, "transcript", "source metadata transcript")
    raw = transcript.get(field)
    path = _artifact_path(bundle, raw)
    return PurePosixPath(cast(str, raw)), path


def _assert_no_symlinks(bundle: Path, role: str) -> None:
    if not _role_exists(bundle, f"{role} role") or bundle.is_symlink() or not bundle.is_dir():
        raise MigrationError(f"{role} role must be a real bundle directory")
    try:
        contains_symlink = any(path.is_symlink() for path in bundle.rglob("*"))
    except OSError as exc:
        raise MigrationError(f"cannot inspect the {role} role") from exc
    if contains_symlink:
        raise MigrationError(f"{role} role contains a symlink")


def _manifest_fields(manifest: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    bundle_id = manifest.get("bundle_id")
    if not isinstance(bundle_id, str) or not bundle_id:
        raise MigrationError("manifest bundle ID is malformed")
    source = _object_field(manifest, "source", "manifest source")
    if source.get("kind") not in {kind.value for kind in SourceKind}:
        raise MigrationError("manifest source kind is malformed")
    if not isinstance(source.get("ref"), str) or not source["ref"]:
        raise MigrationError("manifest source reference is malformed")
    return bundle_id, source


def _assert_declared_integrity(bundle: Path, manifest: dict[str, Any]) -> None:
    stages = manifest.get("stages")
    if not isinstance(stages, dict):
        raise MigrationError("manifest stages are malformed")
    for stage in cast(dict[str, Any], stages).values():
        if not isinstance(stage, dict):
            raise MigrationError("manifest stage outputs are malformed")
        stage_record = cast(dict[str, Any], stage)
        outputs = stage_record.get("outputs")
        if not isinstance(outputs, list):
            raise MigrationError("manifest stage outputs are malformed")
        for item in cast(list[Any], outputs):
            if not isinstance(item, dict):
                raise MigrationError("manifest artifact record is malformed")
            record = cast(dict[str, Any], item)
            expected_digest = record.get("sha256")
            expected_size = record.get("bytes")
            if not isinstance(expected_digest, str) or _HEX64.fullmatch(expected_digest) is None:
                raise MigrationError("manifest artifact digest is malformed")
            if (
                not isinstance(expected_size, int)
                or isinstance(expected_size, bool)
                or expected_size < 0
            ):
                raise MigrationError("manifest artifact size is malformed")
            digest, size = _digest(_artifact_path(bundle, record.get("path")))
            if expected_digest != digest or expected_size != size:
                raise MigrationError("declared artifact integrity does not match bundle bytes")


def _source_identity(bundle: Path, expected_source: dict[str, Any]) -> tuple[str, int]:
    source = _read_object(bundle / "source.json", "source metadata")
    digest = source.get("sha256")
    size = source.get("bytes")
    if not isinstance(digest, str) or _HEX64.fullmatch(digest) is None:
        raise MigrationError("source metadata lacks a valid sha256 identity")
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise MigrationError("source metadata lacks a valid byte size")
    source_fields = _object_field(source, "source", "source metadata source")
    if source_fields.get("kind") not in {kind.value for kind in SourceKind}:
        raise MigrationError("source metadata kind is malformed")
    if not isinstance(source_fields.get("ref"), str) or not source_fields["ref"]:
        raise MigrationError("source metadata reference is malformed")
    if source_fields != expected_source:
        raise MigrationError("manifest and source metadata identities disagree")
    return digest, size


def _marker_payload(bundle_id: str) -> dict[str, str]:
    return {
        "bundle_id": bundle_id,
        "source_version": LEGACY_SCHEMA_VERSION,
        "target_version": TARGET_SCHEMA_VERSION,
    }


def _marker_matches(staging: Path, bundle_id: str) -> bool:
    marker_path = staging / MARKER_NAME
    try:
        mode = marker_path.lstat().st_mode
    except OSError:
        return False
    if not stat.S_ISREG(mode):
        return False
    try:
        marker = _read_object(marker_path, "migration marker")
    except MigrationError:
        return False
    return marker == _marker_payload(bundle_id)


def _remove_owned_staging(staging: Path, bundle_id: str) -> None:
    if not _marker_matches(staging, bundle_id):
        raise MigrationError("staging role is ambiguous")
    try:
        shutil.rmtree(staging)
    except OSError as exc:
        raise MigrationError("cannot remove the owned staging role") from exc


def _json_text(value: object) -> str:
    try:
        return json.dumps(value, indent=2, allow_nan=False) + "\n"
    except (ValueError, RecursionError) as exc:
        raise MigrationError("cannot serialize migration JSON") from exc


def _copy_json_object(value: dict[str, Any], role: str) -> dict[str, Any]:
    try:
        return copy.deepcopy(value)
    except RecursionError as exc:
        raise MigrationError(f"cannot copy {role}") from exc


def _write_json(path: Path, value: object, role: str) -> None:
    try:
        atomic_write_text(path, _json_text(value))
    except OSError as exc:
        raise MigrationError(f"cannot write {role}") from exc


def _assert_json_document(path: Path, expected: dict[str, Any], role: str) -> None:
    try:
        actual = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise MigrationError(f"cannot read {role}") from exc
    if actual != _json_text(expected):
        raise MigrationError(f"{role} differs from approved transformation")


def _portable_command_name(value: str) -> str:
    """Retain executable identity without retaining its machine-local location."""

    return PurePosixPath(value.replace("\\", "/")).name


def _derive_target_documents(
    source: Path,
    manifest: dict[str, Any],
    source_sha256: str,
    source_bytes: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest_source = _object_field(manifest, "source", "manifest source")
    source_document = _read_object(source / "source.json", "source metadata")
    source_record = _object_field(source_document, "source", "source metadata source")
    if manifest_source.get("kind") == SourceKind.LOCAL.value:
        ref = f"sha256:{source_sha256}"
        manifest_source["ref"] = ref
        manifest_source["bytes"] = source_bytes
        source_record["ref"] = ref
        source_record["bytes"] = source_bytes

    stages = manifest.get("stages")
    if isinstance(stages, dict):
        for stage in cast(dict[str, Any], stages).values():
            if isinstance(stage, dict):
                record = cast(dict[str, Any], stage)
                if isinstance(record.get("error"), str):
                    record["error"] = redact_paths(cast(str, record["error"]))

    sidecar = source_document.get("transcript_sidecar")
    if isinstance(sidecar, dict):
        cast(dict[str, Any], sidecar).pop("path", None)

    _, metadata_path = _transcript_artifact_path(source, source_document, "metadata")
    metadata = _read_object(metadata_path, "transcript metadata")
    for key in ("source_media", "backend"):
        value = metadata.get(key)
        if isinstance(value, dict):
            cast(dict[str, Any], value).pop("path", None)
    backend = metadata.get("backend")
    if isinstance(backend, dict):
        backend_record = cast(dict[str, Any], backend)
        backend_record.pop("command", None)
        argv0 = backend_record.get("argv0")
        if isinstance(argv0, str):
            backend_record["argv0"] = _portable_command_name(argv0)
    return manifest, source_document, metadata


def _refresh_output_records(staging: Path, manifest: dict[str, Any]) -> None:
    for stage in cast(dict[str, Any], manifest["stages"]).values():
        for item in cast(list[Any], cast(dict[str, Any], stage)["outputs"]):
            record = cast(dict[str, Any], item)
            digest, size = _digest(_artifact_path(staging, record["path"]))
            record["sha256"] = digest
            record["bytes"] = size


_STATIC_TARGET_DOCUMENTS = {
    PurePosixPath(MANIFEST_NAME),
    PurePosixPath("source.json"),
}


def _without_paths(snapshot: TreeSnapshot, paths: set[PurePosixPath]) -> TreeSnapshot:
    return tuple(entry for entry in snapshot if entry.path not in paths)


def _assert_preserved_tree(
    target: Path,
    source_snapshot: TreeSnapshot,
    target_documents: set[PurePosixPath],
) -> None:
    target_snapshot = _tree_snapshot(target, "target")
    target_ignored = target_documents | {PurePosixPath(MARKER_NAME)}
    if _without_paths(target_snapshot, target_ignored) != _without_paths(
        source_snapshot, target_documents
    ):
        raise MigrationError("target changed a non-target source entry")


def _validate_target(
    target: Path,
    *,
    expected_bundle_id: str,
    source_sha256: str,
    source_bytes: int,
    schema_version: str = TARGET_SCHEMA_VERSION,
) -> Manifest:
    _assert_no_symlinks(target, "target")
    manifest_text, manifest_raw = _read_object_with_text(target / MANIFEST_NAME, "target manifest")
    if manifest_raw.get("schema_version") != schema_version:
        raise MigrationError(f"target manifest does not declare schema {schema_version}")
    stages = manifest_raw.get("stages")
    if isinstance(stages, dict):
        for stage in cast(dict[str, Any], stages).values():
            if isinstance(stage, dict):
                error = cast(dict[str, Any], stage).get("error")
                if isinstance(error, str) and error != redact_paths(error):
                    raise MigrationError("target manifest contains a path-bearing stage error")
    _assert_declared_integrity(target, manifest_raw)
    _, source_raw = _read_object_with_text(target / "source.json", "target source metadata")
    _, metadata_path = _transcript_artifact_path(target, source_raw, "metadata")
    metadata_text, metadata_raw = _read_object_with_text(
        metadata_path, "target transcript metadata"
    )
    _, segments_path = _transcript_artifact_path(target, source_raw, "segments")
    _read_json_value(segments_path, "target transcript segments")
    try:
        manifest = Manifest.model_validate_json(manifest_text, strict=True)
        evidence = read_selected_evidence(target, manifest, require_manifest_integrity=True)
        source_document = evidence.source
        metadata_document = TranscriptMetadataDocument.model_validate_json(
            metadata_text, strict=True
        )
    except (OSError, ValueError, RecursionError) as exc:
        raise MigrationError("target does not satisfy current artifact models") from exc
    if manifest.bundle_id != expected_bundle_id:
        raise MigrationError("target bundle ID differs from source bundle ID")
    if source_document.source != manifest.source:
        raise MigrationError("target manifest and source metadata disagree")
    if source_document.sha256 != source_sha256 or source_document.bytes != source_bytes:
        raise MigrationError("target source identity differs from source evidence")
    if metadata_document.source_media.sha256 != source_document.sha256 or (
        metadata_document.source_media.bytes is not None
        and metadata_document.source_media.bytes != source_document.bytes
    ):
        raise MigrationError(
            "target transcript metadata source identity disagrees with source evidence"
        )
    normalized = metadata_document.normalized_audio
    normalized_digest, normalized_size = _digest(_artifact_path(target, normalized.path))
    if normalized.sha256 != normalized_digest or normalized.bytes != normalized_size:
        raise MigrationError("target normalized audio identity disagrees with its named artifact")
    if manifest.source.kind is SourceKind.LOCAL:
        expected_ref = f"sha256:{source_sha256}"
        if manifest.source.ref != expected_ref or manifest.source.bytes != source_bytes:
            raise MigrationError("target local source identity is incorrect")
    sidecar = source_raw.get("transcript_sidecar")
    if isinstance(sidecar, dict) and "path" in sidecar:
        raise MigrationError("target transcript sidecar still carries a path")
    for key in ("source_media", "backend"):
        value = metadata_raw.get(key)
        if isinstance(value, dict) and "path" in value:
            raise MigrationError(f"target {key} metadata still carries a path")
    backend = metadata_raw.get("backend")
    if isinstance(backend, dict):
        argv0 = cast(dict[str, Any], backend).get("argv0")
        if isinstance(argv0, str) and argv0 != _portable_command_name(argv0):
            raise MigrationError("target backend argv0 still carries a path")
    return manifest


def _validate_prepared_target(
    target: Path,
    *,
    legacy_source: Path,
    source_snapshot: TreeSnapshot,
    expected_bundle_id: str,
    source_sha256: str,
    source_bytes: int,
) -> tuple[Manifest, TreeSnapshot]:
    legacy_manifest = _copy_json_object(
        _read_object(legacy_source / MANIFEST_NAME, "legacy manifest"),
        "legacy manifest",
    )
    legacy_manifest["schema_version"] = TARGET_SCHEMA_VERSION
    expected_manifest, expected_source, expected_metadata = _derive_target_documents(
        legacy_source,
        legacy_manifest,
        source_sha256,
        source_bytes,
    )
    metadata_relative, metadata_path = _transcript_artifact_path(
        target, expected_source, "metadata"
    )
    target_documents = _STATIC_TARGET_DOCUMENTS | {metadata_relative}

    def validate_source_bound_target() -> Manifest:
        if not _marker_matches(target, expected_bundle_id):
            raise MigrationError("target migration marker is invalid")
        _assert_json_document(target / "source.json", expected_source, "target source metadata")
        _assert_json_document(
            metadata_path,
            expected_metadata,
            "target transcript metadata",
        )
        _assert_preserved_tree(target, source_snapshot, target_documents)
        _refresh_output_records(target, expected_manifest)
        _assert_json_document(target / MANIFEST_NAME, expected_manifest, "target manifest")
        manifest = _validate_target(
            target,
            expected_bundle_id=expected_bundle_id,
            source_sha256=source_sha256,
            source_bytes=source_bytes,
        )
        _assert_preserved_tree(target, source_snapshot, target_documents)
        if not _marker_matches(target, expected_bundle_id):
            raise MigrationError("target migration marker is invalid")
        return manifest

    manifest = validate_source_bound_target()
    target_snapshot = _tree_snapshot(target, "target")
    manifest = validate_source_bound_target()
    _assert_snapshot(target, target_snapshot, "target")
    return manifest, target_snapshot


def prepare_bundle_migration(bundle_dir: Path) -> PreparedBundleMigration:
    if not schema_version_is_compatible(TARGET_SCHEMA_VERSION):
        raise RuntimeError("migration target is not readable by this build")
    source = _source_path(bundle_dir)
    _assert_no_symlinks(source, "source")
    if _role_exists(source / MARKER_NAME, "source migration marker"):
        raise MigrationError("source role contains a migration marker")
    raw = _read_object(source / MANIFEST_NAME, "source manifest")
    if raw.get("schema_version") != LEGACY_SCHEMA_VERSION:
        raise MigrationError("source manifest does not declare schema 0.1.0")
    bundle_id, manifest_source = _manifest_fields(raw)
    _assert_declared_integrity(source, raw)
    source_sha256, source_bytes = _source_identity(source, manifest_source)
    source_snapshot = _tree_snapshot(source, "source")

    staging = source.with_name(source.name + STAGING_SUFFIX)
    backup = source.with_name(source.name + BACKUP_SUFFIX)
    if _role_exists(backup, "backup role"):
        raise MigrationError("backup role is already occupied")
    if _role_exists(staging, "staging role"):
        raise MigrationError("staging role is already occupied")
    try:
        staging.mkdir(mode=0o700)
    except OSError as exc:
        raise MigrationError("cannot reserve the staging role") from exc
    try:
        _write_json(
            staging / MARKER_NAME,
            _marker_payload(bundle_id),
            "migration marker",
        )
    except MigrationError as exc:
        try:
            staging.rmdir()
        except OSError as cleanup_exc:
            raise MigrationError(
                "cannot initialize the staging role; empty staging cleanup also failed"
            ) from cleanup_exc
        raise MigrationError("cannot initialize the staging role") from exc
    try:
        shutil.copytree(source, staging, symlinks=True, dirs_exist_ok=True)
        copied = _tree_snapshot(staging, "staging")
        if _without_paths(copied, {PurePosixPath(MARKER_NAME)}) != source_snapshot:
            raise MigrationError("copied staging tree differs from accepted source")
        _assert_snapshot(source, source_snapshot, "source")
    except MigrationError:
        try:
            _remove_owned_staging(staging, bundle_id)
        except MigrationError as cleanup_exc:
            raise MigrationError(
                "cannot copy the source role and cannot remove owned staging"
            ) from cleanup_exc
        raise
    except OSError as exc:
        try:
            _remove_owned_staging(staging, bundle_id)
        except MigrationError as cleanup_exc:
            raise MigrationError(
                "cannot copy the source role and cannot remove owned staging"
            ) from cleanup_exc
        raise MigrationError("cannot copy the source role into staging") from exc
    try:
        target = _copy_json_object(raw, "source manifest")
        target["schema_version"] = TARGET_SCHEMA_VERSION
        target, expected_source, expected_metadata = _derive_target_documents(
            source, target, source_sha256, source_bytes
        )
        _write_json(staging / "source.json", expected_source, "source metadata")
        _, target_metadata_path = _transcript_artifact_path(staging, expected_source, "metadata")
        _write_json(
            target_metadata_path,
            expected_metadata,
            "transcript metadata",
        )
        _refresh_output_records(staging, target)
        _write_json(staging / MANIFEST_NAME, target, "target manifest")
        _, target_snapshot = _validate_prepared_target(
            staging,
            legacy_source=source,
            source_snapshot=source_snapshot,
            expected_bundle_id=bundle_id,
            source_sha256=source_sha256,
            source_bytes=source_bytes,
        )
        _assert_snapshot(source, source_snapshot, "source")
    except MigrationError:
        try:
            _remove_owned_staging(staging, bundle_id)
        except MigrationError as cleanup_exc:
            raise MigrationError(
                "target preparation failed and owned staging cleanup also failed"
            ) from cleanup_exc
        raise
    return PreparedBundleMigration(
        bundle_id=bundle_id,
        source_version=LEGACY_SCHEMA_VERSION,
        target_version=TARGET_SCHEMA_VERSION,
        source_sha256=source_sha256,
        source_bytes=source_bytes,
        source_dir=source,
        staging_dir=staging,
        backup_dir=backup,
        source_snapshot=source_snapshot,
        target_snapshot=target_snapshot,
    )


def _rename(source: Path, destination: Path) -> None:
    """Publish a sibling role without replacing even a late-created destination."""
    if source.parent != destination.parent or not source.name or not destination.name:
        raise MigrationError("migration publication requires sibling roles")
    if "\x00" in str(source) or "\x00" in str(destination):
        raise MigrationError("migration publication path contains a NUL character")
    if sys.platform == "darwin":
        symbol, flag = "renameatx_np", 4  # RENAME_EXCL
    elif sys.platform == "linux":
        symbol, flag = "renameat2", 1  # RENAME_NOREPLACE
    else:
        raise MigrationError("atomic no-replace publication is unsupported on this platform")
    try:
        native = getattr(ctypes.CDLL(None, use_errno=True), symbol)
    except (AttributeError, OSError) as exc:
        raise MigrationError("atomic no-replace publication is unavailable") from exc
    native.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    native.restype = ctypes.c_int
    parent_fd = os.open(source.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        result = native(
            parent_fd, os.fsencode(source.name), parent_fd, os.fsencode(destination.name), flag
        )
        if result != 0:
            error = ctypes.get_errno()
            if error in {errno.ENOSYS, errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP}:
                raise MigrationError(
                    "atomic no-replace publication is unsupported by this filesystem"
                )
            raise OSError(error, os.strerror(error))
    finally:
        os.close(parent_fd)


def _rename_to_absent(source: Path, destination: Path, role: str) -> None:
    if _role_exists(destination, f"{role} role"):
        raise MigrationError(f"{role} role became occupied during migration publication")
    _rename(source, destination)


def _clear_marker(target: Path) -> None:
    try:
        (target / MARKER_NAME).unlink(missing_ok=True)
    except OSError as exc:
        raise MigrationError("cannot clear the migration marker") from exc


def _role_is_directory(path: Path, role: str) -> bool:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise MigrationError(f"cannot inspect the {role} role") from exc
    return stat.S_ISDIR(mode)


def _publish(prepared: PreparedBundleMigration) -> None:
    _assert_snapshot(prepared.source_dir, prepared.source_snapshot, "source")
    _assert_snapshot(prepared.staging_dir, prepared.target_snapshot, "target")
    try:
        _rename_to_absent(prepared.source_dir, prepared.backup_dir, "backup")
    except OSError as exc:
        raise MigrationError("cannot move the source role into the backup role") from exc
    try:
        _rename_to_absent(prepared.staging_dir, prepared.source_dir, "source")
    except (OSError, MigrationError) as exc:
        try:
            _rename_to_absent(prepared.backup_dir, prepared.source_dir, "source")
        except (OSError, MigrationError) as restore_exc:
            raise MigrationError(
                "migration publication failed and source restoration also failed"
            ) from restore_exc
        raise MigrationError("cannot publish migrated bundle; original was restored") from exc

    _assert_snapshot(prepared.source_dir, prepared.target_snapshot, "target")
    _, published_snapshot = _validate_prepared_target(
        prepared.source_dir,
        legacy_source=prepared.backup_dir,
        source_snapshot=prepared.source_snapshot,
        expected_bundle_id=prepared.bundle_id,
        source_sha256=prepared.source_sha256,
        source_bytes=prepared.source_bytes,
    )
    if published_snapshot != prepared.target_snapshot:
        raise MigrationError("target tree changed during migration publication")
    _assert_snapshot(prepared.source_dir, prepared.target_snapshot, "target")
    _clear_marker(prepared.source_dir)


def _current_version(bundle: Path, role: str) -> str:
    value = _read_object(bundle / MANIFEST_NAME, f"{role} manifest").get("schema_version")
    if not isinstance(value, str):
        raise MigrationError(f"{role} manifest schema version is malformed")
    return value


def _validate_legacy(
    bundle: Path, *, role: str, expected_bundle_id: str | None = None
) -> tuple[str, str, int]:
    _assert_no_symlinks(bundle, role)
    if _role_exists(bundle / MARKER_NAME, f"{role} migration marker"):
        raise MigrationError(f"{role} role contains a migration marker")
    raw = _read_object(bundle / MANIFEST_NAME, f"{role} manifest")
    if raw.get("schema_version") != LEGACY_SCHEMA_VERSION:
        raise MigrationError(f"{role} role does not declare schema 0.1.0")
    bundle_id, manifest_source = _manifest_fields(raw)
    if expected_bundle_id is not None and bundle_id != expected_bundle_id:
        raise MigrationError("backup and source bundle IDs disagree")
    _assert_declared_integrity(bundle, raw)
    source_sha256, source_bytes = _source_identity(bundle, manifest_source)
    return bundle_id, source_sha256, source_bytes


def _validate_current(bundle: Path) -> tuple[Manifest, str, int]:
    raw = _read_object(bundle / MANIFEST_NAME, "current manifest")
    bundle_id, manifest_source = _manifest_fields(raw)
    _assert_declared_integrity(bundle, raw)
    source_sha256, source_bytes = _source_identity(bundle, manifest_source)
    manifest = _validate_target(
        bundle,
        expected_bundle_id=bundle_id,
        source_sha256=source_sha256,
        source_bytes=source_bytes,
        schema_version=raw["schema_version"],
    )
    return manifest, source_sha256, source_bytes


def _prepared_from_existing_staging(
    source: Path,
    staging: Path,
    backup: Path,
    *,
    bundle_id: str,
    source_sha256: str,
    source_bytes: int,
) -> PreparedBundleMigration:
    if not _role_is_directory(staging, "staging") or not _marker_matches(staging, bundle_id):
        raise MigrationError("staging role is ambiguous")
    source_snapshot = _tree_snapshot(source, "source")
    _, target_snapshot = _validate_prepared_target(
        staging,
        legacy_source=source,
        source_snapshot=source_snapshot,
        expected_bundle_id=bundle_id,
        source_sha256=source_sha256,
        source_bytes=source_bytes,
    )
    return PreparedBundleMigration(
        bundle_id=bundle_id,
        source_version=LEGACY_SCHEMA_VERSION,
        target_version=TARGET_SCHEMA_VERSION,
        source_sha256=source_sha256,
        source_bytes=source_bytes,
        source_dir=source,
        staging_dir=staging,
        backup_dir=backup,
        source_snapshot=source_snapshot,
        target_snapshot=target_snapshot,
    )


def migrate_bundle(bundle_dir: Path) -> BundleMigrationResult:
    """Migrate one legacy bundle in place or recover its deterministic roles."""
    source = _source_path(bundle_dir)
    staging = source.with_name(source.name + STAGING_SUFFIX)
    backup = source.with_name(source.name + BACKUP_SUFFIX)

    if _role_exists(source, "source role"):
        if not _role_is_directory(source, "source"):
            raise MigrationError("source role is occupied by a non-directory")
        _assert_no_symlinks(source, "source")
        version = _current_version(source, "source")
        if schema_version_is_compatible(version):
            if _role_exists(staging, "staging role"):
                raise MigrationError("staging role is occupied beside a current source role")
            manifest, current_sha256, current_bytes = _validate_current(source)
            marker_exists = _role_exists(source / MARKER_NAME, "source migration marker")
            backup_exists = _role_exists(backup, "backup role")
            if marker_exists and (
                not backup_exists
                or not _role_is_directory(backup, "backup")
                or not _marker_matches(source, manifest.bundle_id)
            ):
                raise MigrationError("current source migration state is ambiguous")
            if backup_exists:
                if not _role_is_directory(backup, "backup"):
                    raise MigrationError("backup role is occupied by a non-directory")
                backup_id, backup_sha256, backup_bytes = _validate_legacy(
                    backup,
                    role="backup",
                    expected_bundle_id=manifest.bundle_id,
                )
                if (backup_sha256, backup_bytes) != (current_sha256, current_bytes):
                    raise MigrationError("backup source identity differs from current target")
                if marker_exists:
                    backup_snapshot = _tree_snapshot(backup, "backup")
                    _validate_prepared_target(
                        source,
                        legacy_source=backup,
                        source_snapshot=backup_snapshot,
                        expected_bundle_id=backup_id,
                        source_sha256=backup_sha256,
                        source_bytes=backup_bytes,
                    )
            if marker_exists:
                _clear_marker(source)
            return BundleMigrationResult(
                bundle_id=manifest.bundle_id,
                source_version=version,
                target_version=version,
                outcome="recovered" if marker_exists else "already_current",
                backup_retained=backup_exists,
            )
        if version != LEGACY_SCHEMA_VERSION:
            raise MigrationError("source role declares an unsupported schema version")

        bundle_id, source_sha256, source_bytes = _validate_legacy(source, role="source")
        if _role_exists(backup, "backup role"):
            raise MigrationError("backup role is already occupied")
        if _role_exists(staging, "staging role"):
            if not _role_is_directory(staging, "staging") or not _marker_matches(
                staging, bundle_id
            ):
                raise MigrationError("staging role is ambiguous")
            try:
                prepared = _prepared_from_existing_staging(
                    source,
                    staging,
                    backup,
                    bundle_id=bundle_id,
                    source_sha256=source_sha256,
                    source_bytes=source_bytes,
                )
            except MigrationError:
                _remove_owned_staging(staging, bundle_id)
                prepared = prepare_bundle_migration(source)
            _publish(prepared)
            return BundleMigrationResult(
                bundle_id=bundle_id,
                source_version=LEGACY_SCHEMA_VERSION,
                target_version=TARGET_SCHEMA_VERSION,
                outcome="recovered",
                backup_retained=True,
            )

        prepared = prepare_bundle_migration(source)
        _publish(prepared)
        return BundleMigrationResult(
            bundle_id=prepared.bundle_id,
            source_version=prepared.source_version,
            target_version=prepared.target_version,
            outcome="migrated",
            backup_retained=True,
        )

    if not _role_exists(backup, "backup role"):
        raise MigrationError("source role is missing and no backup role can restore it")
    if not _role_is_directory(backup, "backup"):
        raise MigrationError("backup role is occupied by a non-directory")

    bundle_id, source_sha256, source_bytes = _validate_legacy(backup, role="backup")
    backup_snapshot = _tree_snapshot(backup, "backup")
    staging_exists = _role_exists(staging, "staging role")
    if staging_exists:
        try:
            if not _role_is_directory(staging, "staging") or not _marker_matches(
                staging, bundle_id
            ):
                raise MigrationError("staging role is ambiguous")
            _, target_snapshot = _validate_prepared_target(
                staging,
                legacy_source=backup,
                source_snapshot=backup_snapshot,
                expected_bundle_id=bundle_id,
                source_sha256=source_sha256,
                source_bytes=source_bytes,
            )
        except MigrationError:
            pass
        else:
            _assert_snapshot(backup, backup_snapshot, "backup")
            _assert_snapshot(staging, target_snapshot, "target")
            try:
                _rename_to_absent(staging, source, "source")
            except (OSError, MigrationError) as exc:
                raise MigrationError("cannot finish publication from the staging role") from exc
            _assert_snapshot(source, target_snapshot, "target")
            _, published_snapshot = _validate_prepared_target(
                source,
                legacy_source=backup,
                source_snapshot=backup_snapshot,
                expected_bundle_id=bundle_id,
                source_sha256=source_sha256,
                source_bytes=source_bytes,
            )
            if published_snapshot != target_snapshot:
                raise MigrationError("target tree changed during recovery publication")
            _assert_snapshot(source, target_snapshot, "target")
            _clear_marker(source)
            return BundleMigrationResult(
                bundle_id=bundle_id,
                source_version=LEGACY_SCHEMA_VERSION,
                target_version=TARGET_SCHEMA_VERSION,
                outcome="recovered",
                backup_retained=True,
            )

    _assert_snapshot(backup, backup_snapshot, "backup")
    try:
        _rename_to_absent(backup, source, "source")
    except (OSError, MigrationError) as exc:
        raise MigrationError("cannot restore the source role from backup") from exc
    _assert_snapshot(source, backup_snapshot, "source")
    if staging_exists:
        raise MigrationError("restored the original bundle; staging role remains unusable")
    raise MigrationError("restored the original bundle after an incomplete migration")
