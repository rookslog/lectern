"""Bundle schema seed tests: round-trip and schema export (M0 acceptance basis)."""

import json
import os
import stat
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError
from pytest import MonkeyPatch

from lectern.bundle import (
    MANIFEST_NAME,
    SCHEMA_VERSION,
    Manifest,
    Source,
    SourceKind,
    StageName,
    StageRecord,
    StageState,
    atomic_write_text,
    export_json_schema,
    schema_version_is_compatible,
)

CONTENT_DIGEST = "a" * 64


def make_manifest() -> Manifest:
    return Manifest(
        bundle_id="test-0001",
        source=Source(
            kind=SourceKind.LOCAL,
            ref=f"sha256:{CONTENT_DIGEST}",
            bytes=12,
            title="Fixture Talk",
        ),
    )


def test_manifest_schema_is_one_one() -> None:
    assert SCHEMA_VERSION == "1.1.0"


@pytest.mark.parametrize("version", ["0.1.0", "1.0.1", "1.2.0", "2.0.0", "1", "garbage"])
def test_manifest_schema_compatibility_requires_a_supported_version(version: str) -> None:
    assert not schema_version_is_compatible(version)


def test_manifest_schema_compatibility_accepts_the_current_version() -> None:
    assert schema_version_is_compatible(SCHEMA_VERSION)


def test_manifest_load_refuses_a_later_minor_before_artifact_validation(tmp_path: Path) -> None:
    manifest = make_manifest()
    path = manifest.save(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema_version"] = "1.2.0"
    payload["optional_1_2_field"] = "unknown to the 1.1.0 model"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported bundle schema version '1.2.0'"):
        Manifest.load(tmp_path)


def test_local_source_requires_content_identity_and_size() -> None:
    with pytest.raises(ValidationError, match="content identity"):
        Source(kind=SourceKind.LOCAL, ref="/tmp/talk.wav", bytes=12)
    with pytest.raises(ValidationError, match="source byte size"):
        Source(kind=SourceKind.LOCAL, ref=f"sha256:{CONTENT_DIGEST}")


def test_local_source_accepts_content_identity() -> None:
    source = Source(kind=SourceKind.LOCAL, ref=f"sha256:{CONTENT_DIGEST}", bytes=12)
    assert source.ref == f"sha256:{CONTENT_DIGEST}"


@pytest.mark.parametrize(
    "published",
    [
        "2026-10-23T12:34:56.125z",
        "2026-10-23T12:34:56.125Z",
        "2026-10-23T12:34:56.125+00:00",
        "2026-10-23T15:04:56.125+02:30",
        "2026-10-23t12:34:56.125z",
    ],
)
def test_published_accepts_rfc3339_zone_forms(published: str) -> None:
    payload = {"kind": "url", "ref": "synthetic-date-control", "published": published}
    source = Source.model_validate_json(json.dumps(payload), strict=True)
    assert source.published == datetime(2026, 10, 23, 12, 34, 56, 125000, tzinfo=UTC)
    assert Source.model_validate_json(source.model_dump_json(), strict=True) == source


@pytest.mark.parametrize(
    "published",
    [
        "2026-10-23",
        "2026-10-23 12:34:56Z",
        "2026-10-23T12:34:56",
        "2026-02-30T12:34:56z",
        "2026-10-23T24:00:00Z",
        "2026-10-23T12:34:56+25:00",
        "2026-10-23T12:34:56zz",
    ],
)
def test_published_rejects_malformed_rfc3339_dates(published: str) -> None:
    payload = {"kind": "url", "ref": "synthetic-date-control", "published": published}
    with pytest.raises(ValidationError):
        Source.model_validate_json(json.dumps(payload), strict=True)


def test_local_source_rejects_a_negative_size() -> None:
    with pytest.raises(ValidationError, match="source byte size"):
        Source(kind=SourceKind.LOCAL, ref=f"sha256:{CONTENT_DIGEST}", bytes=-1)


@pytest.mark.parametrize("raw_bytes", ["12", True])
def test_local_source_rejects_coercive_byte_values(raw_bytes: object) -> None:
    with pytest.raises(ValidationError, match="source byte size"):
        Source.model_validate(
            {
                "kind": SourceKind.LOCAL,
                "ref": f"sha256:{CONTENT_DIGEST}",
                "bytes": raw_bytes,
            }
        )


@pytest.mark.parametrize("kind", [SourceKind.YOUTUBE, SourceKind.URL])
def test_non_local_source_preserves_negative_bytes(kind: SourceKind) -> None:
    source = Source(kind=kind, ref="remote-id", bytes=-1)
    assert source.bytes == -1


@pytest.mark.parametrize(("raw_bytes", "expected"), [("12", 12), (True, 1)])
@pytest.mark.parametrize("kind", [SourceKind.YOUTUBE, SourceKind.URL])
def test_non_local_source_preserves_coercive_bytes(
    kind: SourceKind, raw_bytes: object, expected: int
) -> None:
    source = Source.model_validate({"kind": kind, "ref": "remote-id", "bytes": raw_bytes})
    assert source.bytes == expected


def test_manifest_round_trip(tmp_path: Path) -> None:
    m = make_manifest()
    m.stages[StageName.NORMALIZE] = StageRecord(state=StageState.DONE)
    m.save(tmp_path)
    loaded = Manifest.load(tmp_path)
    assert loaded == m
    assert loaded.schema_version == SCHEMA_VERSION


def test_manifest_save_leaves_no_temporary_files(tmp_path: Path) -> None:
    m = make_manifest()
    m.save(tmp_path)
    m.save(tmp_path)

    assert sorted(path.name for path in tmp_path.iterdir()) == [MANIFEST_NAME]


def test_interrupted_manifest_save_keeps_previous_manifest(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    m = make_manifest()
    m.save(tmp_path)
    original = (tmp_path / MANIFEST_NAME).read_bytes()

    def fail_replace(source: object, destination: object) -> None:
        raise OSError("synthetic manifest publish failure")

    m.stages[StageName.NORMALIZE] = StageRecord(state=StageState.DONE)
    monkeypatch.setattr(os, "replace", fail_replace)

    with pytest.raises(OSError, match="synthetic manifest publish failure"):
        m.save(tmp_path)

    monkeypatch.undo()

    # A crash mid-publish must not leave a committed bundle pointing at a
    # truncated manifest: the previous bytes stay in place and stay parseable.
    assert (tmp_path / MANIFEST_NAME).read_bytes() == original
    assert Manifest.load(tmp_path).stages[StageName.NORMALIZE].state is StageState.PENDING
    assert sorted(path.name for path in tmp_path.iterdir()) == [MANIFEST_NAME]


def test_all_stages_present_by_default() -> None:
    m = make_manifest()
    assert set(m.stages) == set(StageName)
    assert all(r.state is StageState.PENDING for r in m.stages.values())


def test_json_schema_exports() -> None:
    schema = export_json_schema()
    assert '"Manifest"' in schema or '"title": "Manifest"' in schema


def test_committed_json_schema_matches_model() -> None:
    schema_path = Path(__file__).resolve().parent.parent / "schemas" / "manifest.schema.json"
    assert schema_path.read_text() == export_json_schema()


def test_atomic_write_temporary_is_restricted_before_content(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    target = tmp_path / "manifest.json"
    target.write_text("{}\n", encoding="utf-8")
    os.chmod(target, 0o600)
    previous_umask = os.umask(0o022)
    observed: list[int] = []
    real_fdopen = os.fdopen

    def spy_fdopen(fd: int, *args: object, **kwargs: object) -> object:
        # Sample the temp file's mode before any content bytes are written.
        observed.append(stat.S_IMODE(os.fstat(fd).st_mode))
        return real_fdopen(fd, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(os, "fdopen", spy_fdopen)
    try:
        atomic_write_text(target, '{"a": 1}\n')
    finally:
        os.umask(previous_umask)

    assert observed == [0o600]
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert target.read_text(encoding="utf-8") == '{"a": 1}\n'


@pytest.mark.parametrize("raw_bytes", [12, 12.0])
def test_manifest_load_preserves_integral_local_byte_compatibility(
    tmp_path: Path, raw_bytes: int | float
) -> None:
    path = make_manifest().save(tmp_path)
    payload = json.loads(path.read_text())
    payload["source"]["bytes"] = raw_bytes
    path.write_text(json.dumps(payload))
    original = path.read_bytes()
    assert Manifest.load(tmp_path).source.bytes == 12
    assert path.read_bytes() == original


@pytest.mark.parametrize("raw_bytes", [12.5, "12", True])
def test_manifest_load_refuses_fractional_or_coercive_local_bytes(
    tmp_path: Path, raw_bytes: object
) -> None:
    path = make_manifest().save(tmp_path)
    payload = json.loads(path.read_text())
    payload["source"]["bytes"] = raw_bytes
    path.write_text(json.dumps(payload))
    with pytest.raises(ValidationError):
        Manifest.load(tmp_path)
