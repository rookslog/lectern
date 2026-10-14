"""Every written artifact type has a schema, and every schema matches what is written.

The Tier-A clause is "artifact schemas export and round-trip under the recorded
promise level". Two failure modes sit under that, and only one of them is
obvious.

The obvious one: a schema drifts from its model. The committed-file check
catches that.

The one worth building for: an artifact type is written with no schema at all.
Checking the schemas that exist can never find that, because the missing schema
is missing from the list being checked too. So the enumeration is derived from
what an ingest actually produces on disk, not from a list someone maintains.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest

from lectern import cli
from lectern.bundle import (
    ARTIFACT_MODELS,
    ArtifactRef,
    Manifest,
    NormalizedAudio,
    Source,
    SourceDocument,
    SourceKind,
    StageName,
    TranscriptArtifacts,
    TranscriptBackend,
    TranscriptMetadataDocument,
    TranscriptSegmentsDocument,
    export_artifact_schemas,
)

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
SYNTHETIC_TALK = FIXTURE_DIR / "synthetic_talk.wav"
SYNTHETIC_TRANSCRIPT = FIXTURE_DIR / "synthetic_talk.transcript.txt"
SCHEMA_DIR = Path(__file__).resolve().parent.parent / "schemas"

# Bundle-relative JSON artifacts, and the model that describes each. `media/` and
# the Markdown renderings are not JSON and carry no schema.
ARTIFACT_FILES = {
    "manifest.json": "manifest",
    "source.json": "source",
    "transcript/segments.json": "transcript-segments",
    "transcript/metadata.json": "transcript-metadata",
}


def _bundle(tmp_path: Path) -> Path:
    media_dir = tmp_path / "media"
    media_dir.mkdir(parents=True, exist_ok=True)
    media = media_dir / "synthetic_talk.wav"
    media.write_bytes(SYNTHETIC_TALK.read_bytes())
    media.with_suffix(".transcript.txt").write_text(
        SYNTHETIC_TRANSCRIPT.read_text(encoding="utf-8"), encoding="utf-8"
    )
    out = tmp_path / "bundles"
    assert (
        cli.main(["ingest", str(media), "--output", str(out), "--state", str(tmp_path / "s.db")])
        == 0
    )
    return sorted(path for path in out.iterdir() if path.is_dir())[-1]


def test_every_written_json_artifact_has_a_schema(tmp_path: Path) -> None:
    """Derived from disk, so a new unschema'd artifact fails here.

    A list maintained by hand cannot catch its own omissions; walking the
    produced bundle can.
    """

    bundle = _bundle(tmp_path)
    written = {
        path.relative_to(bundle).as_posix()
        for path in bundle.rglob("*.json")
        if path.is_file() and not path.name.startswith(".")
    }
    assert written == set(ARTIFACT_FILES), (
        "a JSON artifact is written that no schema describes, or vice versa; "
        f"written={sorted(written)} described={sorted(ARTIFACT_FILES)}"
    )
    # Visual/speaker JSON is opt-in and exercised by their acceptance tests; the default bundle
    # must still contain exactly the unconditional artifacts above.
    assert set(ARTIFACT_FILES.values()) | {"frames", "diarization"} == set(ARTIFACT_MODELS)


def test_manifest_round_trips(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    loaded = Manifest.load(bundle)
    reparsed = Manifest.model_validate_json(loaded.model_dump_json())
    assert reparsed == loaded


def test_source_document_round_trips(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    payload: dict[str, Any] = json.loads((bundle / "source.json").read_text(encoding="utf-8"))
    document = SourceDocument.model_validate(payload)
    assert SourceDocument.model_validate_json(document.model_dump_json()) == document


def test_transcript_segments_round_trip(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    # Validated as the whole ARTIFACT, not element by element. Round-tripping
    # elements exercises the model and never the file, so it could not have
    # noticed that the exported schema declared an object root for an array.
    raw = (bundle / "transcript" / "segments.json").read_text(encoding="utf-8")
    document = TranscriptSegmentsDocument.model_validate_json(raw)
    assert document.root
    assert TranscriptSegmentsDocument.model_validate_json(document.model_dump_json()) == document


def test_transcript_metadata_round_trips(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    payload: dict[str, Any] = json.loads(
        (bundle / "transcript" / "metadata.json").read_text(encoding="utf-8")
    )
    document = TranscriptMetadataDocument.model_validate(payload)
    assert (
        TranscriptMetadataDocument.model_validate_json(document.model_dump_json(by_alias=True))
        == document
    )


def test_transcript_backend_rejects_undeclared_path_fields() -> None:
    with pytest.raises(ValueError):
        TranscriptBackend.model_validate({"kind": "sidecar", "path": "/private/transcript.txt"})

    definitions = json.loads(export_artifact_schemas()["transcript-metadata"])["$defs"]
    assert definitions["TranscriptBackend"]["additionalProperties"] is False


@pytest.mark.parametrize("path", ["/Users/alice/audio.wav", "../outside.wav", "a/../b.wav"])
def test_artifact_paths_must_be_normalized_bundle_relative(path: str) -> None:
    with pytest.raises(ValueError):
        ArtifactRef(path=path, sha256="0" * 64, bytes=1)
    with pytest.raises(ValueError):
        NormalizedAudio(path=path, sha256="0" * 64, bytes=1)
    with pytest.raises(ValueError):
        TranscriptArtifacts(segments=path, transcript="transcript/a.md", summary="analysis/a.md")


def test_source_identity_rejects_undeclared_path_fields() -> None:
    with pytest.raises(ValueError):
        Source.model_validate(
            {"kind": SourceKind.LOCAL, "ref": f"sha256:{'0' * 64}", "bytes": 1, "path": "/x"}
        )

    with pytest.raises(ValueError):
        SourceDocument.model_validate(
            {
                "source": {"kind": "local", "ref": f"sha256:{'0' * 64}", "bytes": 1},
                "sha256": "0" * 64,
                "bytes": 1,
                "transcript": {
                    "method": "fixture",
                    "metadata": "transcript/metadata.json",
                    "segments": "transcript/segments.json",
                    "transcript": "transcript/transcript.md",
                    "evidence_limit": "fixture",
                    "remote_services": {
                        "allowed": False,
                        "scope": "core",
                        "lectern_invoked": False,
                        "requires_explicit_per_item_consent": True,
                        "transcriber_network_posture": "none",
                    },
                },
                "path": "/x",
            }
        )


def test_segments_document_rejects_duplicate_ids() -> None:
    row = {"id": 0, "start_s": 0.0, "text": "one", "source": "fixture"}
    with pytest.raises(ValueError, match="unique"):
        TranscriptSegmentsDocument.model_validate([row, {**row, "text": "two"}])
    schema = json.loads(export_artifact_schemas()["transcript-segments"])
    assert schema["x-lectern-uniqueBy"] == "id"


@pytest.mark.parametrize("timestamp", [float("nan"), float("inf"), float("-inf")])
def test_segments_document_rejects_non_finite_timestamps(timestamp: float) -> None:
    row = {"id": 0, "start_s": timestamp, "text": "one", "source": "fixture"}
    with pytest.raises(ValueError, match="finite"):
        TranscriptSegmentsDocument.model_validate([row])


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_artifact_models_reject_non_finite_duration_and_timeout(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        Source(kind=SourceKind.URL, ref="https://example.test/talk", duration_s=value)
    with pytest.raises(ValueError, match="finite"):
        TranscriptBackend(kind="fixture", timeout_s=value)


def test_artifact_identity_fields_are_constrained() -> None:
    with pytest.raises(ValueError):
        ArtifactRef(path="a", sha256="bad", bytes=1)
    with pytest.raises(ValueError):
        ArtifactRef(path="a", sha256="0" * 64, bytes=-1)


@pytest.mark.parametrize(("value", "expected"), [(-1, -1), ("7", 7), (True, 1)])
def test_non_local_source_byte_values_retain_legacy_coercion(value: object, expected: int) -> None:
    source = Source.model_validate(
        {"kind": SourceKind.URL, "ref": "https://example.test/talk", "bytes": value}
    )
    assert source.bytes == expected


def test_manifest_schema_requires_the_version_runtime_load_requires() -> None:
    schema = json.loads(export_artifact_schemas()["manifest"])
    assert "schema_version" in schema["required"]


@pytest.mark.parametrize(
    "path",
    ["a//b", "./a", "a/.", "a/", "a\nb\\c", "a\n/../b"],
)
def test_artifact_path_schema_rejects_runtime_noncanonical_paths(path: str) -> None:
    with pytest.raises(ValueError):
        ArtifactRef(path=path, sha256="0" * 64, bytes=1)
    schema = json.loads(export_artifact_schemas()["manifest"])
    pattern = schema["$defs"]["ArtifactRef"]["properties"]["path"]["pattern"]
    assert re.fullmatch(pattern, path) is None


def test_stage_error_redaction_survives_assignment_and_manifest_save(tmp_path: Path) -> None:
    manifest = Manifest(
        bundle_id="fixture",
        source=Source(kind=SourceKind.URL, ref="https://example.test/talk"),
    )
    stage = manifest.stages[StageName.ACQUIRE]
    stage.error = "failed at /Users/alice/private/session.wav"

    assert stage.error == "failed at <path>"
    payload = json.loads(manifest.save(tmp_path).read_text(encoding="utf-8"))
    assert payload["stages"][StageName.ACQUIRE]["error"] == "failed at <path>"


@pytest.mark.parametrize("name", sorted(ARTIFACT_MODELS))
def test_committed_schema_matches_its_model(name: str) -> None:
    path = SCHEMA_DIR / f"{name}.schema.json"
    assert path.is_file(), f"missing committed schema: {path.name}"
    assert path.read_text(encoding="utf-8") == export_artifact_schemas()[name]


def test_local_source_schema_requires_content_identity_and_size() -> None:
    for name in ("manifest", "source"):
        definitions = json.loads(export_artifact_schemas()[name])["$defs"]
        local_rule = definitions["Source"]["allOf"][0]
        assert local_rule["if"] == {"properties": {"kind": {"const": "local"}}}
        assert local_rule["then"]["required"] == ["bytes"]
        assert local_rule["then"]["properties"]["ref"]["pattern"] == (r"^sha256:[0-9a-f]{64}$")
        assert local_rule["then"]["properties"]["bytes"] == {
            "minimum": 0,
            "type": "integer",
        }
        shared_bytes = definitions["Source"]["properties"]["bytes"]
        assert shared_bytes["anyOf"][0] == {"type": "integer"}


def test_no_artifact_schema_admits_a_filesystem_path() -> None:
    """The promise LW-11 made, enforced at the level of the contract.

    A schema that still declares a `path` on a source or sidecar leaves the door
    open for one to be written again, and a future reader would take its
    presence as permission.
    """

    for name, schema in export_artifact_schemas().items():
        definitions = json.loads(schema).get("$defs", {})
        for model_name in ("ContentIdentity",):
            model_schema: dict[str, Any] = definitions.get(model_name, {})
            assert "path" not in model_schema.get("properties", {}), (
                f"{name}: {model_name} still declares a path field"
            )


@pytest.mark.parametrize("text", ["", " \t\r\n", "\u00a0\u2003\u3000"])
def test_segments_reject_blank_text_in_model_and_exported_schema(text: str) -> None:
    row = {"id": 0, "start_s": 0.0, "text": text, "source": "fixture"}
    with pytest.raises(ValueError, match="pattern"):
        TranscriptSegmentsDocument.model_validate([row])
    schema = json.loads(export_artifact_schemas()["transcript-segments"])
    pattern = schema["$defs"]["TranscriptSegmentRecord"]["properties"]["text"]["pattern"]
    assert re.search(pattern, text) is None


def test_segment_text_whitespace_boundary_and_exact_preservation() -> None:
    schema = json.loads(export_artifact_schemas()["transcript-segments"])
    pattern = schema["$defs"]["TranscriptSegmentRecord"]["properties"]["text"]["pattern"]
    whitespace = [chr(code) for code in range(sys.maxunicode + 1) if chr(code).isspace()]
    for text in whitespace + ["".join(whitespace)]:
        row = {"id": 7, "start_s": 1.0, "text": text, "source": "fixture"}
        with pytest.raises(ValueError):
            TranscriptSegmentsDocument.model_validate([row])
        assert re.search(pattern, text) is None, repr(text)
    for text in [
        " \tעברית\nالعربية 日本語 🙂\u3000",
        "\u200b",
        "\u200c",
        "\u200d",
        "\u2060",
        "\ufeff",
    ]:
        assert text.strip()
        row = {"id": 7, "start_s": 1.0, "text": text, "source": "fixture"}
        document = TranscriptSegmentsDocument.model_validate([row])
        assert document.model_dump()[0]["text"] == text
        assert json.loads(document.model_dump_json())[0]["text"] == text
        assert re.search(pattern, text) is not None, repr(text)


@pytest.mark.parametrize("path", ["\x00file", "dir/fi\x00le", "dir/file\x00"])
def test_artifact_path_rejects_nul_in_runtime_and_schema(path: str) -> None:
    with pytest.raises(ValueError):
        ArtifactRef(path=path, sha256="0" * 64, bytes=0)
    schema = json.loads(export_artifact_schemas()["manifest"])
    pattern = schema["$defs"]["ArtifactRef"]["properties"]["path"]["pattern"]
    assert re.search(pattern, path) is None


def test_artifact_path_preserves_ordinary_unicode() -> None:
    path = "資料/עברית-évidence.json"
    artifact = ArtifactRef(path=path, sha256="0" * 64, bytes=0)
    assert artifact.path == path
    schema = json.loads(export_artifact_schemas()["manifest"])
    pattern = schema["$defs"]["ArtifactRef"]["properties"]["path"]["pattern"]
    assert re.search(pattern, path) is not None
