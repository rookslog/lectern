"""The bundle schema: the durable contract every consumer reads (ADR-0003).

Stages communicate only through bundle artifacts described by these models.
Breaking schema changes should be deliberate and versioned.
"""

from __future__ import annotations

import json
import os
import re
import stat
import uuid
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Annotated, Literal, cast

from pydantic import (
    AfterValidator,
    BaseModel,
    Field,
    FiniteFloat,
    RootModel,
    StringConstraints,
    WithJsonSchema,
    field_validator,
    model_validator,
)

from lectern.records import redact_paths

SCHEMA_VERSION = "1.1.0"
_RFC3339 = re.compile(r"\d{4}-\d{2}-\d{2}[Tt]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:[Zz]|[+-]\d{2}:\d{2})")
CONTENT_REF_PATTERN = r"^sha256:[0-9a-f]{64}$"
_CONTENT_REF = re.compile(CONTENT_REF_PATTERN)

MANIFEST_NAME = "manifest.json"
BUNDLE_RELATIVE_PATH_PATTERN = (
    r"^(?![A-Za-z]:)(?![\s\S]*\u0000)(?![\s\S]*\\)(?![\s\S]*(?:^|/)\.\.?(?:/|$))[^/]+(?:/[^/]+)*$"
)


def _validate_bundle_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or "\x00" in value
        or "\\" in value
        or path.is_absolute()
        or PureWindowsPath(value).drive
        or path.as_posix() != value
        or any(part in {".", ".."} for part in path.parts)
    ):
        raise ValueError("artifact path must be normalized and bundle-relative")
    return value


BundleRelativePath = Annotated[
    str,
    AfterValidator(_validate_bundle_relative_path),
    WithJsonSchema({"type": "string", "pattern": BUNDLE_RELATIVE_PATH_PATTERN}),
]
Sha256Digest = Annotated[str, StringConstraints(strict=True, pattern=r"^[0-9a-f]{64}$")]
NonNegativeStrictInt = Annotated[int, Field(strict=True, ge=0)]
SegmentId = Annotated[int, Field(strict=True, ge=0, le=(2**63) - 1)]


class ArtifactModel(BaseModel):
    model_config = {"extra": "forbid"}


def _require_manifest_schema_version(schema: dict[str, object]) -> None:
    required = list(cast(list[str], schema.get("required", [])))
    if "schema_version" not in required:
        required.insert(0, "schema_version")
    schema["required"] = required


def schema_version_is_compatible(version: str) -> bool:
    """Whether this build can read a manifest declaring `version`.

    Schema 1.1.0 only adds optional fields, so 1.0.0 remains readable without
    migration. All other versions, including later same-major schemas, are
    refused. Artifact models still reject unknown fields rather than silently
    discarding potentially privacy-sensitive data.
    """

    return version in {"1.0.0", SCHEMA_VERSION}


def atomic_write_text(path: Path, text: str) -> Path:
    """Publish ``text`` at ``path`` via a same-directory temporary file.

    A plain write truncates the destination before the new bytes land, so a
    crash mid-write can leave a committed bundle reference pointing at a
    corrupt artifact. Writing to a unique temporary name and publishing with
    ``os.replace`` makes the swap atomic: readers see either the previous file
    or the new one, never a partial one. The temporary is removed if anything
    fails before the replace.
    """

    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    # Preserve the destination's permissions across the replace, and restrict
    # the temporary BEFORE any content bytes land in it: a restricted artifact
    # must never transit through a umask-mode file another local user could
    # read, and a killed process must not leave a broad-mode copy behind.
    mode = path.stat().st_mode & 0o777 if path.exists() else None
    descriptor = os.open(
        temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, mode if mode is not None else 0o666
    )
    try:
        if mode is not None:
            os.chmod(temporary, mode)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path


class SourceKind(StrEnum):
    YOUTUBE = "youtube"
    LOCAL = "local"
    URL = "url"


class StageName(StrEnum):
    ACQUIRE = "acquire"
    NORMALIZE = "normalize"
    TRANSCRIBE = "transcribe"
    DIARIZE = "diarize"
    VISUAL = "visual"
    ENRICH = "enrich"
    SITUATE = "situate"
    SYNTHESIZE = "synthesize"


class StageState(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


class ExternalSource(ArtifactModel):
    """Allowlisted external identity from local metadata, never download details."""

    provider: str | None = None
    id: str | None = None
    url: str | None = None
    playlist_id: str | None = None
    playlist_title: str | None = None
    playlist_index: NonNegativeStrictInt | None = None
    language_hint: str | None = None


class Source(ArtifactModel):
    """Provenance. `local` sources trigger the privacy hard rule (ADR-0002)."""

    kind: SourceKind
    # A *reference* to the source, never a location on the machine that made
    # the bundle. For remote kinds that is the URL, which is the identity. For
    # `local` it is `sha256:<digest>`: a bundle outlives the filesystem it was
    # built on, so a path in it both discloses the account name and dangles as
    # soon as the bundle is copied (LW-11, ratified 2026-08-01).
    ref: str
    # Size travels with the digest because identity alone cannot tell a caller
    # whether this was a short clip or a long lecture.
    bytes: int | None = None
    title: str | None = None
    channel: str | None = None
    published: datetime | None = None
    duration_s: FiniteFloat | None = None
    external: ExternalSource | None = None

    @field_validator("published", mode="before")
    @classmethod
    def parse_published(cls, value: object) -> object:
        # The model-level before-validator makes strict JSON validation run on
        # parsed Python values, where an ISO-8601 string is not a datetime. A
        # populated `published` (first written by info.json intake) therefore
        # failed every strict evidence read of its bundle.
        # Only RFC 3339 date-times, as the exported schema declares (#93);
        # fromisoformat alone would also accept plain or week dates and naive times.
        if isinstance(value, str):
            if _RFC3339.fullmatch(value) is None:
                raise ValueError("published must be an RFC 3339 date-time")
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        return value

    @model_validator(mode="before")
    @classmethod
    def validate_local_byte_representation(cls, data: object) -> object:
        if isinstance(data, dict):
            values = cast(dict[str, object], data)
            if values.get("kind") in (SourceKind.LOCAL, SourceKind.LOCAL.value):
                size = values.get("bytes")
                if isinstance(size, (str, bool)):
                    raise ValueError("local source byte size must be a non-negative integer")
            return values
        return data

    @model_validator(mode="after")
    def validate_local_identity(self) -> Source:
        if self.kind is SourceKind.LOCAL:
            if _CONTENT_REF.fullmatch(self.ref) is None:
                raise ValueError("local source ref must be a sha256 content identity")
            if self.bytes is None or self.bytes < 0:
                raise ValueError("local source byte size must be a non-negative integer")
        return self

    model_config = {
        "extra": "forbid",
        "json_schema_extra": {
            "allOf": [
                {
                    "if": {"properties": {"kind": {"const": "local"}}},
                    "then": {
                        "properties": {
                            "ref": {"pattern": CONTENT_REF_PATTERN},
                            "bytes": {"type": "integer", "minimum": 0},
                        },
                        "required": ["bytes"],
                    },
                }
            ]
        },
    }


class ArtifactRef(ArtifactModel):
    """A produced file, content-addressed for idempotence checks."""

    path: BundleRelativePath
    sha256: Sha256Digest
    bytes: NonNegativeStrictInt


def empty_artifact_refs() -> list[ArtifactRef]:
    return []


class StageRecord(ArtifactModel):
    state: StageState = StageState.PENDING
    started: datetime | None = None
    finished: datetime | None = None
    outputs: list[ArtifactRef] = Field(default_factory=empty_artifact_refs)
    error: str | None = None
    warnings: list[str] = Field(default_factory=list)

    model_config = {"validate_assignment": True}

    @field_validator("error")
    @classmethod
    def redact_error_paths(cls, value: str | None) -> str | None:
        return None if value is None else redact_paths(value)


class Manifest(ArtifactModel):
    schema_version: str = SCHEMA_VERSION
    bundle_id: str
    created: datetime = Field(default_factory=lambda: datetime.now(UTC))
    source: Source
    stages: dict[StageName, StageRecord] = Field(
        default_factory=lambda: {name: StageRecord() for name in StageName}
    )

    model_config = {"extra": "forbid", "json_schema_extra": _require_manifest_schema_version}

    def save(self, bundle_dir: Path) -> Path:
        # Saving serializes every 1.1.0 field (defaults included), so a manifest
        # loaded as 1.0.0 must not keep claiming 1.0.0 once rewritten (#93).
        self.schema_version = SCHEMA_VERSION
        path = bundle_dir / MANIFEST_NAME
        return atomic_write_text(path, self.model_dump_json(indent=2) + "\n")

    @classmethod
    def load(cls, bundle_dir: Path) -> Manifest:
        """Load a manifest, refusing one this code cannot claim to understand.

        Without this check the version field is decoration: a newer same-shaped
        manifest can validate against the current model, missing fields take their
        defaults, and repurposed fields are read under their old meaning. That
        silence is what makes a same-shaped change dangerous — the consumer gets
        no error, just a wrong answer. Refusing is the only way the compatibility
        promise in SUPPORT.md means anything at the point of use.
        """

        manifest_path = bundle_dir / MANIFEST_NAME
        if bundle_dir.is_symlink() or manifest_path.is_symlink():
            raise ValueError("bundle root and manifest must not be symlinks")
        if not stat.S_ISREG(manifest_path.stat().st_mode):
            raise ValueError("bundle manifest must be a regular file")
        payload: object = json.loads(manifest_path.read_text())
        if not isinstance(payload, dict):
            raise ValueError(
                "bundle manifest must be a JSON object with an explicit schema version"
            )
        version = cast(dict[str, object], payload).get("schema_version")
        if not isinstance(version, str):
            raise ValueError("bundle manifest requires an explicit string schema version")
        if not schema_version_is_compatible(version):
            raise ValueError(
                f"unsupported bundle schema version {version!r}; "
                f"this build reads schema version {SCHEMA_VERSION!r}"
            )
        return cls.model_validate(payload)


class RemoteServices(ArtifactModel):
    """The network posture recorded alongside a transcript.

    Modelled rather than left as free-form JSON because it is the field a
    privacy review reads first, and an unmodelled field is one no schema
    constrains.
    """

    allowed: bool
    scope: str
    lectern_invoked: bool
    requires_explicit_per_item_consent: bool
    transcriber_network_posture: str


class TranscriptPointer(ArtifactModel):
    method: str
    metadata: BundleRelativePath
    segments: BundleRelativePath
    transcript: BundleRelativePath
    evidence_limit: str
    remote_services: RemoteServices


class ContentIdentity(ArtifactModel):
    """Digest and size, with no field for a location.

    The absence is the design: after LW-11 a bundle records what its media was,
    not where it sat, and a schema that still admitted a path would leave the
    door open for one to reappear.
    """

    sha256: Sha256Digest
    bytes: NonNegativeStrictInt | None = None

    # Unknown keys are REJECTED here, and the exported schema says so. Pydantic
    # ignores extras by default, which meant `{"sha256": ..., "path": "/Users/..."}`
    # validated cleanly and the committed schema admitted it -- so the privacy
    # boundary this model exists to state was advisory. Asserting that `path` is
    # not declared was never the same as asserting it is refused.
    model_config = {"extra": "forbid"}


class SourceProvenance(ArtifactModel):
    """The automation record appended to `source.json` on a completed ingest.

    Absent on a bare one-shot ingest and present after queue completion, so it is
    optional here — but declared, because dropping it on round trip would discard
    the state schema version, the source and queue identities, the consent record,
    the policy, and the remote-service posture. Those are exactly the fields a
    later audit would need and would find missing.
    """

    state_schema_version: int
    source_id: str
    source_kind: str
    source_name: str
    source_item_id: str
    queue_item_id: str
    queue_state: str
    policy: str
    # A string naming the basis, e.g. "explicit_queue_approval" — not a
    # structured record. Modelled from the writer rather than assumed.
    consent: str
    remote_services: RemoteServices


class SourceDocument(ArtifactModel):
    """`source.json`."""

    source: Source
    sha256: Sha256Digest
    bytes: NonNegativeStrictInt
    transcript: TranscriptPointer
    transcript_sidecar: ContentIdentity | None = None
    provenance: SourceProvenance | None = None


class TranscriptSegmentRecord(ArtifactModel):
    """One row of `transcript/segments.json`, the unit a citation anchors to."""

    id: SegmentId
    start_s: FiniteFloat
    end_s: FiniteFloat | None = None
    # Explicit Python str.isspace set: \S differs across regex engines. A search
    # for one character outside this class rejects blanks without altering text.
    text: str = Field(
        pattern=r"[^\u0009-\u000d\u001c-\u0020\u0085\u00a0\u1680\u2000-\u200a"
        r"\u2028\u2029\u202f\u205f\u3000]"
    )
    source: str


class NormalizedAudio(ArtifactModel):
    path: BundleRelativePath
    sha256: Sha256Digest
    bytes: NonNegativeStrictInt


class TranscriptBackend(ArtifactModel):
    """How the transcript was produced.

    The `local_command` fields are declared rather than tolerated: pydantic
    ignores unknown keys, so a model that omitted them silently discarded the
    identity of the command that produced the transcript on every round trip,
    and generated-schema consumers were never told the fields exist. Provenance
    that only survives when nobody reads it is not provenance.
    """

    kind: str
    sha256: Sha256Digest | None = None
    command: str | None = None
    argv0: str | None = None
    command_sha256: Sha256Digest | None = None
    argv_sha256: Sha256Digest | None = None
    input_argument_mode: str | None = None
    timeout_s: FiniteFloat | None = None
    sidecar_filename: str | None = None
    model: str | None = None
    library_version: str | None = None
    compute_type: str | None = None
    detected_language: str | None = None
    language_probability: FiniteFloat | None = None

    model_config = {"extra": "forbid"}


class TranscriptArtifacts(ArtifactModel):
    segments: BundleRelativePath
    transcript: BundleRelativePath
    summary: BundleRelativePath


class SchemaContract(ArtifactModel):
    manifest_schema_versioned: bool
    note: str


class TranscriptMetadataDocument(ArtifactModel):
    """`transcript/metadata.json`."""

    schema_: str = Field(alias="schema")
    generated_at: datetime
    method: str
    backend: TranscriptBackend
    language: str | None = None
    remote_services: RemoteServices
    evidence_limit: str
    source_media: ContentIdentity
    normalized_audio: NormalizedAudio
    artifacts: TranscriptArtifacts
    schema_contract: SchemaContract

    model_config = {"populate_by_name": True, "extra": "forbid"}


# Every artifact type Lectern writes as JSON, and the model that describes it.
# Enumerated in one place so "every written artifact type has a schema" is a
# statement a test can check rather than a claim someone has to audit by hand.
class TranscriptSegmentsDocument(RootModel[list[TranscriptSegmentRecord]]):
    """`transcript/segments.json`, which is written as an ARRAY.

    Exporting the element model gave the committed schema an object root, so
    validating a real artifact against it failed immediately. A round-trip test
    over elements cannot catch that: it exercises the model and never the
    artifact.
    """

    root: list[TranscriptSegmentRecord]

    model_config = {"json_schema_extra": {"minItems": 1, "x-lectern-uniqueBy": "id"}}

    @model_validator(mode="after")
    def validate_document(self) -> TranscriptSegmentsDocument:
        if not self.root:
            raise ValueError("transcript segments document requires at least one segment")
        identifiers = [segment.id for segment in self.root]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("transcript segment ids must be unique")
        return self


class FrameRecord(ArtifactModel):
    """One sampled image, with a timestamp relative to the captured video."""

    index: NonNegativeStrictInt
    t_s: Annotated[FiniteFloat, Field(strict=True, ge=0)]
    file: BundleRelativePath
    sha256: Sha256Digest
    reason: Literal["scene", "interval"]
    width: Annotated[int, Field(strict=True, gt=0, le=1280)]
    height: Annotated[int, Field(strict=True, gt=0, le=1280)]

    model_config = {"extra": "forbid", "strict": True}


class FramesDocument(RootModel[list[FrameRecord]]):
    """`visual/frames.json`, an ordered array of unique frame indices."""

    model_config = {"strict": True, "json_schema_extra": {"x-lectern-uniqueBy": "index"}}

    @model_validator(mode="after")
    def validate_document(self) -> FramesDocument:
        indices = [frame.index for frame in self.root]
        if len(indices) != len(set(indices)):
            raise ValueError("frame indices must be unique")
        if any(left.t_s > right.t_s for left, right in zip(self.root, self.root[1:], strict=False)):
            raise ValueError("frames must be ordered by timestamp")
        return self


SpeakerLabel = Annotated[str, StringConstraints(strict=True, pattern=r"^S[1-9][0-9]*$")]


class DiarizationTurn(ArtifactModel):
    start_s: Annotated[FiniteFloat, Field(ge=0)]
    end_s: Annotated[FiniteFloat, Field(ge=0)]
    speaker: SpeakerLabel

    model_config = {"extra": "forbid", "strict": True}

    @model_validator(mode="after")
    def validate_span(self) -> DiarizationTurn:
        if self.end_s <= self.start_s:
            raise ValueError("diarization turn must end after its start")
        return self


class SegmentSpeaker(ArtifactModel):
    segment_id: SegmentId
    speaker: SpeakerLabel | None

    model_config = {"extra": "forbid", "strict": True}


class DiarizationDocument(ArtifactModel):
    """`transcript/diarization.json`, anonymous local speaker evidence."""

    method: str
    speaker_count: NonNegativeStrictInt
    count_source: Literal["given", "estimated"]
    turns: list[DiarizationTurn]
    segment_speakers: list[SegmentSpeaker]

    model_config = {"extra": "forbid", "strict": True}

    @model_validator(mode="after")
    def validate_document(self) -> DiarizationDocument:
        if any(a.start_s > b.start_s for a, b in zip(self.turns, self.turns[1:], strict=False)):
            raise ValueError("diarization turns must be ordered by start time")
        labels = list(dict.fromkeys(turn.speaker for turn in self.turns))
        if labels != [f"S{i + 1}" for i in range(len(labels))]:
            raise ValueError("speaker labels must follow first appearance")
        if len(labels) > self.speaker_count or (
            self.count_source == "estimated" and len(labels) != self.speaker_count
        ):
            raise ValueError("speaker count does not match the labelled turns")
        ids = [assignment.segment_id for assignment in self.segment_speakers]
        if len(ids) != len(set(ids)):
            raise ValueError("segment speaker ids must be unique")
        if any(s.speaker is not None and s.speaker not in labels for s in self.segment_speakers):
            raise ValueError("segment speaker must name a recorded speaker")
        return self


ARTIFACT_MODELS: dict[str, type[BaseModel]] = {
    "manifest": Manifest,
    "source": SourceDocument,
    "transcript-segments": TranscriptSegmentsDocument,
    "transcript-metadata": TranscriptMetadataDocument,
    "frames": FramesDocument,
    "diarization": DiarizationDocument,
}


def export_artifact_schemas() -> dict[str, str]:
    """Every artifact schema, keyed by the name its file takes under `schemas/`."""

    return {
        name: json.dumps(model.model_json_schema(by_alias=True), indent=2) + "\n"
        for name, model in ARTIFACT_MODELS.items()
    }


def export_json_schema() -> str:
    """Export the manifest JSON Schema (committed under schemas/ on change)."""
    return json.dumps(Manifest.model_json_schema(), indent=2) + "\n"
