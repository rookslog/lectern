"""The automation spine's shared vocabulary: errors, records, and identities.

Everything here is inert: enums, frozen records, the queue's legal-transition
table, the discovery protocols, and the pure functions that mint the identities
and digests those records carry. It imports nothing else from `lectern`, which
is what lets the state store, the source adapters, and the provenance writer all
depend on it without depending on each other.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable


def _empty_metadata() -> dict[str, Any]:
    return {}


class AutomationError(RuntimeError):
    """Raised when the local automation spine cannot complete a requested action."""


class SourceKind(StrEnum):
    LOCAL_FOLDER = "local-folder"
    ONE_SHOT = "one-shot"
    YOUTUBE_PLAYLIST = "youtube-playlist"


class LibraryKind(StrEnum):
    """What sort of thing a library record points at today."""

    BUNDLE = "bundle"


class SourcePolicy(StrEnum):
    DISABLED = "disabled"
    SCAN_ONLY = "scan-only"
    REVIEW = "review"


class QueueState(StrEnum):
    DISCOVERED = "discovered"
    APPROVED = "approved"
    SKIPPED = "skipped"
    FAILED = "failed"
    COMPLETED = "completed"
    UNSUPPORTED = "unsupported"


class LibraryStatus(StrEnum):
    """A user-facing summary of trustworthy queue and bundle-stage state."""

    INCOMPLETE = "incomplete"
    FAILED = "failed"
    NEEDS_REPROCESSING = "needs-reprocessing"
    READY = "ready"


TERMINAL_QUEUE_STATES = frozenset({QueueState.UNSUPPORTED})

# Stable ordering so the guarded-UPDATE placeholders bind the same values every run.
TERMINAL_QUEUE_STATE_VALUES = tuple(sorted(state.value for state in TERMINAL_QUEUE_STATES))

# The queue FSM, as an explicit table: for each user verb, the states an item may
# legally be transitioned *from*. Every guarded verb checks this table, so the
# legality of a transition can be read here rather than inferred from call sites.
#
# - `approve` and `skip` stay legal from every non-terminal state; re-approving a
#   COMPLETED item is what the rerun/replay/restore path depends on.
# - `retry` is legal from FAILED only. Retry exists to re-arm a failed attempt;
#   from COMPLETED it would silently discard a finished bundle's queue linkage,
#   and from DISCOVERED/APPROVED/SKIPPED it is a no-op dressed as a state change.
# - UNSUPPORTED appears in no set, so terminal items refuse every verb.
LEGAL_QUEUE_TRANSITION_SOURCES: Mapping[str, frozenset[QueueState]] = MappingProxyType(
    {
        "approve": frozenset(QueueState) - TERMINAL_QUEUE_STATES,
        "skip": frozenset(QueueState) - TERMINAL_QUEUE_STATES,
        "retry": frozenset({QueueState.FAILED}),
    }
)

# Stable ordering, as above, so each verb's guarded-UPDATE placeholders bind the
# same values every run.
LEGAL_QUEUE_TRANSITION_SOURCE_VALUES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        action: tuple(sorted(state.value for state in states))
        for action, states in LEGAL_QUEUE_TRANSITION_SOURCES.items()
    }
)


def derive_library_status(
    queue_state: QueueState,
    stage_states: Iterable[str],
    *,
    source_changed: bool = False,
    manifest_available: bool = True,
) -> LibraryStatus:
    """Summarize only states the queue FSM and manifest actually record.

    Queue transitions remain owned by ``LEGAL_QUEUE_TRANSITION_SOURCES``. This
    pure projection cannot create a transition; it reports the row and stage
    records that survived those guards. Untouched future stages are omitted by
    the state-store adapter, so their default ``pending`` does not make a fully
    produced bundle look incomplete.
    """

    stages = {str(state) for state in stage_states}
    if queue_state in {QueueState.FAILED, QueueState.UNSUPPORTED} or "failed" in stages:
        return LibraryStatus.FAILED
    if source_changed:
        return LibraryStatus.NEEDS_REPROCESSING
    if queue_state is not QueueState.COMPLETED:
        return LibraryStatus.INCOMPLETE
    if not manifest_available or not stages or stages & {"pending", "running"}:
        return LibraryStatus.INCOMPLETE
    return LibraryStatus.READY


# Paths in error text routinely contain spaces, and stopping at whitespace
# redacts only the first component: `/tmp/Private Therapy/session.wav` became
# `<path> Therapy/session.wav`, still naming the directory and the file. The
# terminator is therefore a quote or end-of-string when the path is quoted --
# which is how OSError renders it -- and whitespace only as a fallback for
# unquoted paths.
_QUOTED_ABSOLUTE_PATH = re.compile(r"(?<=')/(?:\\.|[^'\\])*(?=')|(?<=\")/(?:\\.|[^\"\\])*(?=\")")
_BARE_ABSOLUTE_PATH = re.compile(r"(?<![\w/])/(?!/)[^\r\n]*")
_FILE_URI = re.compile(r"(?i)\bfile:/+[^\r\n'\"<>|]*")
_WINDOWS_ABSOLUTE_PATH = re.compile(
    r"(?i)(?<!\w)(?:[a-z]:[\\/]|\\\\|(?<!\\)\\(?!\\))[^\r\n'\"<>|]*"
)
_DOUBLE_SLASH_PATH = re.compile(r"(?<![:/])//[^\r\n'\"<>|]*")

PATH_REDACTED = "<path>"


def redact_paths(text: str) -> str:
    """Replace POSIX absolute paths in free text with a fixed placeholder.

    Needed because paths reach outward-facing data through *messages*, not only
    through fields. An `OSError` for a missing file interpolates the absolute
    filename into its string form, that string is persisted as a queue item's
    `last_error`, and it is served back verbatim. No field-level rule reaches
    that, which is why the boundary is stated over path-bearing *values*
    including free text rather than over a list of columns.

    Deliberately blunt: it removes the whole path rather than a prefix, because
    a partial path is still the user's filesystem — an intermediate directory
    name discloses as much as the home directory does.
    """

    redacted = _QUOTED_ABSOLUTE_PATH.sub(PATH_REDACTED, text)
    for pattern in (_FILE_URI, _WINDOWS_ABSOLUTE_PATH, _DOUBLE_SLASH_PATH, _BARE_ABSOLUTE_PATH):
        redacted = pattern.sub(PATH_REDACTED, redacted)
    return redacted


@dataclass(frozen=True)
class SourceRecord:
    id: str
    kind: SourceKind
    name: str
    root_path: str
    policy: SourcePolicy
    created_at: str
    updated_at: str

    def to_dict(self) -> dict[str, Any]:
        """The outward-facing projection, withholding `root_path` when it is a path.

        `root_path` stays on the record because the scanner needs it; whether it
        is disclosed depends on the kind, because the column is overloaded. For
        a local folder or a one-shot it holds a filesystem path. For a YouTube
        playlist it holds the playlist ID — a public, opaque identifier that
        discloses nothing about the machine, and that a caller needs in order to
        tell two playlist sources apart.

        The disclosure list is an allowlist rather than a denylist: a kind added
        later withholds its `root_path` until someone decides otherwise, which
        is the safe direction to be wrong in.

        Withholding is structural rather than a rule call sites must remember —
        an earlier design asserted that no read surface returns a path while
        four commands were returning one.
        """

        discloses_root = self.kind is SourceKind.YOUTUBE_PLAYLIST
        return {
            "id": self.id,
            "kind": self.kind.value,
            "name": self.name,
            "root_path": self.root_path if discloses_root else None,
            "policy": self.policy.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True)
class SourceItem:
    id: str
    source_id: str
    relative_path: str
    absolute_path: str
    sha256: str
    size_bytes: int
    mtime_ns: int
    present: bool
    created_at: str
    updated_at: str
    metadata: dict[str, Any] = field(default_factory=_empty_metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source_id": self.source_id,
            # `relative_path` survives and `absolute_path` does not: the former
            # is meaningful only against a root the caller already chose, so it
            # discloses nothing about where that root sits.
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "mtime_ns": self.mtime_ns,
            "present": self.present,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class QueueItem:
    id: str
    source_id: str
    source_item_id: str
    content_sha256: str
    state: QueueState
    policy: SourcePolicy
    bundle_id: str | None
    attempts: int
    last_error: str | None
    created_at: str
    updated_at: str
    metadata: dict[str, Any] = field(default_factory=_empty_metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source_id": self.source_id,
            "source_item_id": self.source_item_id,
            "content_sha256": self.content_sha256,
            "state": self.state.value,
            "policy": self.policy.value,
            "bundle_id": self.bundle_id,
            "attempts": self.attempts,
            # Redacted rather than dropped: the failure reason is what makes a
            # stuck queue diagnosable, and it is the path inside the message —
            # not the message — that must not leave. This is the value the
            # column-level inventory missed, because it is populated only when
            # an operation fails and so is absent from every passing fixture.
            "last_error": None if self.last_error is None else redact_paths(self.last_error),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class SearchHit:
    """One matching transcript segment, addressed by bundle and segment.

    Carries no path, and has no field that could hold one. That is the
    difference between a boundary and a habit: a caller cannot leak a location
    through this type by forgetting to project, because there is nowhere to put
    it.
    """

    bundle_id: str
    segment_id: int | None
    snippet: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "bundle_id": self.bundle_id,
            "segment_id": self.segment_id,
            "snippet": self.snippet,
        }


@dataclass(frozen=True)
class LibraryBundle:
    bundle_id: str
    bundle_path: str
    source_id: str
    source_item_id: str
    queue_item_id: str
    created_at: str
    kind: LibraryKind = LibraryKind.BUNDLE
    status: LibraryStatus = LibraryStatus.READY

    def to_dict(self) -> dict[str, Any]:
        return {
            "bundle_id": self.bundle_id,
            # Lectern's own output location, not the user's media — and it
            # leaks the same account name, which is why a rule scoped to
            # "media references" would have left it in place.
            "source_id": self.source_id,
            "source_item_id": self.source_item_id,
            "queue_item_id": self.queue_item_id,
            "created_at": self.created_at,
            "kind": self.kind.value,
            "status": self.status.value,
        }


@dataclass(frozen=True)
class LibraryRecordOutcome:
    """What a staged library record changed, so the undo path can reverse it.

    A rerun into a different output root can produce the same deterministic bundle
    id, in which case the row is updated rather than inserted; reversing that needs
    the prior row, not just the knowledge that nothing was inserted.
    """

    inserted: bool
    previous: LibraryBundle | None


@dataclass(frozen=True)
class ScanDelta:
    source: SourceRecord
    added: list[SourceItem]
    changed: list[SourceItem]
    removed: list[SourceItem]
    unchanged: list[SourceItem]
    queued: list[QueueItem]
    metadata: dict[str, Any] = field(default_factory=_empty_metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source.to_dict(),
            "added": [item.to_dict() for item in self.added],
            "changed": [item.to_dict() for item in self.changed],
            "removed": [item.to_dict() for item in self.removed],
            "unchanged": [item.to_dict() for item in self.unchanged],
            "queued": [item.to_dict() for item in self.queued],
            "metadata": self.metadata,
            "counts": {
                "added": len(self.added),
                "changed": len(self.changed),
                "removed": len(self.removed),
                "unchanged": len(self.unchanged),
                "queued": len(self.queued),
            },
        }


class SourceAdapter(Protocol):
    """Discovery seam for local folders now and external adapters later."""

    def discover(self, source: SourceRecord) -> Sequence[SourceItem]:
        """Return the current source items for a source."""
        ...


@runtime_checkable
class ScanMetadataProvider(Protocol):
    @property
    def scan_metadata(self) -> dict[str, Any]:
        """Return scan-level metadata from the most recent discovery run."""
        ...


def now_timestamp() -> str:
    return datetime.now(UTC).isoformat()


def approval_digest(
    media_sha256: str, sidecar_sha256: str | None, info_sha256: str | None = None
) -> str:
    """Existing approval identity, composed from captured component digests."""

    digest = hashlib.sha256(b"media\0" + media_sha256.encode("ascii") + b"\0")
    digest.update(
        b"transcript-sidecar\0" + sidecar_sha256.encode("ascii")
        if sidecar_sha256 is not None
        else b"transcript-sidecar-absent"
    )
    # Preserve existing approvals when metadata is absent. Presence, removal,
    # and every byte change are covered when an info sidecar is supplied.
    if info_sha256 is not None:
        digest.update(b"\0info-sidecar\0" + info_sha256.encode("ascii"))
    return digest.hexdigest()


def digest_and_size(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return digest.hexdigest(), size


def metadata_to_json(metadata: dict[str, Any]) -> str:
    return json.dumps(metadata, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def make_source_id(kind: str, root_path: str) -> str:
    return _stable_id("src", [kind, root_path])


def make_source_item_id(source_id: str, relative_path: str) -> str:
    return _stable_id("item", [source_id, relative_path])


def make_queue_item_id(source_item_id: str, content_sha256: str) -> str:
    return _stable_id("queue", [source_item_id, content_sha256])


def _stable_id(prefix: str, parts: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\0")
    return f"{prefix}_{digest.hexdigest()[:16]}"
