"""Local-folder discovery: what a scan of a directory tree finds, and what it skips.

Content addressing lives here too, because what a local scan considers "the same
item" is the media bytes plus any transcript sidecar beside them.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lectern.bundle import MANIFEST_NAME
from lectern.records import (
    AutomationError,
    SourceItem,
    SourceRecord,
    approval_digest,
    digest_and_size,
    make_source_item_id,
    now_timestamp,
)

MEDIA_EXTENSIONS = frozenset(
    {".aac", ".avi", ".flac", ".m4a", ".mkv", ".mov", ".mp3", ".mp4", ".ogg", ".wav", ".webm"}
)

EXCLUDED_SCAN_DIR_NAMES = frozenset({".lectern"})

EXCLUDED_SCAN_DIR_PREFIXES = (".lectern-ingest.",)


@dataclass(frozen=True)
class SourcePreflight:
    path: str
    exists: bool
    is_dir: bool
    readable: bool
    media_files: int

    @property
    def ok(self) -> bool:
        return self.exists and self.is_dir and self.readable

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "exists": self.exists,
            "is_dir": self.is_dir,
            "readable": self.readable,
            "media_files": self.media_files,
            "ok": self.ok,
        }


def iter_local_media_files(root: Path) -> Iterator[Path]:
    """Yield, in scan order, the media files a local-folder scan will discover.

    `LocalFolderAdapter.discover` and `preflight_local_folder` share this helper so a
    preflight count reports what a scan of the same tree actually discovers, rather
    than every media-suffixed file including Lectern's own bundle output.
    """

    root_resolved = root.resolve()
    for path in sorted(root.rglob("*")):
        relative_parts = path.parent.relative_to(root).parts
        is_excluded_dir = any(part in EXCLUDED_SCAN_DIR_NAMES for part in relative_parts)
        is_excluded_temp_dir = any(
            part.startswith(EXCLUDED_SCAN_DIR_PREFIXES) for part in relative_parts
        )
        if is_excluded_dir or is_excluded_temp_dir or is_bundle_output_path(root, path):
            continue
        if path.is_symlink() or not path.is_file() or path.suffix.lower() not in MEDIA_EXTENSIONS:
            continue
        try:
            path.resolve().relative_to(root_resolved)
        except ValueError:
            continue
        yield path


def is_bundle_output_path(root: Path, path: Path) -> bool:
    """Report whether `path` sits inside a Lectern bundle under `root`.

    Public because it states half of what a local scan excludes, alongside the
    excluded-directory rules `iter_local_media_files` applies.

    Everything happens in **one** path space: both operands are resolved once,
    and the walk climbs the resolved path. Deciding containment in normalized
    space while walking raw paths is what makes this function subtly wrong, and
    it goes wrong in more than one way — `root/../foreign/x` is lexically
    "relative to" `root`, and a symlinked `root` never equals any ancestor of
    its resolved target, so the walk sails past the root it was supposed to stop
    at and inspects markers above it. Resolving first, then walking, removes the
    whole class rather than each symptom.

    Containment is decided before any marker is inspected, because a path
    outside `root` can still sit inside some *other* Lectern bundle, and that
    foreign bundle is not this root's output.
    """

    try:
        resolved_root = root.resolve()
        resolved_path = path.resolve()
        resolved_path.relative_to(resolved_root)
    except (OSError, ValueError):
        return False

    ancestor = resolved_path.parent
    while True:
        if (ancestor / MANIFEST_NAME).is_file() and (ancestor / "source.json").is_file():
            return True
        if ancestor == resolved_root:
            return False
        parent = ancestor.parent
        if parent == ancestor:
            # Containment plus a resolved walk make this unreachable: climbing
            # from a path under `resolved_root` must meet it. Kept because the
            # filesystem root is its own parent, so without it the loop has no
            # termination condition at all for any caller that reaches it by
            # another route.
            return False
        ancestor = parent


def _transcript_sidecar_escapes_root(path: Path, root: Path) -> bool:
    """Report whether a media file's transcript sidecar resolves outside `root`.

    `LocalFolderAdapter.discover` drops such media (the sidecar is content the
    approval digest would cover, so it must stay inside the source), and
    `preflight_local_folder` applies the same rule so its count matches discovery.

    Both operands are resolved, for the same reason `is_bundle_output_path`
    resolves both: comparing a resolved sidecar against the root as given puts
    the two sides in different path spaces, and a symlinked root then makes a
    fully contained sidecar look like an escape. Today's callers happen to pass
    an already-resolved root, so this was latent rather than live — but the
    function is what states the rule, so it is what should enforce it.
    """

    sidecar = transcript_sidecar_path(path)
    if sidecar is None:
        return False
    return _sidecar_escapes_root(sidecar, root)


def _sidecar_escapes_root(sidecar: Path, root: Path) -> bool:
    if sidecar.is_symlink():
        return True
    try:
        sidecar.resolve().relative_to(root.resolve())
    except (OSError, ValueError):
        return True
    return False


def transcript_sidecar_path(path: Path, *, root: Path | None = None) -> Path | None:
    """Select captions before plain text, identically for discovery and capture."""

    exact = [path.with_suffix(suffix) for suffix in (".vtt", ".srt")]
    tagged = sorted(
        candidate
        for candidate in path.parent.iterdir()
        if caption_sidecar_language(path, candidate) is not None
    )
    sidecar = next(
        (
            candidate
            for candidate in [*exact, *tagged, path.with_suffix(".transcript.txt")]
            if candidate.is_file()
        ),
        None,
    )
    if sidecar is None:
        return None
    if root is not None and _sidecar_escapes_root(sidecar, root):
        raise AutomationError("transcript sidecar must be inside the source root")
    return sidecar


def caption_sidecar_language(media: Path, sidecar: Path) -> str | None:
    """Read a BCP-47-like filename tag, without interpreting it as diarization."""

    match = re.fullmatch(
        re.escape(media.stem) + r"\.([A-Za-z]{2,8}(?:-[A-Za-z0-9]{1,8})*)\.(?:vtt|srt)",
        sidecar.name,
    )
    return match.group(1) if match is not None else None


def approval_digest_and_media_size(path: Path, *, root: Path | None = None) -> tuple[str, int]:
    media_digest, media_size = digest_and_size(path)
    sidecar = transcript_sidecar_path(path, root=root)
    sidecar_digest = digest_and_size(sidecar)[0] if sidecar is not None else None
    return approval_digest(media_digest, sidecar_digest), media_size


class LocalFolderAdapter:
    """Discover media files under a local directory without network access."""

    def discover(self, source: SourceRecord) -> list[SourceItem]:
        root = Path(source.root_path)
        if not root.is_dir():
            raise AutomationError(f"local folder source is not a directory: {root}")

        root_resolved = root.resolve()
        items: list[SourceItem] = []
        for path in iter_local_media_files(root):
            absolute = path.resolve()
            relative = path.relative_to(root).as_posix()
            try:
                digest, size = approval_digest_and_media_size(path, root=root_resolved)
            except AutomationError:
                continue
            stat = path.stat()
            now = now_timestamp()
            items.append(
                SourceItem(
                    id=make_source_item_id(source.id, relative),
                    source_id=source.id,
                    relative_path=relative,
                    absolute_path=str(absolute),
                    sha256=digest,
                    size_bytes=size,
                    mtime_ns=stat.st_mtime_ns,
                    present=True,
                    created_at=now,
                    updated_at=now,
                )
            )
        return items


def preflight_local_folder(path: Path) -> SourcePreflight:
    resolved = path.resolve()
    exists = resolved.exists()
    is_dir = resolved.is_dir()
    readable = False
    media_files = 0
    if is_dir:
        try:
            media_files = sum(
                1
                for media in iter_local_media_files(resolved)
                if not _transcript_sidecar_escapes_root(media, resolved)
            )
            readable = True
        except OSError:
            readable = False
    return SourcePreflight(
        path=str(resolved),
        exists=exists,
        is_dir=is_dir,
        readable=readable,
        media_files=media_files,
    )
