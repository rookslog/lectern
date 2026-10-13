"""The SQLite state store: sources, discovery queue, and library index.

This module owns persistence and the queue FSM. It deliberately does not own the
ingest pipeline: `lectern.automation.AutomationState` composes that on top, so
the layer that writes bundles is separable from the layer that records them.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import sqlite3
import stat
import unicodedata
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn, Self, cast

from lectern.bundle import Manifest, SourceDocument, StageState, TranscriptSegmentsDocument
from lectern.evidence import SelectedEvidence, read_selected_evidence
from lectern.records import (
    LEGAL_QUEUE_TRANSITION_SOURCE_VALUES,
    LEGAL_QUEUE_TRANSITION_SOURCES,
    TERMINAL_QUEUE_STATES,
    AutomationError,
    LibraryBundle,
    LibraryKind,
    LibraryRecordOutcome,
    LibraryStatus,
    QueueItem,
    QueueState,
    ScanDelta,
    ScanMetadataProvider,
    SearchHit,
    SourceAdapter,
    SourceItem,
    SourceKind,
    SourcePolicy,
    SourceRecord,
    derive_library_status,
    digest_and_size,
    make_queue_item_id,
    make_source_id,
    metadata_to_json,
    now_timestamp,
)
from lectern.search import (
    Anchor,
    AnchorIssue,
    AnchorResolution,
    ResolvedAnchor,
    SamplerIssue,
    canonical_text,
    index_signature,
    is_unsegmented_script,
    literal_match_expression,
    make_anchor,
    operator_segment_text,
    resolve_against_segments,
    sample_segment_timings,
    segment_text,
    text_contains_literal,
)

_SEGMENT_INDEX_COLUMNS = ("bundle_id", "segment_id", "display", "body", "literal")


def _restrict_existing_sqlite_sidecar(path: Path) -> None:
    """Restrict a pre-existing SQLite sidecar before SQLite can read or reuse it."""

    flags = os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise AutomationError("cannot securely open a state database sidecar") from exc
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise AutomationError("state database sidecar is not a regular file")
        os.fchmod(descriptor, 0o600)
    finally:
        os.close(descriptor)


def _unlink_if_same_file(path: Path, identity: tuple[int, int]) -> None:
    """Remove only the new path still naming the file this invocation created."""

    try:
        current_stat = os.stat(path, follow_symlinks=False)
    except OSError:
        return
    if (
        stat.S_ISREG(current_stat.st_mode)
        and (
            current_stat.st_dev,
            current_stat.st_ino,
        )
        == identity
    ):
        path.unlink(missing_ok=True)


class AutomationStateStore:
    """SQLite-backed local automation state store."""

    def __init__(self, path: Path) -> None:
        self.path = path
        try:
            self.path.parent.mkdir(parents=True)
        except FileExistsError:
            pass
        else:
            os.chmod(self.path.parent, 0o700)
        parent_mode = stat.S_IMODE(self.path.parent.stat().st_mode)
        if parent_mode & 0o022:
            raise AutomationError("state database directory is not private")
        created = False
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        except FileExistsError:
            flags = os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
            try:
                descriptor = os.open(path, flags)
            except OSError as exc:
                raise AutomationError("cannot securely open the state database") from exc
        else:
            created = True
        opened_identity: tuple[int, int] | None = None
        try:
            opened_stat = os.fstat(descriptor)
            opened_identity = (opened_stat.st_dev, opened_stat.st_ino)
            if not stat.S_ISREG(opened_stat.st_mode):
                raise AutomationError("state database path is not a regular file")
            os.fchmod(descriptor, 0o600)
            for suffix in ("-journal", "-wal", "-shm"):
                _restrict_existing_sqlite_sidecar(Path(f"{path}{suffix}"))

            try:
                self._connection = sqlite3.connect(path)
            except OSError as exc:
                raise AutomationError("cannot securely open the state database") from exc
            try:
                current_stat = os.stat(path, follow_symlinks=False)
            except OSError as exc:
                self._connection.close()
                raise AutomationError("state database path changed while opening") from exc
            if (
                not stat.S_ISREG(current_stat.st_mode)
                or (
                    current_stat.st_dev,
                    current_stat.st_ino,
                )
                != opened_identity
            ):
                self._connection.close()
                raise AutomationError("state database path changed while opening")
        except BaseException:
            if created and opened_identity is not None:
                _unlink_if_same_file(path, opened_identity)
            raise
        finally:
            os.close(descriptor)
        try:
            self._connection.row_factory = sqlite3.Row
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._migrate()
            self._rebuild_index_if_stale()
            self.refresh_changed_bundles()
        except BaseException:
            # A store that never finished initializing is never returned, so no
            # caller holds it to close. Releasing the handle here is the only
            # opportunity; `BaseException` because an interrupt between connect
            # and migrate leaks exactly as a migration error does.
            self._connection.close()
            if created:
                _unlink_if_same_file(path, opened_identity)
            raise

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def add_local_folder_source(
        self,
        name: str,
        root_path: Path,
        policy: SourcePolicy = SourcePolicy.REVIEW,
    ) -> SourceRecord:
        root = root_path.resolve()
        if not root.is_dir():
            raise AutomationError(f"source path is not a directory: {root}")
        source_id = make_source_id(SourceKind.LOCAL_FOLDER.value, str(root))
        now = now_timestamp()
        existing = self._connection.execute(
            "SELECT * FROM sources WHERE id = ? OR name = ?",
            (source_id, name),
        ).fetchone()
        if existing is not None:
            source = _source_from_row(existing)
            if (
                source.kind is SourceKind.LOCAL_FOLDER
                and source.name == name
                and source.root_path == str(root)
                and source.policy is policy
            ):
                return source
            raise AutomationError("source name or path already exists with different settings")
        self._connection.execute(
            """
            INSERT INTO sources(id, kind, name, root_path, policy, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (source_id, SourceKind.LOCAL_FOLDER.value, name, str(root), policy.value, now, now),
        )
        self._connection.commit()
        return self.get_source(source_id)

    def add_youtube_playlist_source(
        self,
        name: str,
        playlist: str,
        policy: SourcePolicy = SourcePolicy.REVIEW,
    ) -> SourceRecord:
        """Register a playlist source from an already-normalized playlist id.

        Turning a user-supplied playlist URL into an id is provider knowledge,
        so it belongs to the layer that knows about providers; the store is
        handed the identity it should persist. `AutomationState` overrides this
        to normalize first, which is the entry point every caller reaches.
        """

        source_id = make_source_id(SourceKind.YOUTUBE_PLAYLIST.value, playlist)
        now = now_timestamp()
        existing = self._connection.execute(
            "SELECT * FROM sources WHERE id = ? OR name = ?",
            (source_id, name),
        ).fetchone()
        if existing is not None:
            source = _source_from_row(existing)
            if (
                source.kind is SourceKind.YOUTUBE_PLAYLIST
                and source.name == name
                and source.root_path == playlist
                and source.policy is policy
            ):
                return source
            raise AutomationError("source name or playlist already exists with different settings")
        self._connection.execute(
            """
            INSERT INTO sources(id, kind, name, root_path, policy, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                source_id,
                SourceKind.YOUTUBE_PLAYLIST.value,
                name,
                playlist,
                policy.value,
                now,
                now,
            ),
        )
        self._connection.commit()
        return self.get_source(source_id)

    def list_sources(self) -> list[SourceRecord]:
        rows = self._connection.execute(
            "SELECT * FROM sources ORDER BY created_at, name"
        ).fetchall()
        return [_source_from_row(row) for row in rows]

    def get_source(self, source_id_or_name: str) -> SourceRecord:
        row = self._connection.execute(
            "SELECT * FROM sources WHERE id = ? OR name = ?",
            (source_id_or_name, source_id_or_name),
        ).fetchone()
        if row is None:
            raise AutomationError(f"source not found: {source_id_or_name}")
        return _source_from_row(row)

    def _default_adapter(self, source: SourceRecord) -> SourceAdapter:
        """Resolve the adapter `scan_source` uses when the caller supplies none.

        The store deliberately knows no concrete adapter. That edge is what let
        a Google HTTP client live in the persistence layer, and importing the
        adapter package here would reinstate it in a subtler form: loading the
        store would transitively load a network transport. The composition root
        overrides this.
        """

        raise AutomationError(f"unsupported source kind for scan: {source.kind.value}")

    def scan_source(
        self,
        source_id_or_name: str,
        adapter: SourceAdapter | None = None,
    ) -> ScanDelta:
        source = self.get_source(source_id_or_name)
        if source.policy is SourcePolicy.DISABLED:
            return ScanDelta(
                source=source,
                added=[],
                changed=[],
                removed=[],
                unchanged=[],
                queued=[],
            )

        adapter_for_scan = adapter or self._default_adapter(source)
        current_items = _dedupe_by_relative_path(adapter_for_scan.discover(source))
        scan_metadata = _scan_metadata_from_adapter(adapter_for_scan)
        truncated = _scan_reports_truncation(scan_metadata)
        if truncated:
            scan_metadata = {**scan_metadata, "removals_skipped_due_to_truncation": True}
        previous = {
            item.relative_path: item
            for item in self._list_source_items(source.id, present_only=False)
        }
        current_by_relative = {item.relative_path: item for item in current_items}

        added: list[SourceItem] = []
        changed: list[SourceItem] = []
        removed: list[SourceItem] = []
        unchanged: list[SourceItem] = []
        queued: list[QueueItem] = []

        for item in current_items:
            old = previous.get(item.relative_path)
            should_enqueue = False
            if old is None or not old.present:
                added.append(item)
                should_enqueue = True
            elif old.sha256 != item.sha256 or old.size_bytes != item.size_bytes:
                changed.append(item)
                should_enqueue = True
            else:
                unchanged.append(item)
            self._upsert_source_item(item, created_at=old.created_at if old is not None else None)
            if should_enqueue:
                queue_item = self._enqueue_if_review(source, item)
                if queue_item is not None:
                    queued.append(queue_item)

        # A truncated discovery is a partial view of the source: absence from it is not
        # evidence of removal, so removals are skipped entirely rather than persisted.
        removal_candidates: list[SourceItem] = [] if truncated else list(previous.values())
        for old in removal_candidates:
            if old.present and old.relative_path not in current_by_relative:
                removed_item = SourceItem(
                    id=old.id,
                    source_id=old.source_id,
                    relative_path=old.relative_path,
                    absolute_path=old.absolute_path,
                    sha256=old.sha256,
                    size_bytes=old.size_bytes,
                    mtime_ns=old.mtime_ns,
                    present=False,
                    created_at=old.created_at,
                    updated_at=now_timestamp(),
                    metadata=old.metadata,
                )
                removed.append(removed_item)
                self._upsert_source_item(removed_item, created_at=old.created_at)

        self._connection.commit()
        return ScanDelta(
            source=source,
            added=added,
            changed=changed,
            removed=removed,
            unchanged=unchanged,
            queued=queued,
            metadata=scan_metadata,
        )

    def list_queue(self, state: QueueState | None = None) -> list[QueueItem]:
        if state is None:
            rows = self._connection.execute(
                "SELECT * FROM queue_items ORDER BY created_at, id"
            ).fetchall()
        else:
            rows = self._connection.execute(
                "SELECT * FROM queue_items WHERE state = ? ORDER BY created_at, id",
                (state.value,),
            ).fetchall()
        return [queue_from_row(row) for row in rows]

    def get_queue_item(self, queue_item_id: str) -> QueueItem:
        row = self._connection.execute(
            "SELECT * FROM queue_items WHERE id = ?",
            (queue_item_id,),
        ).fetchone()
        if row is None:
            raise AutomationError(f"queue item not found: {queue_item_id}")
        return queue_from_row(row)

    def approve_queue_item(self, queue_item_id: str) -> QueueItem:
        self._reject_illegal_transition(queue_item_id, "approve")
        return self._set_queue_state(
            queue_item_id, QueueState.APPROVED, clear_error=True, guarded_action="approve"
        )

    def skip_queue_item(self, queue_item_id: str) -> QueueItem:
        self._reject_illegal_transition(queue_item_id, "skip")
        return self._set_queue_state(
            queue_item_id, QueueState.SKIPPED, clear_error=True, guarded_action="skip"
        )

    def retry_queue_item(self, queue_item_id: str) -> QueueItem:
        self._reject_illegal_transition(queue_item_id, "retry")
        return self._set_queue_state(
            queue_item_id, QueueState.DISCOVERED, clear_error=True, guarded_action="retry"
        )

    def _reject_illegal_transition(self, queue_item_id: str, action: str) -> None:
        """Refuse a verb the FSM table does not allow out of the item's current state.

        Terminal states keep their own wording; other illegal sources name the verb
        and the states it is legal from. The item is read exactly once, so a caller
        racing a concurrent writer decides against a single observed row.
        """

        queue_item = self.get_queue_item(queue_item_id)
        if queue_item.state in TERMINAL_QUEUE_STATES:
            raise AutomationError(
                f"queue item {queue_item.id} is in terminal state "
                f"{queue_item.state.value}; {action} is not a legal transition"
            )
        if queue_item.state not in LEGAL_QUEUE_TRANSITION_SOURCES[action]:
            raise AutomationError(
                f"queue item {queue_item.id} is in state {queue_item.state.value}; "
                f"{action} is only legal from state "
                f"{_join_states(LEGAL_QUEUE_TRANSITION_SOURCE_VALUES[action])}"
            )

    def _reject_guarded_transition_conflict(self, queue_item_id: str, action: str) -> NoReturn:
        """Explain a guarded update that matched no row, from the row as it stands now.

        The pre-check read and the update are separate statements, so another writer
        can move the item into a state the verb is illegal from in between; the
        update's own FSM guard is what refuses that, and this re-read turns the empty
        result into the same error the pre-check would have raised.
        """

        # The zero-row UPDATE still opened an implicit write transaction; release
        # it before raising so a caller that catches the expected rejection does
        # not leave the connection holding a write lock.
        self._connection.rollback()
        self._reject_illegal_transition(queue_item_id, action)
        queue_item = self.get_queue_item(queue_item_id)
        raise AutomationError(f"queue item {queue_item.id} could not be transitioned by {action}")

    def get_source_item(self, source_item_id: str) -> SourceItem:
        row = self._connection.execute(
            "SELECT * FROM source_items WHERE id = ?",
            (source_item_id,),
        ).fetchone()
        if row is None:
            raise AutomationError(f"source item not found: {source_item_id}")
        return source_item_from_row(row)

    def list_library(self) -> list[LibraryBundle]:
        rows = self._connection.execute(
            """
            SELECT library_bundles.*,
                   queue_items.state AS library_queue_state,
                   queue_items.content_sha256 AS library_queue_sha256,
                   source_items.sha256 AS library_source_sha256
            FROM library_bundles
            JOIN queue_items ON queue_items.id = library_bundles.queue_item_id
            JOIN source_items ON source_items.id = library_bundles.source_item_id
            ORDER BY library_bundles.created_at, library_bundles.bundle_id
            """
        ).fetchall()
        return [_library_bundle_from_row(row, status=_library_status_from_row(row)) for row in rows]

    def get_library_bundle(self, bundle_id: str) -> LibraryBundle:
        row = self._connection.execute(
            """
            SELECT library_bundles.*,
                   queue_items.state AS library_queue_state,
                   queue_items.content_sha256 AS library_queue_sha256,
                   source_items.sha256 AS library_source_sha256
            FROM library_bundles
            JOIN queue_items ON queue_items.id = library_bundles.queue_item_id
            JOIN source_items ON source_items.id = library_bundles.source_item_id
            WHERE library_bundles.bundle_id = ?
            """,
            (bundle_id,),
        ).fetchone()
        if row is None:
            raise AutomationError(f"bundle not found in library: {bundle_id}")
        return _library_bundle_from_row(row, status=_library_status_from_row(row))

    def get_library_manifest(self, bundle_id: str) -> Manifest:
        """Load the manifest belonging to the registered bundle, not an arbitrary path."""

        bundle = self.get_library_bundle(bundle_id)
        try:
            return _load_registered_manifest(bundle_id, Path(bundle.bundle_path))
        except (ValueError, RecursionError) as exc:
            raise AutomationError(str(exc)) from exc

    def get_registered_evidence(self, bundle_id: str) -> SelectedEvidence:
        """Read a registered bundle's selected evidence without deriving library status.

        Status derivation rehashes every declared artifact, including normalized
        audio; enriching search hits with a timestamp and title must not reread
        each matching talk's media. Integrity of the selected files is still checked.
        """

        row = self._connection.execute(
            "SELECT bundle_path FROM library_bundles WHERE bundle_id = ?", (bundle_id,)
        ).fetchone()
        if row is None:
            raise AutomationError(f"bundle not found in library: {bundle_id}")
        bundle_path = Path(str(row[0]))
        try:
            manifest = _load_registered_manifest(bundle_id, bundle_path)
            evidence = read_selected_evidence(
                bundle_path, manifest, require_manifest_integrity=True
            )
            _assert_registered_source_identity(evidence.source, manifest)
            return evidence
        except (ValueError, RecursionError) as exc:
            raise AutomationError(str(exc)) from exc

    def get_library_evidence(
        self, bundle_id: str, *, require_ready: bool = False
    ) -> SelectedEvidence:
        """Read declared, integrity-checked evidence with the registered identity policy."""

        bundle = self.get_library_bundle(bundle_id)
        if require_ready and bundle.status is not LibraryStatus.READY:
            raise AutomationError(f"bundle {bundle_id} is {bundle.status.value}; requires ready")
        try:
            manifest = _load_registered_manifest(bundle_id, Path(bundle.bundle_path))
            evidence = read_selected_evidence(
                Path(bundle.bundle_path), manifest, require_manifest_integrity=True
            )
            _assert_registered_source_identity(evidence.source, manifest)
            return evidence
        except (ValueError, RecursionError) as exc:
            raise AutomationError(str(exc)) from exc

    def _migrate(self) -> None:
        version = int(self._connection.execute("PRAGMA user_version").fetchone()[0])
        if version == STATE_SCHEMA_VERSION:
            return
        if version > STATE_SCHEMA_VERSION:
            raise AutomationError(
                f"unsupported automation state schema {version}; expected {STATE_SCHEMA_VERSION}"
            )
        if version == 0:
            self._create_schema_v2()
            self._create_schema_v3()
            return
        if version == 1:
            self._migrate_v1_to_v2()
            self._create_schema_v3()
            return
        if version == 2:
            self._create_schema_v3()
            return
        raise AutomationError(
            f"unsupported automation state schema {version}; expected {STATE_SCHEMA_VERSION}"
        )

    def _create_schema_v2(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE sources (
                id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                name TEXT NOT NULL UNIQUE,
                root_path TEXT NOT NULL,
                policy TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE source_items (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
                relative_path TEXT NOT NULL,
                absolute_path TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                mtime_ns INTEGER NOT NULL,
                present INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                UNIQUE(source_id, relative_path)
            );

            CREATE TABLE queue_items (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
                source_item_id TEXT NOT NULL REFERENCES source_items(id) ON DELETE CASCADE,
                content_sha256 TEXT NOT NULL,
                state TEXT NOT NULL,
                policy TEXT NOT NULL,
                bundle_id TEXT,
                attempts INTEGER NOT NULL DEFAULT 0,
                last_error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                UNIQUE(source_item_id, content_sha256)
            );

            CREATE TABLE library_bundles (
                bundle_id TEXT PRIMARY KEY,
                bundle_path TEXT NOT NULL UNIQUE,
                source_id TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
                source_item_id TEXT NOT NULL REFERENCES source_items(id) ON DELETE CASCADE,
                queue_item_id TEXT NOT NULL REFERENCES queue_items(id) ON DELETE CASCADE,
                created_at TEXT NOT NULL,
                kind TEXT NOT NULL DEFAULT 'bundle'
            );

            """
        )
        self._connection.execute("PRAGMA user_version = 2")
        self._connection.commit()

    def _create_segment_index(self, *, if_not_exists: bool = False) -> None:
        existence_clause = "IF NOT EXISTS " if if_not_exists else ""
        self._connection.execute(
            f"CREATE VIRTUAL TABLE {existence_clause}segment_index USING fts5("
            "bundle_id UNINDEXED, segment_id UNINDEXED, display UNINDEXED, body, literal)"
        )

    def _segment_index_layout_is_current(self) -> bool:
        columns = tuple(
            str(row[1])
            for row in self._connection.execute("PRAGMA table_info(segment_index)").fetchall()
        )
        return columns == _SEGMENT_INDEX_COLUMNS

    def _recreate_segment_index(self) -> None:
        self._connection.execute("DROP TABLE IF EXISTS segment_index")
        self._create_segment_index()

    def _create_schema_v3(self) -> None:
        """Add the retrieval index, and carry every existing bundle into it.

        Backfill is not an optimisation here, it is the whole point. A migration
        that creates an empty index leaves an upgrading user's entire archive
        unsearchable while every fixture-built test stays green, because a suite
        that only ingests new bundles never has anything old to lose.
        """

        self._create_segment_index(if_not_exists=True)
        if not self._segment_index_layout_is_current():
            self._recreate_segment_index()
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS index_signature (
                canon_version INTEGER NOT NULL,
                segmenter_version INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS indexed_bundles (
                bundle_id TEXT PRIMARY KEY,
                segments_sha256 TEXT NOT NULL
            );
            """
        )
        columns = {
            str(row[1])
            for row in self._connection.execute("PRAGMA table_info(library_bundles)").fetchall()
        }
        if "kind" not in columns:
            self._connection.execute(
                "ALTER TABLE library_bundles ADD COLUMN kind TEXT NOT NULL DEFAULT 'bundle'"
            )
        self._write_index_signature()
        self._backfill_segment_index()
        self._connection.execute(f"PRAGMA user_version = {STATE_SCHEMA_VERSION}")
        self._connection.commit()

    def _write_index_signature(self) -> None:
        signature = index_signature()
        self._connection.execute("DELETE FROM index_signature")
        self._connection.execute(
            "INSERT INTO index_signature(canon_version, segmenter_version) VALUES (?, ?)",
            (signature["canon_version"], signature["segmenter_version"]),
        )

    def index_signature_row(self) -> dict[str, int]:
        row = self._connection.execute(
            "SELECT canon_version, segmenter_version FROM index_signature"
        ).fetchone()
        if row is None:
            return {}
        return {"canon_version": int(row[0]), "segmenter_version": int(row[1])}

    def _rebuild_index_if_stale(self) -> None:
        """Rebuild when the rule that produced the index is not the current rule.

        An index written under one segmentation rule and queried under another
        fails by returning nothing, with no error anywhere. Comparing the stored
        signature is what converts that silence into work.
        """

        if (
            self.index_signature_row() == index_signature()
            and self._segment_index_layout_is_current()
        ):
            return
        self._recreate_segment_index()
        self._write_index_signature()
        self._backfill_segment_index()
        self._connection.commit()

    def refresh_changed_bundles(self) -> list[str]:
        """Reindex bundles whose transcript no longer matches what was indexed.

        A matching rule signature says the index was built the same WAY, not
        that it was built from the same CONTENT. Without this a corrected
        transcript stays searchable under its old text indefinitely while
        citation resolution reads the new file and reports different content —
        the index and the citations disagreeing, silently. It also gives a
        bundle that was unreadable during migration a later chance to be
        indexed, so a temporary failure is not permanent.
        """

        refreshed: list[str] = []
        for bundle_id, bundle_path in self._connection.execute(
            "SELECT bundle_id, bundle_path FROM library_bundles"
        ).fetchall():
            identifier = str(bundle_id)
            bundle_dir = Path(str(bundle_path))
            current = self._segments_fingerprint(identifier, bundle_dir)
            if current is None:
                # Search results are claims about the bundle as it exists now.
                # If its supporting transcript vanished or became unreadable,
                # retaining old FTS rows serves text the bundle can no longer
                # substantiate and makes reconciliation falsely look current.
                self._delete_index_rows(identifier)
                refreshed.append(identifier)
                continue
            row = self._connection.execute(
                "SELECT segments_sha256 FROM indexed_bundles WHERE bundle_id = ?", (identifier,)
            ).fetchone()
            current_digest, expected_rows = current
            indexed_rows = tuple(
                (item[0], str(item[1]), str(item[2]), str(item[3]))
                for item in self._connection.execute(
                    "SELECT segment_id, display, body, literal FROM segment_index "
                    "WHERE bundle_id = ? ORDER BY segment_id",
                    (identifier,),
                ).fetchall()
            )
            if row is not None and str(row[0]) == current_digest and indexed_rows == expected_rows:
                continue
            self._index_bundle_segments(identifier, bundle_dir)
            refreshed.append(identifier)
        if refreshed:
            self._connection.commit()
        return refreshed

    def _segments_fingerprint(
        self, bundle_id: str, bundle_dir: Path
    ) -> tuple[str, tuple[tuple[int, str, str, str], ...]] | None:
        try:
            payload, document = _read_registered_segments(
                bundle_id, bundle_dir, require_manifest_integrity=True
            )
        except (OSError, ValueError, RecursionError):
            return None
        expected_rows = [
            (
                segment.id,
                segment.text,
                operator_segment_text(segment.text),
                segment_text(segment.text),
            )
            for segment in document.root
        ]
        expected_rows.sort(key=lambda row: row[0])
        return hashlib.sha256(payload).hexdigest(), tuple(expected_rows)

    def _backfill_segment_index(self) -> None:
        for bundle_id, bundle_path in self._connection.execute(
            "SELECT bundle_id, bundle_path FROM library_bundles"
        ).fetchall():
            self._index_bundle_segments(str(bundle_id), Path(str(bundle_path)))

    def _index_bundle_segments(self, bundle_id: str, bundle_dir: Path) -> int:
        """Stage this bundle's segments for indexing; the caller commits.

        Unreadable bundles are skipped rather than raised on. An archive is
        exactly where a directory gets moved, renamed, or half-deleted, and one
        stale row must not cost the user the ability to open their library at
        all. The reconciliation query is what keeps the skip visible instead of
        silent.
        """

        try:
            payload, document = _read_registered_segments(
                bundle_id, bundle_dir, require_manifest_integrity=True
            )
        except (OSError, ValueError, RecursionError):
            self._delete_index_rows(bundle_id)
            return 0

        self._connection.execute("DELETE FROM segment_index WHERE bundle_id = ?", (bundle_id,))
        rows: list[tuple[str, int, str, str, str]] = []
        for segment in document.root:
            # `display` is the text as written. `body` retains only the original
            # token stream for operator syntax; `literal` carries the
            # expansion-aware stream used for literal candidate lookup.
            rows.append(
                (
                    bundle_id,
                    segment.id,
                    segment.text,
                    operator_segment_text(segment.text),
                    segment_text(segment.text),
                )
            )
        digest = hashlib.sha256(payload).hexdigest()
        self._connection.execute(
            "INSERT INTO indexed_bundles(bundle_id, segments_sha256) VALUES (?, ?) "
            "ON CONFLICT(bundle_id) DO UPDATE SET segments_sha256 = excluded.segments_sha256",
            (bundle_id, digest),
        )
        self._connection.executemany(
            "INSERT INTO segment_index(bundle_id, segment_id, display, body, literal) "
            "VALUES (?, ?, ?, ?, ?)",
            rows,
        )
        return len(rows)

    def search_segments(
        self, query: str, *, literal: bool = True, limit: int = 50
    ) -> list[SearchHit]:
        """Find transcript segments matching `query`.

        Literal by default: the text is treated as data, so a remembered phrase
        containing `+`, `*`, `OR`, or an unmatched quote returns results or
        nothing rather than a parser error. `literal=False` hands the string to
        FTS5's grammar, and its syntax errors are reported as such instead of
        escaping as a bare sqlite exception.
        """

        if limit <= 0:
            return []
        if not literal and any(is_unsegmented_script(character) for character in query):
            raise ValueError(
                "operator-mode search does not support CJK scripts; use literal search"
            )

        expression = literal_match_expression(query) if literal else query
        normalized_query = canonical_text(query)
        # unicode61 can erase a punctuation- or symbol-only literal completely,
        # leaving no FTS term from which to produce candidates. In that case the
        # display column is scanned in bounded pages and the same exact literal
        # confirmation is applied before the requested limit.
        tokenless_literal = (
            literal
            and bool(normalized_query)
            and not any(character.isalnum() for character in normalized_query)
        )
        try:
            if literal:
                hits: list[SearchHit] = []
                batch_size = max(50, limit)
                offset = 0
                while len(hits) < limit:
                    if tokenless_literal:
                        rows = self._connection.execute(
                            """
                            SELECT bundle_id, segment_id, display FROM segment_index
                            ORDER BY rowid LIMIT ? OFFSET ?
                            """,
                            (batch_size, offset),
                        ).fetchall()
                    else:
                        rows = self._connection.execute(
                            """
                            SELECT bundle_id, segment_id, display FROM segment_index
                            WHERE literal MATCH ?
                            ORDER BY rank, rowid LIMIT ? OFFSET ?
                            """,
                            (expression, batch_size, offset),
                        ).fetchall()
                    hits.extend(
                        SearchHit(
                            bundle_id=str(row[0]),
                            segment_id=None if row[1] is None else int(row[1]),
                            snippet=_bounded_search_snippet(str(row[2]), query),
                        )
                        for row in rows
                        if text_contains_literal(str(row[2]), query)
                    )
                    if len(rows) < batch_size:
                        break
                    offset += len(rows)
                return hits[:limit]

            rows = self._connection.execute(
                """
                SELECT bundle_id, segment_id, display FROM segment_index
                WHERE body MATCH ? ORDER BY rank LIMIT ?
                """,
                (expression, limit),
            ).fetchall()
        except sqlite3.OperationalError as exc:
            raise ValueError(f"invalid search query: {exc}") from exc
        return [
            SearchHit(
                bundle_id=str(row[0]),
                segment_id=None if row[1] is None else int(row[1]),
                snippet=_bounded_search_snippet(
                    str(row[2]), _operator_match_term(str(row[2]), query)
                ),
            )
            for row in rows
        ]

    def _bundle_segments(
        self, bundle_id: str, *, require_manifest_integrity: bool = False
    ) -> list[dict[str, Any]] | None:
        row = self._connection.execute(
            "SELECT bundle_path FROM library_bundles WHERE bundle_id = ?", (bundle_id,)
        ).fetchone()
        if row is None:
            return None
        bundle_dir = Path(str(row[0]))
        try:
            _, document = _read_registered_segments(
                bundle_id, bundle_dir, require_manifest_integrity=require_manifest_integrity
            )
        except (OSError, ValueError, RecursionError):
            return None
        return [item.model_dump(mode="json") for item in document.root]

    def resolve_anchor(self, anchor: Anchor) -> ResolvedAnchor:
        """Report which of the four states this citation is in."""

        segments = self._bundle_segments(anchor.bundle_id, require_manifest_integrity=False)
        if segments is None:
            return ResolvedAnchor(outcome=AnchorResolution.MISSING)
        return resolve_against_segments(anchor, segments)

    def cite_segment(self, bundle_id: str, segment_id: int) -> tuple[Anchor, ResolvedAnchor]:
        """Mint an anchor for a segment and resolve it in the same breath.

        Returning the resolution alongside means a caller never has to assume
        the citation it just made is good -- which matters most for the case
        where the bundle is registered but its transcript is unreadable.
        """

        segments = self._bundle_segments(bundle_id, require_manifest_integrity=True)
        if segments is None:
            raise AutomationError(f"no readable transcript for bundle: {bundle_id}")
        for segment in segments:
            if segment.get("id") == segment_id:
                anchor = make_anchor(
                    bundle_id,
                    segment_id,
                    float(segment.get("start_s") or 0.0),
                    str(segment.get("text", "")),
                )
                return anchor, resolve_against_segments(anchor, segments)
        raise AutomationError(f"segment {segment_id} not found in bundle: {bundle_id}")

    def sample_anchor_correctness(self) -> list[SamplerIssue]:
        """Check every registered bundle's segments against its declared duration.

        This is the acceptance clause "cite anchors resolve to real transcript
        segments" read as the promise it makes rather than as the lookup it is
        easy to satisfy: a segment whose timestamp lies outside the recording
        resolves perfectly and still cannot be played.
        """

        issues: list[SamplerIssue] = []
        for row in self._connection.execute(
            "SELECT bundle_id, bundle_path FROM library_bundles ORDER BY bundle_id"
        ).fetchall():
            bundle_id = str(row[0])
            segments = self._bundle_segments(bundle_id, require_manifest_integrity=False)
            if segments is None:
                issues.append(
                    SamplerIssue(
                        kind=AnchorIssue.UNREADABLE,
                        bundle_id=bundle_id,
                        segment_id=None,
                        detail="manifest or transcript segments are unreadable",
                    )
                )
                continue
            duration: float | None = None
            try:
                manifest = _load_registered_manifest(bundle_id, Path(str(row[1])))
                duration = manifest.source.duration_s
            except (OSError, ValueError, RecursionError):
                duration = None
            issues.extend(sample_segment_timings(bundle_id, segments, duration))
        return issues

    def index_synthetic_segment(self, bundle_id: str, segment_id: int, text: str) -> None:
        """Index one segment directly. For fixtures that need a script the audio lacks.

        Public and plainly named so its use in a test reads as fixture
        construction rather than as a claim that some real bundle contained
        Japanese -- the synthetic-fixture policy applies to what a test asserts
        as much as to what it stores.
        """

        self._connection.execute(
            "INSERT INTO segment_index(bundle_id, segment_id, display, body, literal) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                bundle_id,
                segment_id,
                text,
                operator_segment_text(text),
                segment_text(text),
            ),
        )
        self._connection.commit()

    def indexed_segment_count(self, *, bundle_id: str | None = None) -> int:
        if bundle_id is None:
            row = self._connection.execute("SELECT COUNT(*) FROM segment_index").fetchone()
        else:
            row = self._connection.execute(
                "SELECT COUNT(*) FROM segment_index WHERE bundle_id = ?", (bundle_id,)
            ).fetchone()
        return int(row[0])

    def unindexed_bundle_ids(self) -> list[str]:
        """Registered bundles with no index rows: the reconciliation invariant.

        Checkable at runtime rather than only in a test, because the states that
        break it -- an interrupted ingest, a bundle deleted from disk -- happen
        on a user's machine and not in a fixture.
        """

        return [
            str(row[0])
            for row in self._connection.execute(
                """
                SELECT bundle_id FROM library_bundles
                WHERE bundle_id NOT IN (SELECT DISTINCT bundle_id FROM segment_index)
                ORDER BY bundle_id
                """
            ).fetchall()
        ]

    def _delete_index_rows(self, bundle_id: str) -> None:
        self._connection.execute("DELETE FROM segment_index WHERE bundle_id = ?", (bundle_id,))
        self._connection.execute("DELETE FROM indexed_bundles WHERE bundle_id = ?", (bundle_id,))

    def forget_library_bundle(self, bundle_id: str) -> None:
        self._delete_index_rows(bundle_id)
        self._connection.execute("DELETE FROM library_bundles WHERE bundle_id = ?", (bundle_id,))
        self._connection.commit()

    def _migrate_v1_to_v2(self) -> None:
        # `ALTER TABLE` and `PRAGMA user_version` do not share a transaction, so a
        # process killed mid-migration leaves columns present at user_version 1. Each
        # step is therefore guarded and re-runnable, and the v1 file is copied aside
        # first so the pre-migration bytes survive a failed upgrade.
        self._backup_v1_store()
        for table in ("source_items", "queue_items"):
            if self._table_has_column(table, "metadata_json"):
                continue
            try:
                self._connection.execute(
                    f"ALTER TABLE {table} ADD COLUMN metadata_json TEXT NOT NULL DEFAULT '{{}}'"
                )
            except sqlite3.OperationalError as exc:
                # A concurrent migration can add the column between our check
                # and this statement; that outcome is the migration's goal, so
                # tolerate it rather than failing the second process.
                if "duplicate column name" not in str(exc):
                    raise
        # Persist 2, not the current constant. Symbolically referencing
        # STATE_SCHEMA_VERSION meant bumping it to 3 made this step claim a
        # version whose tables it had not created: an interruption between here
        # and v3 creation left a store that every later open would fail to read.
        self._connection.execute("PRAGMA user_version = 2")
        self._connection.commit()

    def _table_has_column(self, table: str, column: str) -> bool:
        rows = self._connection.execute(f"PRAGMA table_info({table})").fetchall()
        return any(cast(str, row["name"]) == column for row in rows)

    def _backup_v1_store(self) -> None:
        if not self.path.is_file():
            return
        backup = self.path.with_name(f"{self.path.name}.v1.bak")
        if backup.exists():
            return
        # SQLite's backup API captures the committed database regardless of
        # journal mode; a plain file copy can miss changes still living in an
        # adjacent -wal file — the exact case where the backup matters.
        #
        # The backup is written to a temporary name and published with os.replace so
        # an interrupted run cannot leave a partial file that the existence check
        # above would later trust as a complete pre-migration copy. The name carries
        # a per-attempt unique component (as atomic_write_text does): a fixed name is
        # shared by concurrent migrations of the same store, where one attempt's
        # cleanup would unlink the other's in-progress copy.
        temporary = self.path.with_name(f".{backup.name}.{uuid.uuid4().hex}.tmp")
        # sqlite3.connect would create the file under the process umask, so the
        # copy would hold the database's bytes under broader permissions for the
        # length of the backup (and past a SIGKILL). Create it with the source
        # database's own mode first, before any bytes are written into it.
        mode = self.path.stat().st_mode & 0o777
        os.close(os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, mode))
        try:
            # The umask also masks os.open's mode, so restate it explicitly.
            os.chmod(temporary, mode)
            destination = sqlite3.connect(temporary)
            try:
                self._connection.backup(destination)
            finally:
                destination.close()
            # Publish only if no backup exists yet: os.link fails on an
            # existing destination, unlike os.replace, so a backup that a
            # concurrent migration published after our existence check is
            # never overwritten with a possibly post-migration snapshot.
            with contextlib.suppress(FileExistsError):
                os.link(temporary, backup)
            temporary.unlink(missing_ok=True)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    def _list_source_items(self, source_id: str, *, present_only: bool) -> list[SourceItem]:
        if present_only:
            rows = self._connection.execute(
                "SELECT * FROM source_items WHERE source_id = ? AND present = 1",
                (source_id,),
            ).fetchall()
        else:
            rows = self._connection.execute(
                "SELECT * FROM source_items WHERE source_id = ?",
                (source_id,),
            ).fetchall()
        return [source_item_from_row(row) for row in rows]

    def _upsert_source_item(self, item: SourceItem, *, created_at: str | None) -> None:
        self._connection.execute(
            """
            INSERT INTO source_items(
                id, source_id, relative_path, absolute_path, sha256, size_bytes, mtime_ns,
                present, created_at, updated_at, metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_id, relative_path) DO UPDATE SET
                absolute_path = excluded.absolute_path,
                sha256 = excluded.sha256,
                size_bytes = excluded.size_bytes,
                mtime_ns = excluded.mtime_ns,
                present = excluded.present,
                updated_at = excluded.updated_at,
                metadata_json = excluded.metadata_json
            """,
            (
                item.id,
                item.source_id,
                item.relative_path,
                item.absolute_path,
                item.sha256,
                item.size_bytes,
                item.mtime_ns,
                1 if item.present else 0,
                created_at or item.created_at,
                item.updated_at,
                metadata_to_json(item.metadata),
            ),
        )

    def _enqueue_if_review(self, source: SourceRecord, item: SourceItem) -> QueueItem | None:
        if source.policy is not SourcePolicy.REVIEW:
            return None
        if _is_placeholder_item(item):
            return None
        queue_item_id = make_queue_item_id(item.id, item.sha256)
        now = now_timestamp()
        cursor = self._connection.execute(
            """
            INSERT OR IGNORE INTO queue_items(
                id, source_id, source_item_id, content_sha256, state, policy,
                bundle_id, attempts, last_error, created_at, updated_at, metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, NULL, 0, NULL, ?, ?, ?)
            """,
            (
                queue_item_id,
                source.id,
                item.id,
                item.sha256,
                QueueState.DISCOVERED.value,
                source.policy.value,
                now,
                now,
                metadata_to_json(item.metadata),
            ),
        )
        if cursor.rowcount == 0:
            return None
        return self.get_queue_item(queue_item_id)

    def _set_queue_state(
        self,
        queue_item_id: str,
        state: QueueState,
        *,
        bundle_id: str | None = None,
        clear_error: bool = False,
        guarded_action: str | None = None,
    ) -> QueueItem:
        queue_item = self._apply_queue_state(
            queue_item_id,
            state,
            bundle_id=bundle_id,
            clear_error=clear_error,
            guarded_action=guarded_action,
        )
        self._connection.commit()
        return self.get_queue_item(queue_item.id)

    def _apply_queue_state(
        self,
        queue_item_id: str,
        state: QueueState,
        *,
        bundle_id: str | None = None,
        clear_error: bool = False,
        guarded_action: str | None = None,
    ) -> QueueItem:
        """Stage a queue-state transition without committing; the caller commits.

        `guarded_action` names a user verb whose legal source states the FSM table
        pins. The guard lives in the UPDATE's WHERE clause so the check and the write
        are one statement: a read-then-write pair lets a concurrent writer slip an
        illegal source state in between and have it overwritten.
        """

        existing = self.get_queue_item(queue_item_id)
        fsm_guard = ""
        parameters: list[object] = [
            state.value,
            bundle_id,
            1 if clear_error else 0,
            now_timestamp(),
            existing.id,
        ]
        if guarded_action is not None:
            legal_sources = LEGAL_QUEUE_TRANSITION_SOURCE_VALUES[guarded_action]
            placeholders = ", ".join("?" for _ in legal_sources)
            fsm_guard = f" AND state IN ({placeholders})"
            parameters.extend(legal_sources)
        cursor = self._connection.execute(
            f"""
            UPDATE queue_items
            SET state = ?, bundle_id = COALESCE(?, bundle_id),
                last_error = CASE WHEN ? THEN NULL ELSE last_error END,
                updated_at = ?
            WHERE id = ?{fsm_guard}
            """,
            parameters,
        )
        if guarded_action is not None and cursor.rowcount == 0:
            self._reject_guarded_transition_conflict(queue_item_id, guarded_action)
        return existing

    def _record_failed_queue_item(self, queue_item_id: str, message: str) -> None:
        self._apply_failed_queue_item(queue_item_id, message)
        self._connection.commit()

    def _apply_failed_queue_item(self, queue_item_id: str, message: str) -> None:
        """Stage a failure transition without committing; the caller commits."""

        queue_item = self.get_queue_item(queue_item_id)
        self._connection.execute(
            """
            UPDATE queue_items
            SET state = ?, attempts = ?, last_error = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                QueueState.FAILED.value,
                queue_item.attempts + 1,
                message,
                now_timestamp(),
                queue_item.id,
            ),
        )
        self._connection.commit()

    def _record_unsupported_queue_item(self, queue_item_id: str, message: str) -> None:
        """Record a terminal unsupported-by-design outcome without spending an attempt.

        The write is conditioned on the row still being APPROVED: a concurrent
        skip or retry that committed after our approved read must not be
        overwritten with an irreversible terminal state. On a zero-row match the
        newer operator transition stands and no terminal state is recorded.
        """

        queue_item = self.get_queue_item(queue_item_id)
        cursor = self._connection.execute(
            """
            UPDATE queue_items
            SET state = ?, last_error = ?, updated_at = ?
            WHERE id = ? AND state = ?
            """,
            (
                QueueState.UNSUPPORTED.value,
                message,
                now_timestamp(),
                queue_item.id,
                QueueState.APPROVED.value,
            ),
        )
        if cursor.rowcount == 0:
            self._connection.rollback()
            return
        self._connection.commit()

    def _record_library_bundle(
        self,
        bundle_dir: Path,
        source: SourceRecord,
        source_item: SourceItem,
        queue_item: QueueItem,
    ) -> LibraryRecordOutcome:
        """Stage the library record without committing; reports what it changed.

        The caller commits, so completion state and library record land together,
        and keeps the outcome so a later failure can restore the prior row.
        """

        manifest = Manifest.load(bundle_dir)
        existing = self._existing_library_bundle(manifest.bundle_id)
        values = (
            str(bundle_dir.resolve()),
            source.id,
            source_item.id,
            queue_item.id,
        )
        if existing is None:
            self._connection.execute(
                """
                INSERT INTO library_bundles(
                    bundle_id, bundle_path, source_id, source_item_id, queue_item_id, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    manifest.bundle_id,
                    *values,
                    now_timestamp(),
                ),
            )
            self._index_bundle_segments(manifest.bundle_id, bundle_dir)
            return LibraryRecordOutcome(inserted=True, previous=None)
        if existing.queue_item_id == queue_item.id:
            self._connection.execute(
                """
                UPDATE library_bundles
                SET bundle_path = ?, source_id = ?, source_item_id = ?, queue_item_id = ?
                WHERE bundle_id = ?
                """,
                (*values, manifest.bundle_id),
            )
            return LibraryRecordOutcome(inserted=False, previous=existing)
        raise AutomationError(
            "bundle id already exists for different source provenance; "
            "duplicate-content multi-source provenance is not implemented"
        )

    def _existing_library_bundle(self, bundle_id: str) -> LibraryBundle | None:
        existing = self._connection.execute(
            "SELECT * FROM library_bundles WHERE bundle_id = ?",
            (bundle_id,),
        ).fetchone()
        if existing is None:
            return None
        return _library_bundle_from_row(existing)


STATE_SCHEMA_VERSION = 3

DEFAULT_STATE_PATH = Path(".lectern") / "state.sqlite"

UPGRADABLE_STATE_SCHEMA_VERSIONS = frozenset({0, 1, 2, STATE_SCHEMA_VERSION})


@dataclass(frozen=True)
class StateStorePreflight:
    path: str
    exists: bool
    writable_location: bool
    schema_version: int | None
    error: str | None

    @property
    def ok(self) -> bool:
        return self.writable_location and self.error is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "exists": self.exists,
            "writable_location": self.writable_location,
            "schema_version": self.schema_version,
            "error": self.error,
            "ok": self.ok,
        }


def preflight_state_store(path: Path = DEFAULT_STATE_PATH) -> StateStorePreflight:
    resolved = path.resolve()
    writable_location = _nearest_existing_parent_is_writable(resolved)
    exists = resolved.exists()
    schema_version: int | None = None
    error: str | None = None
    if exists:
        writable_location = writable_location and os.access(resolved, os.W_OK)
        try:
            # A `sqlite3.Connection` context manager commits or rolls back; it
            # does not close. Preflight is a read-only probe, so the handle has
            # to be released explicitly rather than left to refcounting.
            connection = sqlite3.connect(f"file:{resolved}?mode=ro", uri=True)
            try:
                schema_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            finally:
                connection.close()
        except sqlite3.Error as exc:
            error = str(exc)
        if schema_version not in (None, *UPGRADABLE_STATE_SCHEMA_VERSIONS):
            error = (
                f"unsupported automation state schema {schema_version}; "
                f"expected {STATE_SCHEMA_VERSION}"
            )
    return StateStorePreflight(
        path=str(path),
        exists=exists,
        writable_location=writable_location,
        schema_version=schema_version,
        error=error,
    )


def _source_from_row(row: sqlite3.Row) -> SourceRecord:
    return SourceRecord(
        id=cast(str, row["id"]),
        kind=SourceKind(cast(str, row["kind"])),
        name=cast(str, row["name"]),
        root_path=cast(str, row["root_path"]),
        policy=SourcePolicy(cast(str, row["policy"])),
        created_at=cast(str, row["created_at"]),
        updated_at=cast(str, row["updated_at"]),
    )


def source_item_from_row(row: sqlite3.Row) -> SourceItem:
    return SourceItem(
        id=cast(str, row["id"]),
        source_id=cast(str, row["source_id"]),
        relative_path=cast(str, row["relative_path"]),
        absolute_path=cast(str, row["absolute_path"]),
        sha256=cast(str, row["sha256"]),
        size_bytes=cast(int, row["size_bytes"]),
        mtime_ns=cast(int, row["mtime_ns"]),
        present=bool(row["present"]),
        created_at=cast(str, row["created_at"]),
        updated_at=cast(str, row["updated_at"]),
        metadata=_metadata_from_row(row),
    )


def queue_from_row(row: sqlite3.Row) -> QueueItem:
    return QueueItem(
        id=cast(str, row["id"]),
        source_id=cast(str, row["source_id"]),
        source_item_id=cast(str, row["source_item_id"]),
        content_sha256=cast(str, row["content_sha256"]),
        state=QueueState(cast(str, row["state"])),
        policy=SourcePolicy(cast(str, row["policy"])),
        bundle_id=cast(str | None, row["bundle_id"]),
        attempts=cast(int, row["attempts"]),
        last_error=cast(str | None, row["last_error"]),
        created_at=cast(str, row["created_at"]),
        updated_at=cast(str, row["updated_at"]),
        metadata=_metadata_from_row(row),
    )


def _library_bundle_from_row(
    row: sqlite3.Row, *, status: LibraryStatus = LibraryStatus.READY
) -> LibraryBundle:
    return LibraryBundle(
        bundle_id=cast(str, row["bundle_id"]),
        bundle_path=cast(str, row["bundle_path"]),
        source_id=cast(str, row["source_id"]),
        source_item_id=cast(str, row["source_item_id"]),
        queue_item_id=cast(str, row["queue_item_id"]),
        created_at=cast(str, row["created_at"]),
        kind=_library_kind_from_row(row),
        status=status,
    )


def _library_status_from_row(row: sqlite3.Row) -> LibraryStatus:
    """Project the joined queue row and current manifest into library status."""

    try:
        queue_state = QueueState(str(row["library_queue_state"]))
        queue_sha256 = str(row["library_queue_sha256"])
        source_sha256 = str(row["library_source_sha256"])
    except (IndexError, KeyError, ValueError):
        return LibraryStatus.INCOMPLETE

    bundle_id = str(row["bundle_id"])
    bundle_path = Path(str(row["bundle_path"]))
    try:
        manifest = _load_registered_manifest(bundle_id, bundle_path)
    except (OSError, ValueError, RecursionError):
        return derive_library_status(
            queue_state, (), manifest_available=False, source_changed=bundle_path.is_symlink()
        )

    materialized_stages = [
        record.state.value
        for record in manifest.stages.values()
        if record.state is not StageState.PENDING
        or record.started is not None
        or record.finished is not None
        or record.outputs
        or record.error is not None
    ]
    try:
        evidence = read_selected_evidence(bundle_path, manifest, require_manifest_integrity=True)
        _assert_registered_source_identity(evidence.source, manifest)
    except (OSError, ValueError, RecursionError):
        return derive_library_status(
            queue_state,
            materialized_stages,
            source_changed=True,
        )
    return derive_library_status(
        queue_state,
        materialized_stages,
        source_changed=(
            queue_sha256 != source_sha256
            or not _manifest_outputs_are_materialized(bundle_path, manifest)
        ),
    )


def _load_registered_manifest(bundle_id: str, bundle_path: Path) -> Manifest:
    manifest = Manifest.load(bundle_path)
    if manifest.bundle_id != bundle_id:
        raise ValueError("registered bundle id does not match manifest bundle id")
    return manifest


def _assert_registered_source_identity(source: SourceDocument, manifest: Manifest) -> None:
    if (
        source.source.kind != manifest.source.kind
        or source.source.ref != manifest.source.ref
        or source.source.bytes != manifest.source.bytes
    ):
        raise ValueError("registered source identity does not match manifest source")
    if manifest.source.kind.value == "local" and (
        manifest.source.ref != f"sha256:{source.sha256}" or manifest.source.bytes != source.bytes
    ):
        raise ValueError("registered source content identity does not match manifest source")


def _read_registered_segments(
    bundle_id: str,
    bundle_path: Path,
    *,
    require_manifest_integrity: bool = False,
) -> tuple[bytes, TranscriptSegmentsDocument]:
    manifest = _load_registered_manifest(bundle_id, bundle_path)
    evidence = read_selected_evidence(
        bundle_path, manifest, require_manifest_integrity=require_manifest_integrity
    )
    _assert_registered_source_identity(evidence.source, manifest)
    return evidence.segments_payload, evidence.segments


def _manifest_outputs_are_materialized(bundle_path: Path, manifest: Manifest) -> bool:
    try:
        if bundle_path.is_symlink():
            return False
        resolved_bundle = bundle_path.resolve(strict=True)
    except OSError:
        return False
    for stage in manifest.stages.values():
        for output in stage.outputs:
            relative = Path(output.path)
            if relative.is_absolute() or ".." in relative.parts:
                return False
            candidate = bundle_path / relative
            try:
                resolved_candidate = candidate.resolve(strict=True)
                if (
                    _has_symlink_component(bundle_path, relative)
                    or not resolved_candidate.is_relative_to(resolved_bundle)
                    or not resolved_candidate.is_file()
                ):
                    return False
                digest, size = digest_and_size(resolved_candidate)
            except OSError:
                return False
            if size != output.bytes or digest != output.sha256:
                return False
    return True


def _has_symlink_component(bundle_path: Path, relative: Path) -> bool:
    candidate = bundle_path
    for part in relative.parts:
        candidate /= part
        if candidate.is_symlink():
            return True
    return False


def _bounded_search_snippet(display: str, query: str, *, limit: int = 240) -> str:
    if len(display) <= limit:
        return display
    content_limit = limit - 2
    match_at = _canonical_match_offset(display, query)
    start = 0 if match_at < 0 else max(0, match_at - content_limit // 2)
    end = min(len(display), start + content_limit)
    if end == len(display):
        start = max(0, end - content_limit)
    return f"{'…' if start else ''}{display[start:end]}{'…' if end < len(display) else ''}"


def _canonical_match_offset(display: str, query: str) -> int:
    canonical_parts: list[str] = []
    raw_offsets: list[int] = []
    for token_match in re.finditer(r"\S+", display):
        if canonical_parts:
            canonical_parts.append(" ")
            raw_offsets.append(token_match.start())
        normalized, token_offsets = _canonical_token_offsets(
            token_match.group(), token_match.start()
        )
        canonical_parts.append(normalized)
        raw_offsets.extend(token_offsets)
    match_at = "".join(canonical_parts).find(canonical_text(query).casefold())
    return -1 if match_at < 0 else raw_offsets[match_at]


def _canonical_token_offsets(raw_token: str, raw_start: int) -> tuple[str, list[int]]:
    normalized_parts: list[str] = []
    offsets: list[int] = []
    index = 0
    while index < len(raw_token):
        end = _normalization_cluster_end(raw_token, index)
        normalized = unicodedata.normalize("NFC", raw_token[index:end]).casefold()
        normalized_parts.append(normalized)
        offsets.extend([raw_start + index] * len(normalized))
        index = end
    return "".join(normalized_parts), offsets


def _normalization_cluster_end(raw_token: str, start: int) -> int:
    """Return the end of one independently NFC-normalizable source cluster."""

    end = start + 1
    code_point = ord(raw_token[start])
    # Hangul NFC composes modern L + V (+ T) Jamo even though every component
    # has combining class zero, so a combining-mark-only cluster rule splits
    # exactly the decomposed Korean text canonicalization is meant to preserve.
    if 0x1100 <= code_point <= 0x1112 and end < len(raw_token):
        if 0x1161 <= ord(raw_token[end]) <= 0x1175:
            end += 1
            if end < len(raw_token) and 0x11A8 <= ord(raw_token[end]) <= 0x11C2:
                end += 1
    elif (
        0xAC00 <= code_point <= 0xD7A3
        and (code_point - 0xAC00) % 28 == 0
        and end < len(raw_token)
        and 0x11A8 <= ord(raw_token[end]) <= 0x11C2
    ):
        end += 1
    while end < len(raw_token) and unicodedata.combining(raw_token[end]):
        end += 1
    return end


_OPERATOR_TERM = re.compile(r'"((?:""|[^"])*)"|([\w]+)')
_FTS_OPERATORS = {"AND", "OR", "NOT", "NEAR"}
_FTS_COLUMN_SELECTOR = re.compile(
    r"(?i)(?<!\w)(?:"
    r"\{(?:\s*(?:body|display|bundle_id|segment_id)\s*)+\}"
    r"|(?:body|display|bundle_id|segment_id)"
    r")\s*:"
)


def _operator_match_term(display: str, query: str) -> str:
    folded_display = display.casefold()
    without_selectors = _FTS_COLUMN_SELECTOR.sub("", query)
    for phrase, word in _OPERATOR_TERM.findall(without_selectors):
        candidate = phrase.replace('""', '"') if phrase else word
        if candidate.upper() in _FTS_OPERATORS:
            continue
        if candidate.casefold() in folded_display:
            return candidate
    return query


def _library_kind_from_row(row: sqlite3.Row) -> LibraryKind:
    """Read the discriminator, tolerating one this build does not know.

    A row written by a newer version must stay listable by an older one.
    Refusing an unrecognised value would turn a field reserved for forward
    compatibility into the thing that breaks it.
    """

    try:
        raw = row["kind"]
    except (IndexError, KeyError):
        return LibraryKind.BUNDLE
    try:
        return LibraryKind(str(raw))
    except ValueError:
        return LibraryKind.BUNDLE


def _metadata_from_row(row: sqlite3.Row) -> dict[str, Any]:
    try:
        raw_value = row["metadata_json"]
    except (IndexError, KeyError):
        return {}
    raw = cast(str, raw_value)
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return cast(dict[str, Any], value) if isinstance(value, dict) else {}


def _scan_metadata_from_adapter(adapter: SourceAdapter) -> dict[str, Any]:
    if isinstance(adapter, ScanMetadataProvider):
        return adapter.scan_metadata
    return {}


def _scan_reports_truncation(scan_metadata: dict[str, Any]) -> bool:
    quota = scan_metadata.get("quota")
    if isinstance(quota, dict) and cast(dict[str, Any], quota).get("truncated_by_max_pages"):
        return True
    return bool(scan_metadata.get("truncated_by_max_pages"))


def _dedupe_by_relative_path(items: Iterable[SourceItem]) -> list[SourceItem]:
    """Collapse repeats of one identity within a single discovery; first occurrence wins."""

    seen: set[str] = set()
    deduped: list[SourceItem] = []
    for item in items:
        if item.relative_path in seen:
            continue
        seen.add(item.relative_path)
        deduped.append(item)
    return deduped


def _is_placeholder_item(item: SourceItem) -> bool:
    video = item.metadata.get("video")
    if not isinstance(video, dict):
        return False
    return bool(cast(dict[str, Any], video).get("placeholder"))


def _join_states(values: Sequence[str]) -> str:
    """Render a legal-source set for an error message: `failed`, `a or b`, `a, b, or c`."""

    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} or {values[1]}"
    return f"{', '.join(values[:-1])}, or {values[-1]}"


def _nearest_existing_parent_is_writable(path: Path) -> bool:
    candidate = path.parent
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate.is_dir() and os.access(candidate, os.W_OK)
