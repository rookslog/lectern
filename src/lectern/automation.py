"""Source registry, discovery queue, and library state.

The automation spine records source discovery policy, queues reviewable items,
and bridges approved local items into the existing bundle ingest path. External
adapters may discover metadata, but local-source media artifacts stay local
unless a future stage adds explicit per-item consent.

This module is the spine's composition root and its stable import surface. The
parts live next door — `lectern.records` (vocabulary), `lectern.state` (the
SQLite store), `lectern.sources` (discovery adapters), `lectern.provenance`
(bundle annotation) — and `AutomationState` is where they meet: it is the only
place that both holds the store and drives `lectern.ingest`.
"""

from __future__ import annotations

import shutil
import sqlite3
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from typing import Any

from lectern.bundle import Manifest
from lectern.diarization import DEFAULT_DIARIZE_MODELS
from lectern.ingest import (
    BundleExistsError,
    IngestError,
    IngestResult,
    prepare_local_ingest,
)
from lectern.ingest import (
    can_plan_local_bundle_id as can_plan_local_bundle_id,
)
from lectern.ingest import (
    ingest_local as ingest_local,
)
from lectern.ingest import (
    plan_local_bundle_id as plan_local_bundle_id,
)
from lectern.keyframes import DEFAULT_FRAME_INTERVAL, DEFAULT_MAX_FRAMES
from lectern.provenance import (
    PROVENANCE_KEYS,
    attach_provenance_to_bundle,
    bundle_provenance_needs_repair,
)
from lectern.records import (
    LEGAL_QUEUE_TRANSITION_SOURCE_VALUES,
    LEGAL_QUEUE_TRANSITION_SOURCES,
    TERMINAL_QUEUE_STATE_VALUES,
    TERMINAL_QUEUE_STATES,
    AutomationError,
    LibraryBundle,
    LibraryRecordOutcome,
    LibraryStatus,
    QueueItem,
    QueueState,
    ScanDelta,
    ScanMetadataProvider,
    SourceAdapter,
    SourceItem,
    SourceKind,
    SourcePolicy,
    SourceRecord,
    digest_and_size,
    make_queue_item_id,
    make_source_id,
    make_source_item_id,
    metadata_to_json,
    now_timestamp,
)
from lectern.sources import default_source_adapter
from lectern.sources.local import (
    EXCLUDED_SCAN_DIR_NAMES,
    EXCLUDED_SCAN_DIR_PREFIXES,
    MEDIA_EXTENSIONS,
    LocalFolderAdapter,
    SourcePreflight,
    approval_digest_and_media_size,
    is_bundle_output_path,
    iter_local_media_files,
    preflight_local_folder,
)
from lectern.sources.youtube import (
    DEFAULT_YOUTUBE_API_KEY_ENV,
    YOUTUBE_METADATA_ONLY_ERROR,
    YOUTUBE_PLACEHOLDER_TITLES,
    YOUTUBE_PLAYLIST_ITEMS_ENDPOINT,
    YOUTUBE_PLAYLIST_PAGE_SIZE,
    YOUTUBE_PLAYLIST_PARTS,
    YOUTUBE_PLAYLIST_QUOTA_UNITS_PER_PAGE,
    YOUTUBE_REQUEST_TIMEOUT_S,
    HttpGet,
    YouTubeAPIError,
    YouTubePlaylistAdapter,
    YouTubePreflight,
    normalize_youtube_playlist_id,
    preflight_youtube_playlist,
    urllib_get,
)
from lectern.state import (
    DEFAULT_STATE_PATH,
    STATE_SCHEMA_VERSION,
    UPGRADABLE_STATE_SCHEMA_VERSIONS,
    AutomationStateStore,
    StateStorePreflight,
    preflight_state_store,
    queue_from_row,
    source_item_from_row,
)

# Re-exported so `lectern.automation` remains the spine's single import surface:
# every name a consumer could import from this module before the package split
# is still importable from it. Listing them keeps the imports above live for the
# linter and states the surface in one place.
__all__ = [
    "DEFAULT_STATE_PATH",
    "DEFAULT_YOUTUBE_API_KEY_ENV",
    "EXCLUDED_SCAN_DIR_NAMES",
    "EXCLUDED_SCAN_DIR_PREFIXES",
    "LEGAL_QUEUE_TRANSITION_SOURCES",
    "LEGAL_QUEUE_TRANSITION_SOURCE_VALUES",
    "MEDIA_EXTENSIONS",
    "PROVENANCE_KEYS",
    "STATE_SCHEMA_VERSION",
    "TERMINAL_QUEUE_STATES",
    "TERMINAL_QUEUE_STATE_VALUES",
    "UPGRADABLE_STATE_SCHEMA_VERSIONS",
    "YOUTUBE_METADATA_ONLY_ERROR",
    "YOUTUBE_PLACEHOLDER_TITLES",
    "YOUTUBE_PLAYLIST_ITEMS_ENDPOINT",
    "YOUTUBE_PLAYLIST_PAGE_SIZE",
    "YOUTUBE_PLAYLIST_PARTS",
    "YOUTUBE_PLAYLIST_QUOTA_UNITS_PER_PAGE",
    "YOUTUBE_REQUEST_TIMEOUT_S",
    "AutomationError",
    "AutomationState",
    "AutomationStateStore",
    "HttpGet",
    "IngestError",
    "IngestResult",
    "LibraryBundle",
    "LibraryStatus",
    "LibraryRecordOutcome",
    "LocalFolderAdapter",
    "QueueItem",
    "QueueState",
    "ScanDelta",
    "ScanMetadataProvider",
    "SourceAdapter",
    "SourceItem",
    "SourceKind",
    "SourcePolicy",
    "SourcePreflight",
    "SourceRecord",
    "StateStorePreflight",
    "YouTubeAPIError",
    "YouTubePlaylistAdapter",
    "YouTubePreflight",
    "approval_digest_and_media_size",
    "attach_provenance_to_bundle",
    "bundle_provenance_needs_repair",
    "default_source_adapter",
    "digest_and_size",
    "is_bundle_output_path",
    "iter_local_media_files",
    "make_queue_item_id",
    "make_source_id",
    "make_source_item_id",
    "metadata_to_json",
    "normalize_youtube_playlist_id",
    "now_timestamp",
    "open_state",
    "preflight_local_folder",
    "preflight_state_store",
    "preflight_youtube_playlist",
    "state_summary",
    "urllib_get",
]


class AutomationState(AutomationStateStore):
    """The state store plus the ingest pipeline it feeds.

    Orchestration lives here rather than in `lectern.state` so the persistence
    layer does not own the pipeline: the store records what happened, and this
    class is what makes it happen. `attach_provenance_to_bundle` is resolved
    through this module's globals, which is also what lets a caller substitute
    it when exercising the rollback paths below.
    """

    def _default_adapter(self, source: SourceRecord) -> SourceAdapter:
        """Map a source kind to a concrete adapter.

        This is the composition root's job, and keeping it here is what lets
        `lectern.state` be imported without loading an HTTP transport.
        """

        return default_source_adapter(source)

    def add_youtube_playlist_source(
        self,
        name: str,
        playlist: str,
        policy: SourcePolicy = SourcePolicy.REVIEW,
    ) -> SourceRecord:
        """Register a playlist source from a user-supplied playlist id or URL.

        Normalization is provider knowledge, so it happens here and the store is
        handed the identity it should persist.
        """

        return super().add_youtube_playlist_source(
            name, normalize_youtube_playlist_id(playlist), policy
        )

    def _ensure_one_shot_source(self, registry_path: Path) -> SourceRecord:
        source = registry_path
        source_id = make_source_id(SourceKind.ONE_SHOT.value, str(source))
        now = now_timestamp()
        self._connection.execute(
            """
            INSERT INTO sources(id, kind, name, root_path, policy, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET updated_at = excluded.updated_at
            """,
            (
                source_id,
                SourceKind.ONE_SHOT.value,
                f"one-shot:{source.name}:{source_id[-8:]}",
                str(source),
                SourcePolicy.REVIEW.value,
                now,
                now,
            ),
        )
        self._connection.commit()
        return self.get_source(source_id)

    def _ensure_one_shot_item(
        self, source: SourceRecord, registry_path: Path, *, digest: str, size: int, mtime_ns: int
    ) -> SourceItem:
        path = registry_path
        now = now_timestamp()
        item = SourceItem(
            id=make_source_item_id(source.id, path.name),
            source_id=source.id,
            relative_path=path.name,
            absolute_path=str(path),
            sha256=digest,
            size_bytes=size,
            mtime_ns=mtime_ns,
            present=True,
            created_at=now,
            updated_at=now,
        )
        old = self._connection.execute(
            "SELECT * FROM source_items WHERE id = ?",
            (item.id,),
        ).fetchone()
        self._upsert_source_item(
            item,
            created_at=source_item_from_row(old).created_at if old is not None else None,
        )
        self._connection.commit()
        return self.get_source_item(item.id)

    def _ensure_one_shot_queue(self, source: SourceRecord, item: SourceItem) -> QueueItem:
        queue_item_id = make_queue_item_id(item.id, item.sha256)
        existing = self._connection.execute(
            "SELECT * FROM queue_items WHERE id = ?",
            (queue_item_id,),
        ).fetchone()
        if existing is not None:
            queue_item = queue_from_row(existing)
            if queue_item.state is QueueState.COMPLETED:
                return queue_item
            return self._set_queue_state(queue_item.id, QueueState.APPROVED, clear_error=True)

        now = now_timestamp()
        self._connection.execute(
            """
            INSERT INTO queue_items(
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
                QueueState.APPROVED.value,
                source.policy.value,
                now,
                now,
                metadata_to_json(item.metadata),
            ),
        )
        self._connection.commit()
        return self.get_queue_item(queue_item_id)

    def ingest_queue_item(
        self,
        queue_item_id: str,
        output_root: Path,
        *,
        transcriber_command: str | None = None,
        asr_model: str = "small",
        allow_model_download: bool = False,
        language: str | None = None,
        frames: bool = False,
        frame_interval: float = DEFAULT_FRAME_INTERVAL,
        max_frames: int = DEFAULT_MAX_FRAMES,
        diarize: bool = False,
        speakers: int | None = None,
        diarize_models: Path = DEFAULT_DIARIZE_MODELS,
    ) -> IngestResult:
        queue_item = self.get_queue_item(queue_item_id)
        if queue_item.state is not QueueState.APPROVED:
            raise AutomationError("queue item requires explicit approval before ingest")
        source = self.get_source(queue_item.source_id)
        source_item = self.get_source_item(queue_item.source_item_id)
        if source.kind is SourceKind.YOUTUBE_PLAYLIST:
            self._record_unsupported_queue_item(queue_item.id, YOUTUBE_METADATA_ONLY_ERROR)
            raise AutomationError(YOUTUBE_METADATA_ONLY_ERROR)
        source_path = Path(source_item.absolute_path)
        root = Path(source.root_path).resolve() if source.kind is SourceKind.LOCAL_FOLDER else None
        with ExitStack() as inputs:
            planned_bundle_id: str | None = None
            try:
                prepared = inputs.enter_context(
                    prepare_local_ingest(
                        source_path,
                        transcriber_command=transcriber_command,
                        asr_model=asr_model,
                        allow_model_download=allow_model_download,
                        language=language,
                        frames=frames,
                        frame_interval=frame_interval,
                        max_frames=max_frames,
                        diarize=diarize,
                        speakers=speakers,
                        diarize_models=diarize_models,
                        source_root=root,
                        expected_approval_sha256=queue_item.content_sha256,
                    )
                )
                observed_item = replace(
                    source_item,
                    sha256=prepared.approval_sha256,
                    size_bytes=prepared.media_size,
                    mtime_ns=prepared.mtime_ns,
                    present=True,
                    updated_at=now_timestamp(),
                )
                self._upsert_source_item(observed_item, created_at=source_item.created_at)
                self._connection.commit()
                source_item = self.get_source_item(source_item.id)
                planned_bundle_id = prepared.planned_bundle_id()
                if planned_bundle_id is not None:
                    completed_result = self._queue_owned_bundle_result(
                        queue_item, planned_bundle_id
                    )
                    if completed_result is not None:
                        self._set_queue_state(
                            queue_item.id,
                            QueueState.COMPLETED,
                            bundle_id=planned_bundle_id,
                            clear_error=True,
                        )
                        return self._repaired_replay_result(
                            completed_result,
                            source=source,
                            source_item=source_item,
                            queue_item_id=queue_item.id,
                            consent="explicit_queue_approval",
                        )
                    self._ensure_bundle_id_available(planned_bundle_id, queue_item, output_root)
                result = prepared.ingest(output_root)
                try:
                    self._ensure_library_bundle_id_available(result.manifest.bundle_id, queue_item)
                except AutomationError:
                    shutil.rmtree(result.bundle_dir, ignore_errors=True)
                    raise
            except AutomationError as exc:
                self._record_ingest_failure(queue_item, str(exc))
                raise
            except (IngestError, OSError) as exc:
                if isinstance(exc, BundleExistsError):
                    completed_result = self._queue_owned_bundle_result(queue_item, exc.bundle_id)
                    if completed_result is not None:
                        self._set_queue_state(
                            queue_item.id,
                            QueueState.COMPLETED,
                            bundle_id=exc.bundle_id,
                            clear_error=True,
                        )
                        return self._repaired_replay_result(
                            completed_result,
                            source=source,
                            source_item=source_item,
                            queue_item_id=queue_item.id,
                            consent="explicit_queue_approval",
                        )
                self._record_ingest_failure(queue_item, str(exc))
                raise

            # Commit the completed state and the library record together so a crash can
            # never leave a COMPLETED queue row without the library row that makes the
            # bundle findable (and the on-disk bundle a retry blocker). Provenance is
            # attached afterwards so it can report the queue state the store actually
            # holds instead of asserting a literal.
            try:
                self._apply_queue_state(
                    queue_item.id,
                    QueueState.COMPLETED,
                    bundle_id=result.manifest.bundle_id,
                    clear_error=True,
                )
                library_record = self._record_library_bundle(
                    result.bundle_dir, source, source_item, queue_item
                )
                self._connection.commit()
            except Exception as exc:
                self._connection.rollback()
                shutil.rmtree(result.bundle_dir, ignore_errors=True)
                self._record_ingest_failure(queue_item, str(exc))
                raise
            completed = self.get_queue_item(queue_item.id)
            try:
                attach_provenance_to_bundle(
                    result.bundle_dir,
                    source=source,
                    source_item=source_item,
                    queue_item=completed,
                    consent="explicit_queue_approval",
                )
            except Exception as exc:
                # Remove the half-written bundle: leaving it on disk with no library
                # record makes the retry path collide with an unrecorded directory.
                shutil.rmtree(result.bundle_dir, ignore_errors=True)
                # Mirror of the one-shot path: a re-approved item can be rerun under a
                # different transcriber command, and a failed rerun must not demote the
                # earlier success. Restore the pre-ingest bundle id when its bundle
                # survived this deletion, so the replay path can still reach it.
                self._undo_completed_ingest(
                    queue_item.id,
                    result.manifest.bundle_id,
                    str(exc),
                    library_record=library_record,
                    restore_bundle_id=self._restorable_queue_bundle_id(
                        queue_item, library_record.previous
                    ),
                )
                raise
            return IngestResult(
                bundle_dir=result.bundle_dir,
                manifest=Manifest.load(result.bundle_dir),
            )

    def ingest_one_shot(
        self,
        source_path: Path,
        output_root: Path,
        *,
        transcriber_command: str | None = None,
        asr_model: str = "small",
        allow_model_download: bool = False,
        language: str | None = None,
        frames: bool = False,
        frame_interval: float = DEFAULT_FRAME_INTERVAL,
        max_frames: int = DEFAULT_MAX_FRAMES,
        diarize: bool = False,
        speakers: int | None = None,
        diarize_models: Path = DEFAULT_DIARIZE_MODELS,
    ) -> IngestResult:
        with prepare_local_ingest(
            source_path,
            transcriber_command=transcriber_command,
            asr_model=asr_model,
            allow_model_download=allow_model_download,
            language=language,
            frames=frames,
            frame_interval=frame_interval,
            max_frames=max_frames,
            diarize=diarize,
            speakers=speakers,
            diarize_models=diarize_models,
            expected_approval_sha256=None,
        ) as prepared:
            planned_bundle_id = prepared.planned_bundle_id()
            source = self._ensure_one_shot_source(prepared.registry_path)
            source_item = self._ensure_one_shot_item(
                source,
                prepared.registry_path,
                digest=prepared.approval_sha256,
                size=prepared.media_size,
                mtime_ns=prepared.mtime_ns,
            )
            queue_item = self._ensure_one_shot_queue(source, source_item)
            if planned_bundle_id is not None:
                completed_result = self._queue_owned_bundle_result(queue_item, planned_bundle_id)
                if completed_result is not None:
                    self._set_queue_state(
                        queue_item.id,
                        QueueState.COMPLETED,
                        bundle_id=planned_bundle_id,
                        clear_error=True,
                    )
                    return self._repaired_replay_result(
                        completed_result,
                        source=source,
                        source_item=source_item,
                        queue_item_id=queue_item.id,
                        consent="explicit_cli_invocation",
                    )
            try:
                if planned_bundle_id is not None:
                    self._ensure_bundle_id_available(planned_bundle_id, queue_item, output_root)
            except AutomationError as exc:
                self._record_ingest_failure(queue_item, str(exc))
                raise
            try:
                result = prepared.ingest(output_root)
            except (IngestError, OSError) as exc:
                if isinstance(exc, BundleExistsError):
                    completed_result = self._queue_owned_bundle_result(queue_item, exc.bundle_id)
                    if completed_result is not None:
                        self._set_queue_state(
                            queue_item.id,
                            QueueState.COMPLETED,
                            bundle_id=exc.bundle_id,
                            clear_error=True,
                        )
                        return self._repaired_replay_result(
                            completed_result,
                            source=source,
                            source_item=source_item,
                            queue_item_id=queue_item.id,
                            consent="explicit_cli_invocation",
                        )
                self._record_ingest_failure(queue_item, str(exc))
                raise
            try:
                self._ensure_library_bundle_id_available(result.manifest.bundle_id, queue_item)
            except AutomationError as exc:
                shutil.rmtree(result.bundle_dir, ignore_errors=True)
                self._record_ingest_failure(queue_item, str(exc))
                raise

            # Mirror of the queue-ingest path: completion and its library record share one
            # transaction, and provenance runs against the committed state afterwards.
            try:
                self._apply_queue_state(
                    queue_item.id,
                    QueueState.COMPLETED,
                    bundle_id=result.manifest.bundle_id,
                    clear_error=True,
                )
                library_record = self._record_library_bundle(
                    result.bundle_dir, source, source_item, queue_item
                )
                self._connection.commit()
            except Exception as exc:
                self._connection.rollback()
                shutil.rmtree(result.bundle_dir, ignore_errors=True)
                self._record_ingest_failure(queue_item, str(exc))
                raise
            completed = self.get_queue_item(queue_item.id)
            try:
                attach_provenance_to_bundle(
                    result.bundle_dir,
                    source=source,
                    source_item=source_item,
                    queue_item=completed,
                    consent="explicit_cli_invocation",
                )
            except Exception as exc:
                # Mirror of the queue-ingest failure path: never leave a half-written
                # bundle that a later run would reject as an unrecorded directory.
                shutil.rmtree(result.bundle_dir, ignore_errors=True)
                # A rerun of an already-completed item must not lose the earlier success:
                # restore the prior COMPLETED bundle id rather than recording FAILED with
                # the new (now deleted) bundle on the row.
                restore_bundle_id = self._restorable_queue_bundle_id(
                    queue_item, library_record.previous
                )
                self._undo_completed_ingest(
                    queue_item.id,
                    result.manifest.bundle_id,
                    str(exc),
                    library_record=library_record,
                    restore_bundle_id=restore_bundle_id,
                )
                raise
            return IngestResult(
                bundle_dir=result.bundle_dir,
                manifest=Manifest.load(result.bundle_dir),
            )

    def _ensure_bundle_id_available(
        self, bundle_id: str, queue_item: QueueItem, output_root: Path
    ) -> None:
        existing_bundle = self._existing_library_bundle(bundle_id)
        if existing_bundle is not None:
            if existing_bundle.queue_item_id == queue_item.id:
                return
            raise AutomationError(
                "bundle id already exists for different source provenance; "
                "duplicate-content multi-source provenance is not implemented"
            )
        bundle_dir = output_root / bundle_id
        if bundle_dir.exists():
            raise AutomationError(
                "bundle id already exists on disk but is not recorded in state; "
                "choose a different output directory or remove the existing bundle"
            )

    def _ensure_library_bundle_id_available(self, bundle_id: str, queue_item: QueueItem) -> None:
        existing_bundle = self._existing_library_bundle(bundle_id)
        if existing_bundle is None or existing_bundle.queue_item_id == queue_item.id:
            return
        raise AutomationError(
            "bundle id already exists for different source provenance; "
            "duplicate-content multi-source provenance is not implemented"
        )

    def _queue_owned_bundle_result(
        self, queue_item: QueueItem, candidate_id: str
    ) -> IngestResult | None:
        library_bundle = self._existing_library_bundle(candidate_id)
        if library_bundle is None:
            return None
        if (
            library_bundle.queue_item_id != queue_item.id
            or library_bundle.source_id != queue_item.source_id
            or library_bundle.source_item_id != queue_item.source_item_id
        ):
            return None
        bundle_dir = Path(library_bundle.bundle_path)
        if not bundle_dir.is_dir():
            return None
        try:
            manifest = Manifest.load(bundle_dir)
        except (OSError, ValueError, RecursionError):
            return None
        if manifest.bundle_id != candidate_id:
            return None
        return IngestResult(bundle_dir=bundle_dir, manifest=manifest)

    def _repaired_replay_result(
        self,
        result: IngestResult,
        *,
        source: SourceRecord,
        source_item: SourceItem,
        queue_item_id: str,
        consent: str,
    ) -> IngestResult:
        """Re-attach provenance to a replayed bundle left stale by a crash.

        Completion commits before provenance is attached, so a replay can find a
        completed, library-recorded bundle whose provenance never landed. Re-attach
        is idempotent (identical inputs rewrite identical bytes), so a healthy
        bundle is returned untouched.
        """

        if not bundle_provenance_needs_repair(result.bundle_dir):
            return result
        attach_provenance_to_bundle(
            result.bundle_dir,
            source=source,
            source_item=source_item,
            queue_item=self.get_queue_item(queue_item_id),
            consent=consent,
        )
        return IngestResult(
            bundle_dir=result.bundle_dir,
            manifest=Manifest.load(result.bundle_dir),
        )

    def _record_ingest_failure(self, queue_item: QueueItem, message: str) -> None:
        prior_id = self._restorable_queue_bundle_id(queue_item)
        if prior_id is None:
            self._record_failed_queue_item(queue_item.id, message)
        else:
            self._set_queue_state(
                queue_item.id, QueueState.COMPLETED, bundle_id=prior_id, clear_error=True
            )

    def _restorable_queue_bundle_id(
        self,
        queue_item: QueueItem,
        previous_library_row: LibraryBundle | None = None,
    ) -> str | None:
        """Report a pre-ingest bundle id whose bundle a failed rerun can fall back to.

        Only an item's own still-recorded, still-on-disk bundle qualifies. When a
        same-id rerun repointed the library row before failing, the current row
        references the just-deleted directory, so the pre-update row is the one
        that can prove the earlier bundle survived.
        """

        if queue_item.bundle_id is None:
            return None
        if (
            previous_library_row is not None
            and previous_library_row.bundle_id == queue_item.bundle_id
            and previous_library_row.queue_item_id == queue_item.id
            and previous_library_row.source_id == queue_item.source_id
            and previous_library_row.source_item_id == queue_item.source_item_id
            and Path(previous_library_row.bundle_path).is_dir()
        ):
            return queue_item.bundle_id
        try:
            library_bundle = self.get_library_bundle(queue_item.bundle_id)
        except AutomationError:
            return None
        if (
            library_bundle.queue_item_id != queue_item.id
            or library_bundle.source_id != queue_item.source_id
            or library_bundle.source_item_id != queue_item.source_item_id
        ):
            return None
        if not Path(library_bundle.bundle_path).is_dir():
            return None
        return queue_item.bundle_id

    def _undo_completed_ingest(
        self,
        queue_item_id: str,
        bundle_id: str,
        message: str,
        *,
        library_record: LibraryRecordOutcome,
        restore_bundle_id: str | None,
    ) -> None:
        """Reverse a committed completion in one transaction after provenance failed."""

        if library_record.inserted:
            self._connection.execute(
                "DELETE FROM library_bundles WHERE bundle_id = ? AND queue_item_id = ?",
                (bundle_id, queue_item_id),
            )
            # The index rows go with the library row. Without this the bundle
            # directory is deleted, its registration is gone, and its segments
            # stay searchable -- so a hit resolves to a bundle that cannot be
            # shown or cited. The atomicity test covered the commit path and
            # never this one.
            self._delete_index_rows(bundle_id)
        elif library_record.previous is not None:
            # The row was updated onto the bundle that has just been deleted; the
            # earlier bundle survives on disk, so the library must point back at it.
            previous = library_record.previous
            self._connection.execute(
                """
                UPDATE library_bundles
                SET bundle_path = ?, source_id = ?, source_item_id = ?, queue_item_id = ?
                WHERE bundle_id = ?
                """,
                (
                    previous.bundle_path,
                    previous.source_id,
                    previous.source_item_id,
                    previous.queue_item_id,
                    previous.bundle_id,
                ),
            )
        if restore_bundle_id is None:
            self._apply_failed_queue_item(queue_item_id, message)
        else:
            self._apply_queue_state(
                queue_item_id,
                QueueState.COMPLETED,
                bundle_id=restore_bundle_id,
                clear_error=True,
            )
        self._connection.commit()


def open_state(path: Path = DEFAULT_STATE_PATH) -> AutomationState:
    try:
        return AutomationState(path)
    except sqlite3.Error as exc:
        raise AutomationError(f"state database error: {exc}") from exc


def state_summary(path: Path) -> dict[str, Any]:
    with open_state(path) as state:
        return {
            "schema_version": STATE_SCHEMA_VERSION,
            "state_path": str(path),
            "sources": len(state.list_sources()),
            "queue": len(state.list_queue()),
            "library": len(state.list_library()),
        }
