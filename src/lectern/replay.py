"""Read-only admission for owned ingest replays, including repair preflight."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, cast

from lectern.bundle import ArtifactRef, Manifest, SourceDocument, StageName, StageState
from lectern.evidence import read_selected_evidence, read_selected_segments
from lectern.integrity import (
    assert_registered_source_identity,
    contained_artifact_path,
    load_registered_manifest,
    manifest_outputs_are_materialized,
    materialized_stage_states,
)
from lectern.provenance import PROVENANCE_KEYS, source_with_provenance
from lectern.records import (
    LibraryBundle,
    LibraryStatus,
    QueueItem,
    QueueState,
    SourceItem,
    SourceRecord,
    derive_library_status,
    redact_paths,
)


@dataclass(frozen=True)
class ReplayAdmission:
    outcome: Literal["admit", "admit_after_provenance_repair", "refuse"]
    reason: str = ""
    owned: bool = True
    manifest: Manifest | None = None


def admit_replay(
    candidate_id: str,
    library: LibraryBundle | None,
    *,
    queue_item: QueueItem,
    source: SourceRecord,
    source_item: SourceItem,
    consent: str,
) -> ReplayAdmission:
    """Check the entire replay boundary without changing disk or registry state.

    The registered bundle may live under another output root: a replay answers
    with it, as before, and never adopts a colliding directory in the new root.
    Registration records the bundle id but currently has no manifest digest.
    Source hash exceptions recognize only exact bytes that the existing writer
    can account for, not hypothetical historical provenance or input authenticity.
    """

    if library is None or (
        library.queue_item_id != queue_item.id
        or library.source_id != queue_item.source_id
        or library.source_item_id != queue_item.source_item_id
        or source.id != queue_item.source_id
        or source_item.id != queue_item.source_item_id
        or source_item.source_id != source.id
    ):
        return ReplayAdmission(
            "refuse", "candidate is not registered to this queue item", owned=False
        )
    bundle = Path(library.bundle_path)
    if not bundle.is_dir():
        return ReplayAdmission("refuse", "registered bundle is missing on disk", owned=False)
    try:
        manifest = load_registered_manifest(candidate_id, bundle)
        # Check every declared path before considering any repair exception.
        for stage in manifest.stages.values():
            for output in stage.outputs:
                contained_artifact_path(bundle, output.path)
        source_refs = [
            (stage, output)
            for stage, record in manifest.stages.items()
            for output in record.outputs
            if output.path == "source.json"
        ]
        if len(source_refs) != 1 or source_refs[0][0] is not StageName.ACQUIRE:
            raise ValueError("source.json must be declared exactly once, in ACQUIRE")
        source_ref = source_refs[0][1]
        payload = contained_artifact_path(bundle, "source.json").read_bytes()
        parsed: object = json.loads(payload.decode("utf-8"))
        if not isinstance(parsed, dict):
            raise ValueError("source.json must be an object")
        core = cast(dict[str, Any], parsed).copy()
        provenance = core.pop("provenance", None)
        source_document = SourceDocument.model_validate_json(json.dumps(core), strict=True)

        def matches_recorded(candidate: bytes) -> bool:
            return (
                len(candidate) == source_ref.bytes
                and hashlib.sha256(candidate).hexdigest() == source_ref.sha256
            )

        source_matches = matches_recorded(payload)
        if not source_matches:
            # Known writer representations only. Serializing the preserved core
            # retains unknown values/key order; strict validation above rejects
            # malformed core data instead of dropping/coercing it on round-trip.
            bare = (json.dumps(core, indent=2) + "\n").encode("utf-8")
            # A one-shot item may subsequently be reapproved through the queue.
            # Both existing entry points' exact writer bytes are recognizable;
            # no arbitrary historical annotation is inferred from its digest.
            annotations: list[bytes] = []
            for recorded_consent in {consent, "explicit_cli_invocation", "explicit_queue_approval"}:
                annotated = source_with_provenance(
                    core,
                    source=source,
                    source_item=source_item,
                    queue_item=replace(queue_item, state=QueueState.COMPLETED),
                    consent=recorded_consent,
                )
                annotations.append((json.dumps(annotated, indent=2) + "\n").encode("utf-8"))
            # Repair can be interrupted after publishing a fresh annotation but
            # before declaring it. The old declaration may describe a partial
            # annotation whose bytes are no longer available. Recognize only
            # the exact fresh writer bytes over the strictly validated core;
            # all other declared artifacts are still checked below.
            if payload not in annotations and not any(
                matches_recorded(candidate) for candidate in [bare, *annotations]
            ):
                raise ValueError(
                    "source.json differs from its declaration beyond identifiable provenance"
                )

        # Only this validated source reference may differ. The ordinary library
        # integrity check still checks every path, size and digest in the preview.
        preview = manifest.model_copy(deep=True)
        preview.stages[StageName.ACQUIRE].outputs = [
            ArtifactRef(
                path="source.json", sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload)
            )
            if ref.path == "source.json"
            else ref
            for ref in preview.stages[StageName.ACQUIRE].outputs
        ]
        if not manifest_outputs_are_materialized(bundle, preview):
            raise ValueError("declared artifacts do not match their sha256 and size")
        required = {
            StageName.ACQUIRE: {"source.json"},
            StageName.NORMALIZE: {"media/audio.wav"},
            StageName.TRANSCRIBE: {
                source_document.transcript.segments,
                source_document.transcript.metadata,
                source_document.transcript.transcript,
            },
            StageName.SYNTHESIZE: {"analysis/summary.md"},
        }
        for name, outputs in required.items():
            record = manifest.stages.get(name)
            if (
                record is None
                or record.state is not StageState.DONE
                or not outputs <= {ref.path for ref in record.outputs}
            ):
                raise ValueError(f"completed {name.value} stage and its outputs are required")
        if (
            derive_library_status(QueueState.COMPLETED, materialized_stage_states(manifest))
            is not LibraryStatus.READY
        ):
            raise ValueError("materialized stages are not complete")
        assert_registered_source_identity(source_document, manifest)
        # Share strict selected-segment parsing with search and new citations;
        # only the separately validated provenance field is excluded in preflight.
        read_selected_segments(bundle, manifest, source_document, require_manifest_integrity=True)

        repair = (
            not source_matches
            or not isinstance(provenance, dict)
            or not (set(cast(dict[str, Any], provenance)) >= PROVENANCE_KEYS)
        )
        if not repair:
            try:
                read_selected_evidence(bundle, manifest, require_manifest_integrity=True)
            except ValueError:
                # The strict core and segments passed: only annotation validation
                # can remain repairable here (e.g. complete keys, wrong types).
                repair = True
        return ReplayAdmission(
            "admit_after_provenance_repair" if repair else "admit", manifest=manifest
        )
    except (OSError, ValueError, RecursionError) as exc:
        return ReplayAdmission("refuse", redact_paths(str(exc)))
