"""Automation provenance written into a completed bundle.

A bundle records where it came from: which source, which discovery item, which
queue item, under which policy and consent, and what remote services the bundle
itself reports having involved. Completion commits before this runs, so the
repair check exists to detect a bundle whose provenance never landed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from lectern.bundle import ArtifactRef, Manifest, StageName, atomic_write_text
from lectern.records import QueueItem, SourceItem, SourceRecord, digest_and_size
from lectern.state import STATE_SCHEMA_VERSION

# Keys attach_provenance_to_bundle writes into source.json["provenance"]; a
# completed bundle missing any of them predates a finished provenance attach.
PROVENANCE_KEYS = frozenset(
    {
        "state_schema_version",
        "source_id",
        "source_kind",
        "source_name",
        "source_item_id",
        "queue_item_id",
        "queue_state",
        "policy",
        "consent",
        "remote_services",
    }
)


def attach_provenance_to_bundle(
    bundle_dir: Path,
    *,
    source: SourceRecord,
    source_item: SourceItem,
    queue_item: QueueItem,
    consent: str,
) -> None:
    source_path = bundle_dir / "source.json"
    source_payload = cast(dict[str, Any], json.loads(source_path.read_text(encoding="utf-8")))
    source_payload = source_with_provenance(
        source_payload,
        source=source,
        source_item=source_item,
        queue_item=queue_item,
        consent=consent,
    )
    # Publish atomically: completion and library registration already committed.
    atomic_write_text(source_path, json.dumps(source_payload, indent=2) + "\n")

    manifest = Manifest.load(bundle_dir)
    acquire = manifest.stages[StageName.ACQUIRE]
    updated_outputs: list[ArtifactRef] = []
    for output in acquire.outputs:
        if output.path == "source.json":
            digest, size = digest_and_size(source_path)
            updated_outputs.append(ArtifactRef(path=output.path, sha256=digest, bytes=size))
        else:
            updated_outputs.append(output)
    acquire.outputs = updated_outputs
    manifest.save(bundle_dir)


def source_with_provenance(
    source_payload: dict[str, Any],
    *,
    source: SourceRecord,
    source_item: SourceItem,
    queue_item: QueueItem,
    consent: str,
) -> dict[str, Any]:
    """Build the writer's annotation without modifying the supplied source core."""

    return {
        **source_payload,
        "provenance": {
            "state_schema_version": STATE_SCHEMA_VERSION,
            "source_id": source.id,
            "source_kind": source.kind.value,
            "source_name": source.name,
            "source_item_id": source_item.id,
            "queue_item_id": queue_item.id,
            "queue_state": queue_item.state.value,
            "policy": queue_item.policy.value,
            "consent": consent,
            "remote_services": _bundle_remote_services(source_payload),
        },
    }


def bundle_provenance_needs_repair(bundle_dir: Path) -> bool:
    """Report whether a completed bundle's automation provenance is out of date.

    Completion and the library row commit before provenance is attached, so a
    crash in that window can leave a completed, library-recorded bundle whose
    source.json lacks provenance, or whose manifest still records the
    pre-provenance source.json digest. Both are repairable by re-attaching.
    """

    source_path = bundle_dir / "source.json"
    try:
        payload_obj = json.loads(source_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError):
        # Nothing re-attaching could fix; leave the bundle exactly as found.
        return False
    if not isinstance(payload_obj, dict):
        return False
    payload = cast(dict[str, Any], payload_obj)
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        return True
    if not set(cast(dict[str, Any], provenance)) >= PROVENANCE_KEYS:
        return True

    try:
        manifest = Manifest.load(bundle_dir)
        acquire = manifest.stages[StageName.ACQUIRE]
    except (OSError, KeyError, ValueError, RecursionError):
        return False
    recorded = next(
        (output.sha256 for output in acquire.outputs if output.path == "source.json"),
        None,
    )
    if recorded is None:
        return False
    try:
        digest, _ = digest_and_size(source_path)
    except OSError:
        return False
    return recorded != digest


def _bundle_remote_services(source_payload: dict[str, Any]) -> dict[str, Any]:
    """Record the bundle's own remote-services metadata instead of fresh literals."""

    recorded: dict[str, Any] = {}
    transcript = source_payload.get("transcript")
    if isinstance(transcript, dict):
        candidate = cast(dict[str, Any], transcript).get("remote_services")
        if isinstance(candidate, dict):
            recorded = cast(dict[str, Any], candidate)
    return {
        "allowed": recorded.get("allowed", False),
        "scope": recorded.get("scope", "lectern_core"),
        "lectern_invoked": recorded.get("lectern_invoked", False),
        "requires_explicit_per_item_consent": recorded.get(
            "requires_explicit_per_item_consent", True
        ),
        "transcriber_network_posture": recorded.get("transcriber_network_posture", "not_recorded"),
    }
