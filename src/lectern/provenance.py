"""Automation provenance written into a completed bundle.

A bundle records where it came from: which source, which discovery item, which
queue item, under which policy and consent, and what remote services the bundle
itself reports having involved. New ingest completion commits before this runs;
replay repairs describe a proposed completion, committed only after re-admission.
The repair check detects bundles whose provenance never landed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

from lectern.bundle import ArtifactRef, Manifest, SourceDocument, StageName, atomic_write_text
from lectern.integrity import contained_artifact_path
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

PENDING_PROVENANCE_NAME = ".lectern-provenance.pending.json"
# Historical compatibility is deliberately finite; never search arbitrary versions.
HISTORICAL_PROVENANCE_SCHEMAS = (3,)


def pending_provenance_bytes(bundle_dir: Path, core: dict[str, Any]) -> bytes | None:
    """Read recognized transaction intent without trusting it as a declaration.

    A valid same-core receipt may be stale (kill before declaration). Hash/size
    matching is a separate admission requirement, not a reason to clobber an
    unrecognized reserved-path collision. Never follow a receipt symlink.
    """

    path = bundle_dir / PENDING_PROVENANCE_NAME
    if not path.is_symlink() and not path.exists():
        return None
    payload = contained_artifact_path(bundle_dir, PENDING_PROVENANCE_NAME).read_bytes()
    document = SourceDocument.model_validate_json(payload, strict=True)
    if document.provenance is None or document.provenance.queue_state != "completed":
        raise ValueError("pending provenance must contain completed writer provenance")
    parsed = cast(dict[str, Any], json.loads(payload))
    parsed.pop("provenance", None)
    # Compare preserved values, not a model dump that could insert defaults or
    # coerce/drop data. Key order may differ; JSON types/values may not.
    if json.dumps(parsed, sort_keys=True) != json.dumps(core, sort_keys=True):
        raise ValueError("pending provenance differs from the current source core")
    return payload


def attach_provenance_to_bundle(
    bundle_dir: Path,
    *,
    source: SourceRecord,
    source_item: SourceItem,
    queue_item: QueueItem,
    consent: str,
    source_recovery: bytes | None = None,
) -> None:
    source_path = contained_artifact_path(bundle_dir, "source.json")
    source_payload = cast(dict[str, Any], json.loads(source_path.read_text(encoding="utf-8")))
    core = source_payload.copy()
    core.pop("provenance", None)
    SourceDocument.model_validate_json(json.dumps(core), strict=True)
    pending = pending_provenance_bytes(bundle_dir, core)
    pending_path = bundle_dir / PENDING_PROVENANCE_NAME
    manifest = Manifest.load(bundle_dir)
    source_ref = next(
        output
        for output in manifest.stages[StageName.ACQUIRE].outputs
        if output.path == "source.json"
    )
    if source_recovery is not None:
        recovered = cast(dict[str, Any], json.loads(source_recovery))
        recovered_core = recovered.copy()
        recovered_core.pop("provenance", None)
        SourceDocument.model_validate_json(json.dumps(recovered_core), strict=True)
        if (
            json.dumps(recovered_core, sort_keys=True) != json.dumps(core, sort_keys=True)
            or source_ref.sha256 != hashlib.sha256(source_recovery).hexdigest()
            or source_ref.bytes != len(source_recovery)
        ):
            raise ValueError("source recovery must match the declaration and current core")
        # Settle the old declaration BEFORE rotating its intent. A second kill
        # must not lose the only bytes binding this source core to that digest.
        atomic_write_text(source_path, source_recovery.decode("utf-8"))
        source_payload = recovered
    if pending is not None:
        # Admission first restores the exact declared bytes, if needed. Do not
        # rotate intent while the old declaration still depends on that receipt.
        encoded_source = source_path.read_bytes()
        if source_ref.sha256 != hashlib.sha256(
            encoded_source
        ).hexdigest() or source_ref.bytes != len(encoded_source):
            raise ValueError("pending provenance requires declared source recovery first")
        if isinstance(source_payload.get("provenance"), dict):
            try:
                document = SourceDocument.model_validate_json(encoded_source, strict=True)
            except ValueError:
                pass
            else:
                if document.provenance is not None:
                    pending_path.unlink()
                    return
    source_payload = source_with_provenance(
        source_payload,
        source=source,
        source_item=source_item,
        queue_item=queue_item,
        consent=consent,
    )
    text = json.dumps(source_payload, indent=2) + "\n"
    encoded = text.encode("utf-8")
    # Retain exact target bytes BEFORE declaring them. Recovery does not depend
    # on today's schema number, source name, or provenance key placement.
    atomic_write_text(pending_path, text)
    acquire = manifest.stages[StageName.ACQUIRE]
    acquire.outputs = [
        ArtifactRef(
            path=output.path, sha256=hashlib.sha256(encoded).hexdigest(), bytes=len(encoded)
        )
        if output.path == "source.json"
        else output
        for output in acquire.outputs
    ]
    manifest.save(bundle_dir)
    atomic_write_text(source_path, text)
    pending_path.unlink()


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
