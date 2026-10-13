"""Internal selected-transcript reads shared by strict consumers and diagnostics."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from lectern.bundle import (
    FramesDocument,
    Manifest,
    SourceDocument,
    StageName,
    StageState,
    TranscriptMetadataDocument,
    TranscriptSegmentsDocument,
)


@dataclass(frozen=True)
class SelectedEvidence:
    source: SourceDocument
    segments_payload: bytes
    segments: TranscriptSegmentsDocument


def _contained_payload(bundle: Path, relative: str) -> bytes:
    if bundle.is_symlink():
        raise ValueError("evidence bundle root must not be a symlink")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("evidence path escapes the bundle")
    candidate = bundle
    for part in path.parts:
        candidate /= part
        if candidate.is_symlink():
            raise ValueError("evidence path must not contain a symlink")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(bundle.resolve(strict=True)) or not resolved.is_file():
        raise ValueError("evidence must be a contained regular file")
    return resolved.read_bytes()


def _assert_declared_payload(manifest: Manifest, path: str, payload: bytes) -> None:
    references = [
        output
        for stage in manifest.stages.values()
        for output in stage.outputs
        if output.path == path
    ]
    if not references:
        raise ValueError("selected evidence is not declared by the manifest")
    digest = hashlib.sha256(payload).hexdigest()
    if any(ref.bytes != len(payload) or ref.sha256 != digest for ref in references):
        raise ValueError("selected evidence does not match its manifest artifact")


def read_selected_evidence(
    bundle: Path, manifest: Manifest, *, require_manifest_integrity: bool
) -> SelectedEvidence:
    """Read and parse the exact bytes checked against the manifest.

    Callers retain their own manifest/registration and source-identity policies.
    Diagnostic mode permits changed or undeclared evidence, but still requires
    contained files and valid current document shapes. No text is transformed.
    """

    source_payload = _contained_payload(bundle, "source.json")
    if require_manifest_integrity:
        _assert_declared_payload(manifest, "source.json", source_payload)
    source = SourceDocument.model_validate_json(source_payload, strict=True)
    segments_payload = _contained_payload(bundle, source.transcript.segments)
    if require_manifest_integrity:
        _assert_declared_payload(manifest, source.transcript.segments, segments_payload)
    segments = TranscriptSegmentsDocument.model_validate_json(segments_payload, strict=True)
    return SelectedEvidence(source, segments_payload, segments)


def read_transcript_metadata(
    bundle: Path, manifest: Manifest, source: SourceDocument
) -> TranscriptMetadataDocument:
    """Read the selected metadata through the same contained, declared-byte boundary."""

    path = source.transcript.metadata
    payload = _contained_payload(bundle, path)
    _assert_declared_payload(manifest, path, payload)
    return TranscriptMetadataDocument.model_validate_json(payload, strict=True)


def read_frames(bundle: Path, manifest: Manifest) -> FramesDocument | None:
    """Read declared visual evidence through the same contained-byte boundary."""

    visual = manifest.stages[StageName.VISUAL]
    if visual.state is not StageState.DONE or not any(
        output.path == "visual/frames.json" for output in visual.outputs
    ):
        return None
    payload = _contained_payload(bundle, "visual/frames.json")
    _assert_declared_payload(manifest, "visual/frames.json", payload)
    frames = FramesDocument.model_validate_json(payload, strict=True)
    for frame in frames.root:
        image = _contained_payload(bundle, frame.file)
        _assert_declared_payload(manifest, frame.file, image)
        if hashlib.sha256(image).hexdigest() != frame.sha256:
            raise ValueError("frame image does not match its index digest")
    return frames
