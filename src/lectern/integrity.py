"""Nonwriting bundle integrity checks shared by library status and replay."""

from pathlib import Path

from lectern.bundle import Manifest, SourceDocument, StageState
from lectern.records import digest_and_size


def materialized_stage_states(manifest: Manifest) -> list[str]:
    """Omit only untouched future stages, as library status has always done."""

    return [
        record.state.value
        for record in manifest.stages.values()
        if record.state is not StageState.PENDING
        or record.started is not None
        or record.finished is not None
        or record.outputs
        or record.error is not None
    ]


def load_registered_manifest(bundle_id: str, bundle_path: Path) -> Manifest:
    manifest = Manifest.load(bundle_path)
    if manifest.bundle_id != bundle_id:
        raise ValueError("registered bundle id does not match manifest bundle id")
    return manifest


def assert_registered_source_identity(source: SourceDocument, manifest: Manifest) -> None:
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


def manifest_outputs_are_materialized(bundle_path: Path, manifest: Manifest) -> bool:
    try:
        if bundle_path.is_symlink():
            return False
        bundle_path.resolve(strict=True)
    except OSError:
        return False
    for stage in manifest.stages.values():
        for output in stage.outputs:
            try:
                resolved_candidate = contained_artifact_path(bundle_path, output.path)
                digest, size = digest_and_size(resolved_candidate)
            except (OSError, ValueError):
                return False
            if size != output.bytes or digest != output.sha256:
                return False
    return True


def contained_artifact_path(bundle: Path, relative: str) -> Path:
    """Resolve a regular bundle artifact, refusing symlinks and escapes."""

    if bundle.is_symlink():
        raise ValueError("evidence bundle root must not be a symlink")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("evidence path escapes the bundle")
    if _has_symlink_component(bundle, path):
        raise ValueError("evidence path must not contain a symlink")
    resolved = (bundle / path).resolve(strict=True)
    if not resolved.is_relative_to(bundle.resolve(strict=True)) or not resolved.is_file():
        raise ValueError("evidence must be a contained regular file")
    return resolved


def _has_symlink_component(bundle_path: Path, relative: Path) -> bool:
    candidate = bundle_path
    for part in relative.parts:
        candidate /= part
        if candidate.is_symlink():
            return True
    return False
