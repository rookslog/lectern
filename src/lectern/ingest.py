"""Local ingest pipeline for early Lectern milestones.

Captions and plain transcript sidecars use captured local bytes. Without a
sidecar, transcription uses an explicit command or the optional local ASR extra.
"""

from __future__ import annotations

import hashlib
import html
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import wave
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from lectern import asr
from lectern.bundle import (
    ArtifactRef,
    Manifest,
    Source,
    SourceKind,
    StageName,
    StageRecord,
    StageState,
)
from lectern.readable import format_time
from lectern.records import AutomationError, approval_digest
from lectern.sources.local import caption_sidecar_language, transcript_sidecar_path

CANONICAL_SAMPLE_RATE = 16_000
CANONICAL_CHANNELS = 1
CANONICAL_SAMPLE_WIDTH = 2
DEFAULT_TRANSCRIBER_TIMEOUT_S = 300
ASR_INSTALL_HINT = (
    "no local transcription backend is configured for this file; "
    "install with pip install 'lectern[asr]' or uv sync --extra asr, "
    "pass --transcriber-command, or add a caption/.transcript.txt sidecar"
)


class IngestError(RuntimeError):
    """Raised when local ingest cannot complete under current constraints."""


class BundleExistsError(IngestError):
    """The computed output identity already has a destination on disk."""

    def __init__(self, bundle_id: str, bundle_dir: Path) -> None:
        self.bundle_id = bundle_id
        self.bundle_dir = bundle_dir
        super().__init__(f"bundle already exists: {bundle_dir}")


@dataclass(frozen=True)
class IngestResult:
    """Completed ingest run."""

    bundle_dir: Path
    manifest: Manifest


@dataclass(frozen=True)
class TranscriptSegment:
    id: int
    start_s: float
    end_s: float | None
    text: str
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "start_s": self.start_s,
            "end_s": self.end_s,
            "text": self.text,
            "source": self.source,
        }


@dataclass(frozen=True)
class TranscriptResult:
    text: str
    segments: tuple[TranscriptSegment, ...]
    method: str
    backend: dict[str, Any]
    evidence_limit: str
    remote_services: dict[str, Any]
    identity: dict[str, Any]
    sidecar: dict[str, Any] | None = None
    language: str | None = None

    @property
    def identity_component(self) -> str:
        return hashlib.sha256(
            json.dumps(self.identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True)
class PreparedLocalIngest:
    """Input facts and operations valid for one private snapshot lifetime."""

    original_path: Path
    registry_path: Path
    media_sha256: str
    media_size: int
    mtime_ns: int
    approval_sha256: str
    _media_path: Path
    _sidecar_payload: bytes | None
    _command: str | None
    _sidecar_path: Path | None
    _asr_model: str
    _allow_model_download: bool
    _language: str | None

    def planned_bundle_id(self) -> str | None:
        if self._command is not None or self._sidecar_payload is None:
            return None
        transcript = _with_language(
            _sidecar_transcript(
                self.original_path, self._sidecar_payload, sidecar_path=self._sidecar_path
            ),
            self._language,
        )
        digest = _combined_digest(self.media_sha256, transcript.identity_component)
        return f"{_slug(self.original_path.stem)}-{digest[:12]}"

    def ingest(self, output_root: Path) -> IngestResult:
        source = self.original_path
        source_digest, source_size = self.media_sha256, self.media_size
        output_root.mkdir(parents=True, exist_ok=True)

        temp_root = Path(tempfile.mkdtemp(prefix=".lectern-ingest.", dir=output_root))
        try:
            _ensure_bundle_dirs(temp_root)
            audio_path = temp_root / "media" / "audio.wav"
            _normalize_to_canonical_wav(self._media_path, audio_path)
            source_duration = _wav_duration_seconds(audio_path)

            if self._command is not None:
                transcript = _transcribe_with_local_command(
                    self._command, audio_path, source_duration
                )
            elif self._sidecar_payload is not None:
                transcript = _sidecar_transcript(
                    source, self._sidecar_payload, source_duration, sidecar_path=self._sidecar_path
                )
            else:
                transcript = _transcribe_with_asr(
                    self._asr_model,
                    audio_path,
                    self._language,
                    allow_model_download=self._allow_model_download,
                )
            transcript = _with_language(transcript, self._language)

            current_source_digest, current_source_size = _digest_and_size(source)
            if (current_source_digest, current_source_size) != (source_digest, source_size):
                raise IngestError("source file changed during ingest; retry after changes settle")

            bundle_digest = _combined_digest(source_digest, transcript.identity_component)
            bundle_id = f"{_slug(source.stem)}-{bundle_digest[:12]}"
            bundle_dir = output_root / bundle_id
            if bundle_dir.exists():
                raise BundleExistsError(bundle_id, bundle_dir)

            manifest = Manifest(
                bundle_id=bundle_id,
                source=Source(
                    kind=SourceKind.LOCAL,
                    ref=f"sha256:{source_digest}",
                    bytes=source_size,
                    title=source.stem.replace("_", " ").title(),
                    duration_s=source_duration,
                ),
            )

            source_json = temp_root / "source.json"
            _write_json(
                source_json,
                {
                    "source": manifest.source.model_dump(mode="json"),
                    "sha256": source_digest,
                    "bytes": source_size,
                    "transcript": {
                        "method": transcript.method,
                        "metadata": "transcript/metadata.json",
                        "segments": "transcript/segments.json",
                        "transcript": "transcript/transcript.md",
                        "evidence_limit": transcript.evidence_limit,
                        "remote_services": transcript.remote_services,
                    },
                    "transcript_sidecar": transcript.sidecar,
                },
            )
            manifest.stages[StageName.ACQUIRE] = _done_stage(temp_root, [source_json])
            manifest.stages[StageName.NORMALIZE] = _done_stage(temp_root, [audio_path])

            segments_path = temp_root / "transcript" / "segments.json"
            transcript_path = temp_root / "transcript" / "transcript.md"
            metadata_path = temp_root / "transcript" / "metadata.json"
            _write_json(segments_path, [segment.to_dict() for segment in transcript.segments])
            transcript_path.write_text(transcript.text.rstrip() + "\n", encoding="utf-8")
            _write_json(
                metadata_path,
                _transcript_metadata(
                    transcript=transcript,
                    source=source,
                    source_digest=source_digest,
                    normalized_audio=audio_path,
                ),
            )
            manifest.stages[StageName.TRANSCRIBE] = _done_stage(
                temp_root,
                [segments_path, transcript_path, metadata_path],
            )

            summary_path = temp_root / "analysis" / "summary.md"
            summary_path.write_text(_summary_lite(transcript, manifest.source), encoding="utf-8")
            manifest.stages[StageName.SYNTHESIZE] = _done_stage(temp_root, [summary_path])

            manifest.save(temp_root)
            temp_root.replace(bundle_dir)
            return IngestResult(bundle_dir=bundle_dir, manifest=Manifest.load(bundle_dir))
        except BaseException:
            shutil.rmtree(temp_root, ignore_errors=True)
            raise


@contextmanager
def prepare_local_ingest(
    source_path: Path,
    *,
    transcriber_command: str | None = None,
    asr_model: str = "small",
    allow_model_download: bool = False,
    language: str | None = None,
    source_root: Path | None = None,
    expected_approval_sha256: str | None,
) -> Generator[PreparedLocalIngest]:
    source = source_path.expanduser()
    if not source.is_file():
        raise IngestError(f"source file does not exist: {source}")
    registry_path = source.resolve()
    command = _explicit_transcriber_command(transcriber_command)
    with tempfile.TemporaryDirectory(prefix="lectern-input-") as temporary:
        snapshot = Path(temporary) / source.name
        digest = hashlib.sha256()
        size = 0
        with registry_path.open("rb") as incoming:
            mtime_ns = os.fstat(incoming.fileno()).st_mtime_ns
            descriptor = os.open(snapshot, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "wb") as outgoing:
                for chunk in iter(lambda: incoming.read(1024 * 1024), b""):
                    outgoing.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
        sidecar = transcript_sidecar_path(source, root=source_root)
        sidecar_payload = sidecar.read_bytes() if sidecar is not None else None
        media_digest = digest.hexdigest()
        sidecar_digest = (
            hashlib.sha256(sidecar_payload).hexdigest() if sidecar_payload is not None else None
        )
        composite = approval_digest(media_digest, sidecar_digest)
        if expected_approval_sha256 is not None and composite != expected_approval_sha256:
            raise AutomationError("source file changed since queue approval; rescan before ingest")
        yield PreparedLocalIngest(
            original_path=source,
            registry_path=registry_path,
            media_sha256=media_digest,
            media_size=size,
            mtime_ns=mtime_ns,
            approval_sha256=composite,
            _media_path=snapshot,
            _sidecar_payload=sidecar_payload,
            _command=command,
            _sidecar_path=sidecar,
            _asr_model=asr_model,
            _allow_model_download=allow_model_download,
            _language=language,
        )


def plan_local_bundle_id(
    source_path: Path, transcriber_command: str | None = None, *, language: str | None = None
) -> str:
    """Return the bundle id for a private capture of the source and sidecar."""

    with prepare_local_ingest(
        source_path,
        transcriber_command=transcriber_command,
        language=language,
        expected_approval_sha256=None,
    ) as prepared:
        if _explicit_transcriber_command(transcriber_command) is not None:
            raise IngestError(
                "bundle id for --transcriber-command cannot be planned before transcription"
            )
        planned = prepared.planned_bundle_id()
        if planned is None:
            raise IngestError(
                "no local transcription backend can be planned for this file; "
                "bundle id for built-in ASR cannot be planned before transcription; "
                "add a caption/.transcript.txt sidecar for deterministic planning"
            )
        return planned


def can_plan_local_bundle_id(source_path: Path, transcriber_command: str | None = None) -> bool:
    """Return whether bundle id can be known before media normalization/transcription."""

    source = source_path.expanduser()
    if _explicit_transcriber_command(transcriber_command) is not None:
        return False
    return transcript_sidecar_path(source) is not None


def ingest_local(
    source_path: Path,
    output_root: Path = Path("bundles"),
    *,
    transcriber_command: str | None = None,
    asr_model: str = "small",
    allow_model_download: bool = False,
    language: str | None = None,
) -> IngestResult:
    with prepare_local_ingest(
        source_path,
        transcriber_command=transcriber_command,
        asr_model=asr_model,
        allow_model_download=allow_model_download,
        language=language,
        expected_approval_sha256=None,
    ) as prepared:
        return prepared.ingest(output_root)


def _ensure_bundle_dirs(bundle_dir: Path) -> None:
    for relative in ("media", "transcript", "analysis", "log"):
        (bundle_dir / relative).mkdir(parents=True, exist_ok=True)


def _explicit_transcriber_command(transcriber_command: str | None) -> str | None:
    if transcriber_command is not None:
        command = transcriber_command.strip()
        return command or None
    return None


def _sidecar_transcript(
    source: Path,
    payload: bytes | None,
    duration_s: float | None = None,
    *,
    sidecar_path: Path | None = None,
) -> TranscriptResult:
    if payload is None:
        raise IngestError(ASR_INSTALL_HINT)
    transcript_path = sidecar_path or source.with_suffix(".transcript.txt")
    digest, size = hashlib.sha256(payload).hexdigest(), len(payload)
    is_caption = transcript_path.suffix in (".vtt", ".srt")
    try:
        transcript = (
            payload.decode("utf-8-sig" if is_caption else "utf-8")
            .replace("\r\n", "\n")
            .replace("\r", "\n")
            .strip()
        )
    except UnicodeDecodeError as exc:
        label = "caption" if is_caption else "fixture transcript"
        raise IngestError(f"{label} is not valid UTF-8: {transcript_path}") from exc
    if is_caption:
        kind = transcript_path.suffix[1:]
        segments = _caption_segments(transcript, kind)
        language = caption_sidecar_language(source, transcript_path)
        method = f"caption_sidecar_{kind}"
        return TranscriptResult(
            text="\n".join(segment.text for segment in segments),
            segments=segments,
            method=method,
            backend={"kind": "sidecar", "sha256": digest, "sidecar_filename": transcript_path.name},
            evidence_limit=(
                "captions are third-party/auto-generated text; "
                "Lectern records cue times but makes no transcript faithfulness claim"
            ),
            remote_services=_remote_services(transcriber_network_posture="not_applicable_sidecar"),
            identity={"method": method, "sha256": digest, "sidecar_filename": transcript_path.name},
            sidecar={"sha256": digest, "bytes": size},
            language=language,
        )
    if transcript:
        return TranscriptResult(
            text=transcript,
            segments=(
                TranscriptSegment(
                    id=0,
                    start_s=0.0,
                    end_s=duration_s,
                    text=transcript,
                    source="fixture_transcript",
                ),
            ),
            method="fixture_transcript_sidecar",
            backend={"kind": "sidecar", "sha256": digest},
            evidence_limit="fixture transcript passthrough; no ASR quality claim",
            remote_services=_remote_services(transcriber_network_posture="not_applicable_sidecar"),
            identity={"method": "fixture_transcript_sidecar", "sha256": digest},
            sidecar={"sha256": digest, "bytes": size},
        )
    raise IngestError(f"fixture transcript is empty: {transcript_path}")


def _with_language(transcript: TranscriptResult, language: str | None) -> TranscriptResult:
    if language is None:
        return transcript
    return replace(
        transcript, language=language, identity={**transcript.identity, "language": language}
    )


_CAPTION_TIME = r"(?:\d{2,}:)?\d{2}:\d{2}[.,]\d{3}"
_CAPTION_TIMING = re.compile(rf"^({_CAPTION_TIME})\s+-->\s+({_CAPTION_TIME})(?:\s+.*)?$")


def _caption_seconds(timestamp: str) -> float:
    parts = timestamp.replace(",", ".").split(":")
    seconds = float(parts[-1]) + int(parts[-2]) * 60
    return seconds + (int(parts[0]) * 3600 if len(parts) == 3 else 0)


def _caption_segments(transcript: str, kind: str) -> tuple[TranscriptSegment, ...]:
    segments: list[TranscriptSegment] = []
    previous_last_line = ""
    for block in re.split(r"\n[ \t]*\n", transcript):
        lines = block.strip().splitlines()
        if not lines or re.match(r"^(?:WEBVTT|NOTE|STYLE|REGION)(?:\s|$)", lines[0]):
            continue
        timing_index = next((i for i, line in enumerate(lines) if "-->" in line), None)
        if timing_index is None:
            continue
        timing = _CAPTION_TIMING.fullmatch(lines[timing_index].strip())
        if timing is None:
            raise IngestError("caption cue has invalid timing")
        start_s, end_s = (_caption_seconds(value) for value in timing.groups())
        if end_s < start_s:
            raise IngestError("caption cue end must not precede start")
        cleaned_lines = [
            " ".join(html.unescape(re.sub(r"<[^>]*>", "", line)).split())
            for line in lines[timing_index + 1 :]
        ]
        cleaned_lines = [line for line in cleaned_lines if line]
        if not cleaned_lines:
            continue
        # YouTube emits the previous last line again at the start of the next
        # cue, including brief repeat-only cues. Keep that line as the rolling
        # reference even when stripping it leaves no new text.
        repeated_first_line = kind == "vtt" and cleaned_lines[0] == previous_last_line
        previous_last_line = cleaned_lines[-1]
        text = "\n".join(cleaned_lines[1:] if repeated_first_line else cleaned_lines)
        if text:
            segments.append(
                TranscriptSegment(len(segments), start_s, end_s, text, f"caption_{kind}")
            )
    if not segments:
        raise IngestError("caption sidecar has no non-empty timed cues")
    return tuple(segments)


def _transcribe_with_asr(
    model: str, audio: Path, language: str | None, *, allow_model_download: bool = False
) -> TranscriptResult:
    try:
        backend = asr.create_asr_backend(model, allow_model_download=allow_model_download)
    except ImportError as exc:
        raise IngestError(ASR_INSTALL_HINT) from exc
    except Exception as exc:
        hint = (
            ""
            if allow_model_download
            else "; if the model is not cached locally, pass --allow-model-download to fetch it"
        )
        raise IngestError(f"local ASR initialization failed: {exc}{hint}") from exc
    try:
        result = backend.transcribe(audio, language=language)
    except Exception as exc:
        raise IngestError(f"local ASR transcription failed: {exc}") from exc
    segments = tuple(
        TranscriptSegment(i, segment.start_s, segment.end_s, segment.text, "faster_whisper")
        for i, segment in enumerate(result.segments)
    )
    if not segments:
        raise IngestError("local ASR produced no non-empty segments")
    provenance = {
        "kind": "faster_whisper",
        "model": asr.model_identifier(result.model),
        "library_version": result.library_version,
        "compute_type": result.compute_type,
        "detected_language": result.language,
        "language_probability": result.language_probability,
    }
    return TranscriptResult(
        text="\n".join(segment.text for segment in segments),
        segments=segments,
        method="faster_whisper",
        backend=provenance,
        evidence_limit="local ASR-generated text; Lectern makes no transcript faithfulness claim",
        remote_services=_remote_services(
            transcriber_network_posture=(
                "local_inference_model_download_allowed"
                if allow_model_download
                else "local_inference_cached_model_only"
            ),
            lectern_invoked=allow_model_download,
        ),
        identity={
            "method": "faster_whisper",
            "backend": provenance,
            "allow_model_download": allow_model_download,
            "segments": [segment.to_dict() for segment in segments],
        },
        language=result.language,
    )


def _transcribe_with_local_command(
    command: str,
    normalized_audio: Path,
    duration_s: float | None,
) -> TranscriptResult:
    argv = _build_transcriber_argv(command, normalized_audio)
    try:
        result = subprocess.run(
            argv,
            check=False,
            capture_output=True,
            shell=False,
            timeout=DEFAULT_TRANSCRIBER_TIMEOUT_S,
        )
    except FileNotFoundError as exc:
        raise IngestError(f"local transcriber command not found: {argv[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise IngestError(
            f"local transcriber command timed out after {DEFAULT_TRANSCRIBER_TIMEOUT_S}s"
        ) from exc
    stdout = _decode_transcriber_pipe(result.stdout, "stdout")
    if result.returncode != 0:
        stderr = _decode_transcriber_pipe(result.stderr, "stderr")
        message = stderr.strip() or stdout.strip() or "no diagnostic output"
        raise IngestError(f"local transcriber command failed ({result.returncode}): {message}")

    payload = _parse_transcriber_json(stdout)
    segments, transcript_text = _segments_from_payload(payload, duration_s)
    command_digest = _command_identity_component(command)
    argv_digest = _digest_text(json.dumps(argv, separators=(",", ":")))
    transcript_digest = _digest_text(
        json.dumps(
            {
                "segments": [segment.to_dict() for segment in segments],
                "text": transcript_text,
            },
            sort_keys=True,
        )
    )
    return TranscriptResult(
        text=transcript_text,
        segments=tuple(segments),
        method="local_command_json",
        backend={
            "kind": "local_command",
            "argv0": Path(argv[0]).name,
            "command_sha256": command_digest,
            "argv_sha256": argv_digest,
            "input_argument_mode": (
                "placeholder" if _command_uses_placeholder(command) else "append"
            ),
            "timeout_s": DEFAULT_TRANSCRIBER_TIMEOUT_S,
        },
        evidence_limit=(
            "transcript supplied by user-configured local command; Lectern records "
            "timestamps but does not claim transcript faithfulness"
        ),
        remote_services=_remote_services(transcriber_network_posture="unverifiable_user_command"),
        identity={
            "method": "local_command_json",
            "command_sha256": command_digest,
            "transcript_sha256": transcript_digest,
        },
    )


def _decode_transcriber_pipe(data: bytes, stream_name: str) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise IngestError(f"local transcriber {stream_name} is not valid UTF-8") from exc


def _build_transcriber_argv(command: str, audio_path: Path) -> list[str]:
    argv = _split_transcriber_command(command)
    input_value = str(audio_path)
    if _command_uses_placeholder(command):
        return [part.replace("{input}", input_value) for part in argv]
    return [*argv, input_value]


def _split_transcriber_command(command: str) -> list[str]:
    if _looks_like_remote_endpoint(command):
        raise IngestError("local transcriber command must be a local executable, not a URL")
    try:
        argv = shlex.split(command)
    except ValueError as exc:
        raise IngestError(
            f"local transcriber command is not valid shell-style syntax: {exc}"
        ) from exc
    if not argv:
        raise IngestError("local transcriber command is empty")
    if any(_looks_like_remote_endpoint(part) for part in argv):
        raise IngestError("local transcriber command must not include remote endpoint URLs")
    return argv


def _looks_like_remote_endpoint(value: str) -> bool:
    lowered = value.strip().lower()
    return lowered.startswith(("http://", "https://"))


def _command_uses_placeholder(command: str) -> bool:
    return "{input}" in command


def _command_identity_component(command: str) -> str:
    argv = _split_transcriber_command(command)
    return _digest_text(json.dumps({"method": "local_command_json", "argv": argv}, sort_keys=True))


def _parse_transcriber_json(stdout: str) -> dict[str, Any]:
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise IngestError(f"local transcriber command did not emit valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise IngestError("local transcriber JSON must be an object")
    return cast(dict[str, Any], payload)


def _segments_from_payload(
    payload: dict[str, Any],
    duration_s: float | None,
) -> tuple[list[TranscriptSegment], str]:
    raw_segments = payload.get("segments")
    if raw_segments is None:
        raw_text = payload.get("text")
        if not isinstance(raw_text, str) or not raw_text.strip():
            raise IngestError("local transcriber JSON must include non-empty text or segments")
        text = raw_text.strip()
        return (
            [
                TranscriptSegment(
                    id=0,
                    start_s=0.0,
                    end_s=duration_s,
                    text=text,
                    source="local_command_text",
                )
            ],
            text,
        )
    if not isinstance(raw_segments, list):
        raise IngestError("local transcriber JSON field 'segments' must be a list")

    segments: list[TranscriptSegment] = []
    for index, raw_segment in enumerate(cast(list[Any], raw_segments)):
        if not isinstance(raw_segment, dict):
            raise IngestError("local transcriber segment must be an object")
        segment_payload = cast(dict[str, Any], raw_segment)
        raw_text = segment_payload.get("text")
        if not isinstance(raw_text, str) or not raw_text.strip():
            raise IngestError("local transcriber segment text must be a non-empty string")
        start_s = _strict_number(segment_payload.get("start_s"), "start_s")
        end_value = segment_payload.get("end_s")
        end_s = None if end_value is None else _strict_number(end_value, "end_s")
        if end_s is not None and end_s < start_s:
            raise IngestError("local transcriber segment end_s must be greater than start_s")
        segments.append(
            TranscriptSegment(
                id=index,
                start_s=start_s,
                end_s=end_s,
                text=raw_text.strip(),
                source="local_command",
            )
        )
    if not segments:
        raise IngestError("local transcriber JSON must include at least one segment")

    raw_text = payload.get("text")
    if isinstance(raw_text, str) and raw_text.strip():
        transcript_text = raw_text.strip()
    else:
        transcript_text = "\n".join(segment.text for segment in segments)
    return segments, transcript_text


def _strict_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise IngestError(f"local transcriber segment {field_name} must be a number")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise IngestError(f"local transcriber segment {field_name} must be finite and non-negative")
    return parsed


def _normalize_to_canonical_wav(source: Path, output: Path) -> None:
    if _is_canonical_wav(source):
        shutil.copyfile(source, output)
        return

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise IngestError("ffmpeg is required to normalize non-canonical audio")

    temp_output = output.with_suffix(".tmp.wav")
    result = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-ac",
            str(CANONICAL_CHANNELS),
            "-ar",
            str(CANONICAL_SAMPLE_RATE),
            str(temp_output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or "unknown ffmpeg failure"
        temp_output.unlink(missing_ok=True)
        raise IngestError(f"audio normalization failed: {message}")
    temp_output.replace(output)


def _is_canonical_wav(path: Path) -> bool:
    try:
        with wave.open(str(path), "rb") as audio:
            return (
                audio.getnchannels() == CANONICAL_CHANNELS
                and audio.getframerate() == CANONICAL_SAMPLE_RATE
                and audio.getsampwidth() == CANONICAL_SAMPLE_WIDTH
            )
    except (EOFError, wave.Error):
        return False


def _wav_duration_seconds(path: Path) -> float | None:
    try:
        with wave.open(str(path), "rb") as audio:
            frame_rate = audio.getframerate()
            if frame_rate <= 0:
                return None
            return audio.getnframes() / frame_rate
    except (EOFError, wave.Error) as exc:
        raise IngestError(f"normalized audio is not readable WAV: {path}") from exc


def _summary_lite(transcript: TranscriptResult, source: Source) -> str:
    word_count = sum(len(segment.text.split()) for segment in transcript.segments)
    start = min(segment.start_s for segment in transcript.segments)
    end = (
        max(segment.end_s for segment in transcript.segments if segment.end_s is not None)
        if all(segment.end_s is not None for segment in transcript.segments)
        else None
    )
    return (
        f"# {source.title or 'Untitled talk'}\n\n"
        f"Duration: {format_time(source.duration_s, duration=True)}\n"
        f"Transcript method: {transcript.method}\n"
        f"Segments: {len(transcript.segments)}\n"
        f"Words: {word_count}\n"
        f"Time span: {format_time(start)}–{format_time(end)}\n\n"
        "Lectern does not summarize. Read the transcript with `lectern transcript <bundle-id>` "
        "or hand it to an agent.\n"
    )


def _transcript_metadata(
    *,
    transcript: TranscriptResult,
    source: Path,
    source_digest: str,
    normalized_audio: Path,
) -> dict[str, Any]:
    audio_digest, audio_size = _digest_and_size(normalized_audio)
    return {
        "schema": "lectern.transcript.metadata.v0",
        "generated_at": datetime.now(UTC).isoformat(),
        "method": transcript.method,
        "language": transcript.language,
        "backend": transcript.backend,
        "remote_services": transcript.remote_services,
        "evidence_limit": transcript.evidence_limit,
        "source_media": {
            "sha256": source_digest,
        },
        "normalized_audio": {
            "path": "media/audio.wav",
            "sha256": audio_digest,
            "bytes": audio_size,
        },
        "artifacts": {
            "segments": "transcript/segments.json",
            "transcript": "transcript/transcript.md",
            "summary": "analysis/summary.md",
        },
        "schema_contract": {
            "manifest_schema_versioned": False,
            "note": (
                "M3 transcript metadata is a local bundle artifact, not a manifest schema field."
            ),
        },
    }


def _remote_services(
    *, transcriber_network_posture: str, lectern_invoked: bool = False
) -> dict[str, Any]:
    return {
        "allowed": False,
        "scope": "lectern_core",
        "lectern_invoked": lectern_invoked,
        "requires_explicit_per_item_consent": True,
        "transcriber_network_posture": transcriber_network_posture,
    }


def _done_stage(bundle_dir: Path, outputs: Sequence[Path]) -> StageRecord:
    now = datetime.now(UTC)
    return StageRecord(
        state=StageState.DONE,
        started=now,
        finished=now,
        outputs=[_artifact_ref(bundle_dir, output) for output in outputs],
    )


def _artifact_ref(bundle_dir: Path, path: Path) -> ArtifactRef:
    digest, size = _digest_and_size(path)
    return ArtifactRef(
        path=path.relative_to(bundle_dir).as_posix(),
        sha256=digest,
        bytes=size,
    )


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _combined_digest(*digests: str) -> str:
    digest = hashlib.sha256()
    for value in digests:
        digest.update(value.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _digest_and_size(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return digest.hexdigest(), size


def _slug(value: str) -> str:
    chars = [char.lower() if char.isalnum() else "-" for char in value]
    slug = "-".join(part for part in "".join(chars).split("-") if part)
    return slug or "bundle"
