"""Optional local diarization, pinned user-fetched models, and anonymous labels."""

from __future__ import annotations

import hashlib
import importlib
import math
import os
import tempfile
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from lectern.asr import pcm16_mono_samples
from lectern.bundle import DiarizationDocument, DiarizationTurn, SegmentSpeaker
from lectern.records import redact_paths

DEFAULT_DIARIZE_MODELS = Path(".lectern") / "cache" / "diarize"
DEFAULT_SPEAKER_THRESHOLD = 1.0
DIARIZATION_METHOD = "sherpa-onnx pyannote-seg-3.0 + wespeaker-resnet34"
SPEAKER_EVIDENCE_LIMIT = (
    "speaker labels are anonymous clusters (S1, S2, ...), not identities; "
    "overlapping speech is attributed to one speaker; "
    "similar voices may merge and one voice may split"
)


class DiarizationError(RuntimeError):
    """Diarization cannot complete with the requested local configuration."""


@dataclass(frozen=True)
class ModelFile:
    name: str
    url: str
    sha256: str
    bytes: int


MODEL_FILES = (
    ModelFile(
        "segmentation.onnx",
        "https://huggingface.co/csukuangfj/sherpa-onnx-pyannote-segmentation-3-0/resolve/main/model.onnx",
        "220ad67ca923bef2fa91f2390c786097bf305bceb5e261d4af67b38e938e1079",
        5_992_913,
    ),
    ModelFile(
        "embedding.onnx",
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
        "speaker-recongition-models/wespeaker_en_voxceleb_resnet34_LM.onnx",
        "e9848563da86f263117134dfd7ad63c92355b37de492b55e325400c9d9c39012",
        26_530_550,
    ),
)


def validate_diarization_options(speakers: int | None, threshold: float) -> None:
    if speakers is not None and (isinstance(speakers, bool) or speakers < 1):
        raise DiarizationError("--speakers must be a positive integer")
    if not math.isfinite(threshold) or threshold <= 0:
        raise DiarizationError("--speaker-threshold must be finite and positive")


def _verify_model(path: Path, model: ModelFile) -> None:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    if digest.hexdigest() != model.sha256 or size != model.bytes:
        raise DiarizationError(f"diarization model verification failed: {model.name} (sha256/size)")


class _HTTPSRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        if not newurl.startswith("https://"):
            raise DiarizationError("diarization model download requires HTTPS")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _download_model(directory: Path, model: ModelFile) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        opener = urllib.request.build_opener(_HTTPSRedirectHandler())
        with (
            opener.open(model.url, timeout=60) as response,
            tempfile.NamedTemporaryFile(
                prefix=f".{model.name}.", dir=directory, delete=False
            ) as stream,
        ):
            temporary = Path(stream.name)
            size = 0
            while chunk := response.read(1024 * 1024):
                size += len(chunk)
                if size > model.bytes:
                    raise DiarizationError(
                        f"diarization model verification failed: {model.name} (size)"
                    )
                stream.write(chunk)
        _verify_model(temporary, model)
        os.replace(temporary, directory / model.name)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def ensure_models(directory: Path, *, allow_model_download: bool = False) -> Path:
    """Resolve only pinned local files unless the caller explicitly allows fetching."""

    directory = directory.expanduser()
    try:
        for model in MODEL_FILES:
            path = directory / model.name
            if not path.is_file():
                if not allow_model_download:
                    raise DiarizationError(
                        f"missing diarization model: {model.name}; "
                        "rerun with --diarize --allow-model-download to fetch verified models, "
                        "or save the model from "
                        + model.url
                        + " as "
                        + model.name
                        + " in the directory selected by --diarize-models DIR"
                    )
                _download_model(directory, model)
            else:
                _verify_model(path, model)
    except DiarizationError:
        raise
    except Exception as exc:
        raise DiarizationError(f"diarization models: {redact_paths(str(exc))}") from exc
    return directory


@dataclass(frozen=True)
class SpeakerTurn:
    start_s: float
    end_s: float
    speaker: int


class Diarizer(Protocol):
    def diarize(self, audio: Path) -> tuple[SpeakerTurn, ...]: ...


class _SherpaTurn(Protocol):
    start: float
    end: float
    speaker: int


def _load_sherpa() -> Any:
    """Lazy import boundary, also guarded against real model loading in unit tests."""

    try:
        return importlib.import_module("sherpa_onnx")
    except ImportError as exc:
        raise DiarizationError(
            "local diarization needs sherpa-onnx; install with pip install 'lectern[diarize]' "
            "or uv sync --extra diarize"
        ) from exc


class SherpaOnnxDiarizer:
    def __init__(self, models_dir: Path, *, speakers: int | None, threshold: float) -> None:
        sherpa = _load_sherpa()
        config = sherpa.OfflineSpeakerDiarizationConfig(
            segmentation=sherpa.OfflineSpeakerSegmentationModelConfig(
                pyannote=sherpa.OfflineSpeakerSegmentationPyannoteModelConfig(
                    model=str(models_dir / "segmentation.onnx")
                ),
                num_threads=2,
            ),
            embedding=sherpa.SpeakerEmbeddingExtractorConfig(
                model=str(models_dir / "embedding.onnx"),
                num_threads=2,
            ),
            clustering=sherpa.FastClusteringConfig(
                num_clusters=speakers if speakers is not None else -1,
                threshold=threshold,
            ),
            min_duration_on=0.3,
            min_duration_off=0.5,
        )
        if not config.validate():
            raise DiarizationError("invalid local diarization model configuration")
        self._model: Any = sherpa.OfflineSpeakerDiarization(config)

    def diarize(self, audio: Path) -> tuple[SpeakerTurn, ...]:
        if self._model.sample_rate != 16_000:
            raise DiarizationError("diarization expects a 16 kHz model")
        turns = cast(
            Sequence[_SherpaTurn],
            self._model.process(pcm16_mono_samples(audio)).sort_by_start_time(),
        )
        return tuple(SpeakerTurn(turn.start, turn.end, turn.speaker) for turn in turns)


def create_diarizer(
    models_dir: Path = DEFAULT_DIARIZE_MODELS,
    *,
    speakers: int | None = None,
    threshold: float = DEFAULT_SPEAKER_THRESHOLD,
    allow_model_download: bool = False,
) -> Diarizer:
    validate_diarization_options(speakers, threshold)
    directory = ensure_models(models_dir, allow_model_download=allow_model_download)
    return SherpaOnnxDiarizer(directory, speakers=speakers, threshold=threshold)


class _Segment(Protocol):
    @property
    def id(self) -> int: ...
    @property
    def start_s(self) -> float: ...
    @property
    def end_s(self) -> float | None: ...


def diarization_document(
    turns: Sequence[SpeakerTurn],
    segments: Sequence[_Segment],
    *,
    speakers: int | None,
) -> DiarizationDocument:
    """Normalize cluster ids and assign the greatest total overlap per segment."""

    labels: dict[int, str] = {}
    labelled: list[DiarizationTurn] = []
    for turn in sorted(turns, key=lambda turn: turn.start_s):
        label = labels.setdefault(turn.speaker, f"S{len(labels) + 1}")
        labelled.append(DiarizationTurn(start_s=turn.start_s, end_s=turn.end_s, speaker=label))
    last_end = max((turn.end_s for turn in labelled), default=0.0)
    assignments: list[SegmentSpeaker] = []
    for index, segment in enumerate(segments):
        end = segment.end_s
        if end is None:
            end = segments[index + 1].start_s if index + 1 < len(segments) else last_end
        overlap = dict.fromkeys(labels.values(), 0.0)
        for turn in labelled:
            overlap[turn.speaker] += max(
                0.0, min(end, turn.end_s) - max(segment.start_s, turn.start_s)
            )
        # Insertion order is first appearance; max retains that order on ties.
        speaker = max(overlap, key=lambda label: overlap[label]) if overlap else None
        if speaker is not None and overlap[speaker] <= 0:
            speaker = None
        assignments.append(SegmentSpeaker(segment_id=segment.id, speaker=speaker))
    return DiarizationDocument(
        method=DIARIZATION_METHOD,
        speaker_count=speakers if speakers is not None else len(labels),
        count_source="given" if speakers is not None else "estimated",
        turns=labelled,
        segment_speakers=assignments,
    )
