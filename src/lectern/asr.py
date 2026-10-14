"""Optional local ASR boundary; importing Lectern does not load model weights."""

from __future__ import annotations

import importlib
import wave
from collections.abc import Iterable
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Protocol, cast


@dataclass(frozen=True)
class ASRSegment:
    start_s: float
    end_s: float
    text: str


@dataclass(frozen=True)
class ASRResult:
    segments: tuple[ASRSegment, ...]
    language: str
    language_probability: float
    model: str
    library_version: str | None
    compute_type: str


class ASRBackend(Protocol):
    def transcribe(self, audio: Path, *, language: str | None = None) -> ASRResult: ...


class _WhisperSegment(Protocol):
    start: float
    end: float
    text: str


class _WhisperInfo(Protocol):
    language: str
    language_probability: float


class _WhisperModel(Protocol):
    def transcribe(
        self, audio: object, *, language: str | None
    ) -> tuple[Iterable[_WhisperSegment], _WhisperInfo]: ...


class _NumpyArray(Protocol):
    def astype(self, dtype: str) -> _NumpyArray: ...
    def __truediv__(self, other: float) -> _NumpyArray: ...


class _NumpyModule(Protocol):
    def frombuffer(self, buffer: bytes, dtype: str) -> _NumpyArray: ...


def pcm16_mono_samples(audio: Path) -> _NumpyArray:
    """Read Lectern's canonical WAV (16 kHz, mono, 16-bit) as float32 in [-1, 1)."""

    with wave.open(str(audio), "rb") as reader:
        if (reader.getframerate(), reader.getnchannels(), reader.getsampwidth()) != (16_000, 1, 2):
            raise ValueError(f"ASR expects 16 kHz mono 16-bit WAV: {audio.name}")
        frames = reader.readframes(reader.getnframes())
    numpy = cast(_NumpyModule, importlib.import_module("numpy"))
    return numpy.frombuffer(frames, dtype="<i2").astype("float32") / 32768.0


class _WhisperModule(Protocol):
    def WhisperModel(
        self, model: str, *, device: str, compute_type: str, local_files_only: bool
    ) -> _WhisperModel: ...


def model_identifier(model: str) -> str:
    """Keep model provenance useful without recording a local filesystem path."""

    path = Path(model.replace("\\", "/")).expanduser()
    if "/" in model or "\\" in model or path.is_dir():
        return f"local:{path.name or path.resolve().name}"
    return model


class FasterWhisperBackend:
    """CPU/int8 inference; fetching model weights requires explicit opt-in."""

    def __init__(self, model: str = "small", *, allow_model_download: bool = False) -> None:
        module = cast(_WhisperModule, importlib.import_module("faster_whisper"))
        self.model = model
        self.compute_type = "int8"
        self._model = module.WhisperModel(
            model,
            device="cpu",
            compute_type=self.compute_type,
            local_files_only=not allow_model_download,
        )
        try:
            self.library_version = version("faster-whisper")
        except PackageNotFoundError:
            self.library_version = None

    def transcribe(self, audio: Path, *, language: str | None = None) -> ASRResult:
        # Hand faster-whisper decoded samples, not a path: its own decoder calls
        # PyAV with an argument PyAV 19 removed, and Lectern's normalized audio is
        # already 16 kHz mono 16-bit PCM, which is exactly what the model expects.
        segments, info = self._model.transcribe(pcm16_mono_samples(audio), language=language)
        return ASRResult(
            segments=tuple(
                ASRSegment(segment.start, segment.end, segment.text.strip())
                for segment in segments
                if segment.text.strip()
            ),
            language=info.language,
            language_probability=info.language_probability,
            model=self.model,
            library_version=self.library_version,
            compute_type=self.compute_type,
        )


def create_asr_backend(model: str = "small", *, allow_model_download: bool = False) -> ASRBackend:
    return FasterWhisperBackend(model, allow_model_download=allow_model_download)
