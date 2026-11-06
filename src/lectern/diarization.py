"""Optional local diarization, pinned user-fetched models, and anonymous labels."""

from __future__ import annotations

import hashlib
import importlib
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
DIARIZATION_METHOD = (
    "silero-vad-v6 + wespeaker-resnet34 (kaldi fbank, CMN) + spectral clustering (diarize 0.1.2)"
)
DIARIZATION_IMPORTS = ("sherpa_onnx", "onnxruntime", "kaldi_native_fbank", "sklearn", "numpy")
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
        "silero_vad.onnx",
        "https://github.com/snakers4/silero-vad/raw/v6.2.3/src/silero_vad/data/silero_vad.onnx",
        "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3",
        2_327_524,
    ),
    ModelFile(
        "embedding.onnx",
        "https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet34-LM/resolve/main/"
        "voxceleb_resnet34_LM.onnx",
        "7bb2f06e9df17cdf1ef14ee8a15ab08ed28e8d0ef5054ee135741560df2ec068",
        26_530_309,
    ),
)


def validate_diarization_options(speakers: int | None) -> None:
    if speakers is not None and (isinstance(speakers, bool) or speakers < 1):
        raise DiarizationError("--speakers must be a positive integer")


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


class _PipelineTurn(Protocol):
    start: float
    end: float
    speaker: str


def _load_dependency(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise DiarizationError(
            f"local diarization needs {name}; install with pip install 'lectern[diarize]' "
            "or uv sync --extra diarize"
        ) from exc


def _load_sherpa() -> Any:
    """Lazy import boundary, also guarded against real model loading in unit tests."""

    return _load_dependency("sherpa_onnx")


class SileroSpectralDiarizer:
    def __init__(self, models_dir: Path, *, speakers: int | None) -> None:
        # Keep this first: the unit-suite guard prevents either real model loading.
        self._sherpa: Any = _load_sherpa()
        ort = _load_dependency("onnxruntime")
        self._fbank: Any = _load_dependency("kaldi_native_fbank")
        self._numpy: Any = _load_dependency("numpy")
        self._pipeline: Any = _load_dependency("lectern._vendor.diarize")
        self._types: Any = _load_dependency("lectern._vendor.diarize.utils")
        self._windows: Any = _load_dependency("lectern._vendor.diarize.embeddings")
        self._vad_defaults: Any = _load_dependency("lectern._vendor.diarize.vad")
        self._speakers = speakers

        config = self._sherpa.VadModelConfig()
        config.silero_vad.model = str(models_dir / "silero_vad.onnx")
        config.silero_vad.threshold = self._vad_defaults.VAD_THRESHOLD
        config.silero_vad.min_speech_duration = self._vad_defaults.MIN_SPEECH_DURATION_MS / 1000
        config.silero_vad.min_silence_duration = self._vad_defaults.MIN_SILENCE_DURATION_MS / 1000
        config.silero_vad.window_size = 512
        config.sample_rate = 16_000
        config.num_threads = 2
        self._vad_config: Any = config

        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self._session: Any = ort.InferenceSession(
            str(models_dir / "embedding.onnx"),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )

        features = self._fbank.FbankOptions()
        features.frame_opts.dither = 0.0
        features.frame_opts.samp_freq = 16_000
        features.frame_opts.frame_length_ms = 25
        features.frame_opts.frame_shift_ms = 10
        features.frame_opts.window_type = "hamming"
        features.mel_opts.num_bins = 80
        features.use_energy = False
        self._feature_options: Any = features

    def _speech_segments(self, samples: Any, duration: float) -> list[Any]:
        detector = self._sherpa.VoiceActivityDetector(
            self._vad_config, buffer_size_in_seconds=duration + 1.0
        )
        segments: list[Any] = []
        pad = self._vad_defaults.SPEECH_PAD_MS / 1000

        def drain() -> None:
            while not detector.empty():
                segment = detector.front
                start = segment.start / 16_000
                end = start + len(segment.samples) / 16_000
                segments.append(
                    self._types.SpeechSegment(
                        start=min(duration, max(0.0, start - pad)),
                        end=max(0.0, min(duration, end + pad)),
                    )
                )
                detector.pop()

        for offset in range(0, len(samples), 512):
            detector.accept_waveform(samples[offset : offset + 512])
            drain()
        detector.flush()
        drain()
        return segments

    def _extract_embeddings(
        self, samples: Any, speech_segments: list[Any]
    ) -> tuple[Any, list[Any]]:
        embeddings: list[Any] = []
        subsegments: list[Any] = []
        for parent_idx, segment in enumerate(speech_segments):
            if segment.duration < self._windows.MIN_SEGMENT_DURATION:
                continue
            if segment.duration <= self._windows.EMBEDDING_WINDOW * 1.5:
                windows = [(segment.start, segment.end)]
            else:
                windows: list[tuple[float, float]] = []
                start = segment.start
                while start + self._windows.MIN_SEGMENT_DURATION < segment.end:
                    windows.append(
                        (start, min(start + self._windows.EMBEDDING_WINDOW, segment.end))
                    )
                    start += self._windows.EMBEDDING_STEP

            for start, end in windows:
                window = samples[int(start * 16_000) : int(end * 16_000)]
                fbank = self._fbank.OnlineFbank(self._feature_options)
                fbank.accept_waveform(16_000, (window * 32768).tolist())
                fbank.input_finished()
                if fbank.num_frames_ready == 0:
                    continue
                feats = self._numpy.array(
                    [fbank.get_frame(i) for i in range(fbank.num_frames_ready)],
                    dtype=self._numpy.float32,
                )
                feats -= feats.mean(axis=0)
                embedding = self._session.run(["embs"], {"feats": feats[None]})[0][0]
                embeddings.append(embedding)
                subsegments.append(
                    self._types.SubSegment(start=start, end=end, parent_idx=parent_idx)
                )

        if not embeddings:
            return self._numpy.empty((0, 256), dtype=self._numpy.float32), []
        return self._numpy.stack(embeddings), subsegments

    def diarize(self, audio: Path) -> tuple[SpeakerTurn, ...]:
        samples = cast(Any, pcm16_mono_samples(audio))
        duration = len(samples) / 16_000
        speech_segments = self._speech_segments(samples, duration)
        embeddings, subsegments = self._extract_embeddings(samples, speech_segments)
        turns = cast(
            Sequence[_PipelineTurn],
            self._pipeline.diarize_embeddings(
                speech_segments, embeddings, subsegments, duration, num_speakers=self._speakers
            ),
        )
        return tuple(
            SpeakerTurn(turn.start, turn.end, int(turn.speaker.removeprefix("SPEAKER_")))
            for turn in turns
        )


def create_diarizer(
    models_dir: Path = DEFAULT_DIARIZE_MODELS,
    *,
    speakers: int | None = None,
    allow_model_download: bool = False,
) -> Diarizer:
    validate_diarization_options(speakers)
    directory = ensure_models(models_dir, allow_model_download=allow_model_download)
    return SileroSpectralDiarizer(directory, speakers=speakers)


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
