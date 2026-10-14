"""Voice Activity Detection using Silero VAD.

Detects speech segments in audio files. Returns a list of
:class:`~diarize.utils.SpeechSegment` objects with start/end timestamps.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .utils import SpeechSegment

logger = logging.getLogger(__name__)

__all__ = ["run_vad"]


def run_vad(
    audio_path: str | Path,
    *,
    threshold: float = 0.45,
    min_speech_duration_ms: int = 200,
    min_silence_duration_ms: int = 50,
    speech_pad_ms: int = 20,
) -> list[SpeechSegment]:
    """Detect speech segments using Silero VAD.

    Args:
        audio_path: Path to the audio file.
        threshold: VAD probability threshold (0.0 to 1.0).
            Higher values produce fewer, more confident detections.
        min_speech_duration_ms: Minimum speech segment duration in
            milliseconds. Segments shorter than this are discarded.
        min_silence_duration_ms: Minimum silence duration in milliseconds
            required to split speech into separate segments.
        speech_pad_ms: Padding added around each detected speech
            segment in milliseconds.

    Returns:
        List of :class:`SpeechSegment` with timestamps in seconds,
        sorted by start time.

    Example::

        segments = run_vad("meeting.wav")
        for seg in segments:
            print(f"Speech: {seg.start:.2f}s - {seg.end:.2f}s ({seg.duration:.2f}s)")
    """
    from silero_vad import get_speech_timestamps, load_silero_vad, read_audio

    logger.info("Running Voice Activity Detection (Silero VAD)...")

    vad_model = load_silero_vad()
    wav = read_audio(str(audio_path))  # 1-D tensor, 16 kHz

    speech_timestamps: list[dict[str, float]] = get_speech_timestamps(
        wav,
        vad_model,
        sampling_rate=16000,
        threshold=threshold,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
        speech_pad_ms=speech_pad_ms,
        return_seconds=True,
    )

    segments = [SpeechSegment(start=ts["start"], end=ts["end"]) for ts in speech_timestamps]

    total_speech = sum(seg.duration for seg in segments)
    logger.info(
        "VAD complete: %d speech segments, %.1f seconds of speech",
        len(segments),
        total_speech,
    )

    return segments
