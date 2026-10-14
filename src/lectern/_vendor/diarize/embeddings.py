"""Speaker embedding extraction using WeSpeaker ResNet34-LM (ONNX).

Extracts 256-dimensional speaker embeddings from audio segments detected
by VAD.  Long segments are split with a sliding window so that each
window produces its own embedding, improving clustering granularity.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from .utils import SpeechSegment, SubSegment

logger = logging.getLogger(__name__)

__all__ = ["extract_embeddings"]

# ── Constants ────────────────────────────────────────────────────────────────

#: Minimum segment duration for embedding extraction (seconds).
#: Segments shorter than this are skipped during embedding extraction
#: and later assigned the nearest speaker label.
MIN_SEGMENT_DURATION: float = 0.4

#: Sliding window length for splitting long segments (seconds).
EMBEDDING_WINDOW: float = 1.2

#: Sliding window step size (seconds).  Overlap = WINDOW − STEP.
EMBEDDING_STEP: float = 0.6


def extract_embeddings(
    audio_path: str | Path,
    speech_segments: list[SpeechSegment],
) -> tuple[np.ndarray, list[SubSegment]]:
    """Extract 256-dim speaker embeddings using WeSpeaker ResNet34-LM (ONNX).

    Long segments are split using a sliding window for more accurate
    clustering.  Each window produces its own embedding.

    Args:
        audio_path: Path to the audio file (wav, mp3, flac, etc.).
        speech_segments: Speech segments detected by VAD.

    Returns:
        A ``(embeddings, subsegments)`` tuple where:

        -   **embeddings** --- ``np.ndarray`` of shape ``(N, 256)`` with
            raw speaker embeddings (not yet L2-normalised; normalisation
            is applied later during clustering).
        -   **subsegments** --- list of :class:`SubSegment` objects that
            record the time window and parent segment index for each
            embedding row.

    Raises:
        FileNotFoundError: If *audio_path* does not exist.

    Example::

        from diarize.vad import run_vad
        from diarize.embeddings import extract_embeddings

        segments = run_vad("meeting.wav")
        embeddings, subs = extract_embeddings("meeting.wav", segments)
        print(embeddings.shape)  # (N, 256)
    """
    import wespeakerruntime as wespeaker_rt

    logger.info("Extracting speaker embeddings (WeSpeaker ResNet34-LM, 256-dim)...")

    model = wespeaker_rt.Speaker(lang="en")

    # Load full audio for segment slicing
    audio_data, sr = sf.read(str(audio_path))
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)  # stereo → mono

    embeddings: list[np.ndarray] = []
    subsegments: list[SubSegment] = []

    for idx, seg in enumerate(speech_segments):
        seg_duration = seg.duration

        if seg_duration < MIN_SEGMENT_DURATION:
            continue

        # Split long segments with a sliding window
        if seg_duration <= EMBEDDING_WINDOW * 1.5:
            windows = [(seg.start, seg.end)]
        else:
            windows: list[tuple[float, float]] = []
            win_start = seg.start
            while win_start + MIN_SEGMENT_DURATION < seg.end:
                win_end = min(win_start + EMBEDDING_WINDOW, seg.end)
                windows.append((win_start, win_end))
                win_start += EMBEDDING_STEP

        for win_start, win_end in windows:
            start_sample = int(win_start * sr)
            end_sample = int(win_end * sr)
            segment_audio = audio_data[start_sample:end_sample]

            tmp_path: str | None = None
            try:
                # wespeakerruntime accepts file paths — write segment to a temp wav
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp_path = tmp.name
                    sf.write(tmp_path, segment_audio, sr)

                emb = model.extract_embedding(tmp_path)
            except Exception:
                logger.debug(
                    "Embedding extraction failed for window %.2f-%.2f",
                    win_start,
                    win_end,
                )
                continue
            finally:
                if tmp_path is not None:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

            if emb is not None:
                if emb.ndim == 2:
                    emb = emb[0]
                embeddings.append(emb)
                subsegments.append(SubSegment(start=win_start, end=win_end, parent_idx=idx))

    if not embeddings:
        return np.empty((0, 256), dtype=np.float32), []

    X = np.stack(embeddings)  # (N, 256)
    logger.info("Extracted %d embeddings (dim=%d)", X.shape[0], X.shape[1])
    return X, subsegments
