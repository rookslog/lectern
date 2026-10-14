# Modified for Lectern, 2026-10-13: retain VAD defaults only; Silero ONNX runs in Lectern glue.
# Original: diarize 0.1.2 (Apache-2.0).
"""Upstream Silero VAD defaults."""

VAD_THRESHOLD = 0.45
MIN_SPEECH_DURATION_MS = 200
MIN_SILENCE_DURATION_MS = 50
SPEECH_PAD_MS = 20
