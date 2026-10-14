# Modified for Lectern, 2026-10-13: retain window constants only; embeddings run in Lectern glue.
# Original: diarize 0.1.2 (Apache-2.0).
"""Upstream embedding-window constants (seconds)."""

# ── Constants ────────────────────────────────────────────────────────────────

#: Minimum segment duration for embedding extraction (seconds).
#: Segments shorter than this are skipped during embedding extraction
#: and later assigned the nearest speaker label.
MIN_SEGMENT_DURATION: float = 0.4

#: Sliding window length for splitting long segments (seconds).
EMBEDDING_WINDOW: float = 1.2

#: Sliding window step size (seconds).  Overlap = WINDOW − STEP.
EMBEDDING_STEP: float = 0.6
