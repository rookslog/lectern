"""Validation of user-supplied transcription hints; never infer terms from media."""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence
from pathlib import Path


def normalize_terms(terms: Sequence[str]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for position, raw in enumerate(terms, 1):
        term = raw.strip()
        if any(unicodedata.category(char) == "Cc" or char in "\u2028\u2029" for char in raw):
            raise ValueError(f"term {position}: control characters and newlines are not allowed")
        if not 1 <= len(term) <= 64:
            raise ValueError(f"term {position}: must contain 1–64 characters after stripping")
        if term not in seen:
            if len(result) == 50:
                raise ValueError(f"term {position}: at most 50 distinct terms are allowed")
            result.append(term)
            seen.add(term)
    return tuple(result)


def load_terms(terms: Sequence[str], files: Sequence[Path]) -> tuple[str, ...]:
    combined = list(terms)
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise ValueError(f"cannot read UTF-8 terms file {path}: {exc}") from exc
        # Split only physical lines, so embedded Unicode controls are validated
        # as part of a term rather than silently becoming line boundaries.
        combined.extend(
            line for line in text.split("\n") if line.strip() and not line.lstrip().startswith("#")
        )
    return normalize_terms(combined)
