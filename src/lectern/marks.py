"""Minimal time-bound gestures: parsing, content binding, and shared display."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from lectern.bundle import TranscriptSegmentRecord
from lectern.readable import format_time
from lectern.search import make_anchor

_TIME = re.compile(
    r"(?:(?P<h>[0-9]+):)?(?P<m>[0-5][0-9]):(?P<s>[0-5][0-9])(?:\.(?P<ms>[0-9]{1,3}))?"
)
_KINDS = {"?": "question", "!": "important", "->": "follow_up", "→": "follow_up"}
_SYMBOLS = {"question": "?", "important": "!", "follow_up": "->"}


@dataclass(frozen=True)
class ParsedMark:
    t_ms: int
    kind: str | None
    text: str


def parse_mark(line: str, duration_s: float | None) -> ParsedMark | None:
    """Ignore blanks/comments; reject a malformed gesture with a path-free reason."""

    line = line.strip()
    if not line or line.startswith("#"):
        return None
    parts = line.split(maxsplit=1)
    match = _TIME.fullmatch(parts[0])
    if match is None:
        raise ValueError("invalid time; expected [H:]MM:SS[.fff]")
    try:
        t_ms = (
            int(match["h"] or "0") * 3600 + int(match["m"]) * 60 + int(match["s"])
        ) * 1000 + int((match["ms"] or "").ljust(3, "0"))
        t_s = t_ms / 1000
    except (ValueError, OverflowError) as exc:
        raise ValueError("invalid time; numeric component is too large") from exc
    if duration_s is not None and t_s > duration_s:
        raise ValueError("time exceeds bundle duration")
    body = parts[1].strip() if len(parts) == 2 else ""
    fields = body.split(maxsplit=1)
    kind = _KINDS.get(fields[0]) if fields else None
    if kind is not None:
        body = fields[1].strip() if len(fields) == 2 else ""
    elif not body:
        raise ValueError("text is required when no kind is given")
    elif len(fields) == 1 and not any(char.isalnum() for char in body):
        raise ValueError("unknown kind symbol; expected ?, !, -> or →")
    return ParsedMark(t_ms=t_ms, kind=kind, text=body)


def bind_mark(
    bundle_id: str, mark: ParsedMark, segments: Sequence[TranscriptSegmentRecord]
) -> dict[str, Any]:
    """Keep gesture time distinct from its bound segment's citation time."""

    t_s = mark.t_ms / 1000
    segment = next(
        (s for s in segments if s.end_s is not None and s.start_s <= t_s < s.end_s), None
    )
    binding = "exact" if segment is not None else "none"
    if segment is None:
        # Construct exact decimal bounds from integer milliseconds, without a tolerance.
        earliest_end = Decimal(f"{mark.t_ms - 5000}e-3")
        latest_end = Decimal(f"{mark.t_ms}e-3")
        preceding = [
            s
            for s in segments
            if s.end_s is not None and earliest_end <= Decimal(str(s.end_s)) <= latest_end
        ]
        if preceding:
            segment = max(preceding, key=lambda s: (s.end_s, s.start_s, s.id))
            binding = "nearest"
    digest_input = f"{bundle_id}\0{mark.t_ms}\0{mark.kind or ''}\0{mark.text}"
    anchor: dict[str, Any] = {
        "segment_id": None,
        "start_s": None,
        "text_sha256": None,
        "canon_version": None,
    }
    if segment is not None:
        anchor = make_anchor(bundle_id, segment.id, segment.start_s, segment.text).to_dict()
    return {
        **anchor,
        "bundle_id": bundle_id,
        "mark_id": hashlib.sha256(digest_input.encode("utf-8")).hexdigest()[:16],
        "t_s": t_s,
        "kind": mark.kind,
        "text": mark.text,
        "binding": binding,
    }


def mark_cite(mark: dict[str, Any], cite_id: str) -> str | None:
    if mark["segment_id"] is None:
        return None
    return f"[t={format_time(mark['start_s'])} {cite_id}#{mark['segment_id']}]"


def mark_line(t_s: float, kind: str | None, text: str, cite: str | None) -> str:
    prefix = cite if cite is not None else format_time(t_s)
    body = " ".join(part for part in (_SYMBOLS.get(kind or "", ""), text) if part)
    return f"{prefix} {body}".rstrip() + (" (unbound)" if cite is None else "")
