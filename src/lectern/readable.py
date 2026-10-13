"""Plain time labels shared by the overview and readable CLI views."""

from __future__ import annotations


def format_time(seconds: float | None, *, duration: bool = False) -> str:
    if seconds is None:
        return "unknown"
    sign = "-" if seconds < 0 else ""
    hours, remainder = divmod(int(abs(seconds)), 3600)
    minutes, seconds_part = divmod(remainder, 60)
    if hours or duration:
        return f"{sign}{hours}:{minutes:02d}:{seconds_part:02d}"
    return f"{sign}{minutes:02d}:{seconds_part:02d}"
