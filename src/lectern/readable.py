"""Plain time labels shared by the overview and readable CLI views."""

from __future__ import annotations

from lectern.bundle import Source


def plain_field(value: str) -> str:
    """Keep one field on one terminal row without changing its evidence."""

    return "".join(
        f"\\u{ord(character):04x}"
        if ord(character) < 0x20 or 0x7F <= ord(character) <= 0x9F or character in "\u2028\u2029"
        else character
        for character in value
    )


def source_details(source: Source) -> dict[str, str | None]:
    """Optional provenance facts shared by the overview card and library show."""

    external = source.external
    playlist = None if external is None else external.playlist_title or external.playlist_id
    if external is not None and external.playlist_index is not None:
        playlist = f"{playlist or ''} #{external.playlist_index}".strip()
    return {
        "channel": source.channel,
        "published": None if source.published is None else source.published.date().isoformat(),
        "url": None if external is None else external.url,
        "playlist": playlist,
    }


def format_time(seconds: float | None, *, duration: bool = False) -> str:
    if seconds is None:
        return "unknown"
    sign = "-" if seconds < 0 else ""
    hours, remainder = divmod(int(abs(seconds)), 3600)
    minutes, seconds_part = divmod(remainder, 60)
    if hours or duration:
        return f"{sign}{hours}:{minutes:02d}:{seconds_part:02d}"
    return f"{sign}{minutes:02d}:{seconds_part:02d}"
