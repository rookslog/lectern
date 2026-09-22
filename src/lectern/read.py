"""Internal read results that bind a hit to selected evidence at one observation."""

from __future__ import annotations

from dataclasses import dataclass

from lectern.records import LibraryStatus, SearchHit


@dataclass(frozen=True)
class ObservedHit:
    hit: SearchHit
    observation: str


@dataclass(frozen=True)
class NonreadyBundle:
    """Current library status; selected transcript evidence may still be readable."""

    bundle_id: str
    status: LibraryStatus


@dataclass(frozen=True)
class ObservedSearchResult:
    hits: list[ObservedHit]
    nonready: list[NonreadyBundle]
