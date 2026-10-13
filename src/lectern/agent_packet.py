"""Experimental talk packets and checks of short citations against current evidence."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from lectern.bundle import StageName, StageState
from lectern.evidence import read_frames, read_transcript_metadata
from lectern.keyframes import FRAME_EVIDENCE_LIMIT, FRAME_METHOD
from lectern.readable import format_time, plain_field
from lectern.records import AutomationError, LibraryStatus, redact_paths
from lectern.search import AnchorResolution, make_anchor, resolve_against_segments
from lectern.state import AutomationStateStore

CITATION_INSTRUCTION = (
    "Cite moments as `[t=MM:SS <12hex>#<seg>]`; Lectern verifies these with "
    "`lectern check-citations`."
)
_SHORT_ID = re.compile(r"[0-9a-f]{12}")
_CITATION = re.compile(
    r"\[t=(?P<time>(?:[0-9]+:)?[0-5][0-9]:[0-5][0-9]) "
    r"(?P<bundle>[^\s\[\]#]+)#(?P<frame>f)?(?P<segment>[0-9]+)\]"
)
_CITATION_LIKE = re.compile(r"\[t=[^\[\]\r\n]*#[^\[\]\r\n]*\]")
_HEADING = re.compile(r"^\s{0,3}#{1,6}(?:\s|$)")
_LIST_MARKER = re.compile(r"^\s*(?:[-+*]|[0-9]+[.)])\s*$")
_SETEXT = re.compile(r"^\s*(?:=+|-+)\s*$")


def talk_packet(state: AutomationStateStore, bundle_id: str) -> dict[str, Any]:
    """Export only registered ready evidence; keep hashes over the original text."""

    evidence = state.get_library_evidence(bundle_id, require_ready=True)
    short_id = bundle_id.rsplit("-", 1)[-1]
    bundle = state.get_library_bundle(bundle_id)
    manifest = state.get_library_manifest(bundle_id)
    metadata = read_transcript_metadata(Path(bundle.bundle_path), manifest, evidence.source)
    visual = read_frames(Path(bundle.bundle_path), manifest)
    source = evidence.source.source
    source_facts = source.model_dump(
        mode="json", include={"title", "channel", "published", "duration_s", "external"}
    )
    for key, value in source_facts.items():
        if isinstance(value, str):
            source_facts[key] = redact_paths(value)
    if source.external is not None:
        source_facts["external"] = {
            key: redact_paths(value) if isinstance(value, str) else value
            for key, value in source.external.model_dump(mode="json").items()
        }
    layers: dict[str, Any] = {
        "transcript": {
            "present": manifest.stages[StageName.TRANSCRIBE].state is StageState.DONE,
            "method": redact_paths(metadata.method),
            "language": None if metadata.language is None else redact_paths(metadata.language),
            "evidence_limit": redact_paths(metadata.evidence_limit),
            "segment_count": len(evidence.segments.root),
        }
    }
    for name, stage in (("visual", StageName.VISUAL), ("speakers", StageName.DIARIZE)):
        record = manifest.stages[stage]
        present = record.state is StageState.DONE
        layers[name] = {"present": present}
        if not present:
            layers[name]["reason"] = (
                redact_paths(record.warnings[0])
                if name == "visual" and record.warnings
                else "not processed"
                if record.state is StageState.PENDING
                else record.state.value
            )
    if visual is not None:
        layers["visual"] = {
            "present": True,
            "frame_count": len(visual.root),
            "method": FRAME_METHOD,
            "evidence_limit": FRAME_EVIDENCE_LIMIT,
        }
    # No translation stage or translation artifact is defined by the current schema.
    # ENRICH is not evidence of translation and must not be used as a proxy for it.
    layers["translation"] = {"present": False, "reason": "not processed"}
    segments: list[dict[str, Any]] = []
    for segment in evidence.segments.root:
        anchor = make_anchor(bundle_id, segment.id, segment.start_s, segment.text)
        segments.append(
            {
                **segment.model_dump(mode="json"),
                "text": redact_paths(segment.text),
                "anchor": {
                    **anchor.to_dict(),
                    "bundle_id": redact_paths(bundle_id),
                    "short_id": short_id,
                    "selectors": [
                        {
                            "type": "timecode_range",
                            "start_s": segment.start_s,
                            "end_s": segment.end_s,
                        },
                        {"type": "quote", "exact": redact_paths(segment.text)},
                    ],
                    "cite": redact_paths(f"{anchor.rendered()[:-1]} {short_id}#{segment.id}]"),
                },
            }
        )
    return {
        "packet": "lectern.talk-packet",
        "packet_version": "0.1",
        "stability": "experimental",
        "bundle_id": redact_paths(bundle_id),
        "source": source_facts,
        "layers": layers,
        "segments": segments,
        "frames": []
        if visual is None
        else [
            {
                "index": frame.index,
                "t_s": frame.t_s,
                "file": frame.file,
                "reason": frame.reason,
                "cite": f"[t={format_time(frame.t_s)} {short_id}#f{frame.index}]",
            }
            for frame in visual.root
        ],
    }


def packet_markdown(packet: dict[str, Any]) -> str:
    """Render source facts, honest layer limits, and one citeable line per segment."""

    source = packet["source"]
    lines = [
        CITATION_INSTRUCTION,
        "",
        f"# {source['title'] or 'Untitled talk'}",
        "",
        "Packet: lectern.talk-packet 0.1 (experimental; shape may change)",
        f"Bundle: {packet['bundle_id']}",
        f"Channel: {source['channel'] or 'unknown'}",
        f"Published: {source['published'] or 'unknown'}",
        f"Duration: {format_time(source['duration_s'], duration=True)}",
    ]
    if source["external"] is not None:
        for key, value in source["external"].items():
            if value is not None:
                lines.append(f"External {key}: {value}")
    lines.extend(["", "## Layers", ""])
    for name, layer in packet["layers"].items():
        description = (
            "present" if layer["present"] else f"absent ({layer.get('reason', 'not processed')})"
        )
        if name == "transcript":
            description += (
                f"; method: {layer['method']}; language: {layer['language'] or 'unknown'}"
                f"; segments: {layer['segment_count']}; evidence limit: {layer['evidence_limit']}"
            )
        elif name == "visual" and layer["present"]:
            description += (
                f"; frames: {layer['frame_count']}; method: {layer['method']}"
                f"; evidence limit: {layer['evidence_limit']}"
            )
        lines.append(f"- {name}: {description}")
    lines.extend(["", "## Segments", ""])
    for segment in packet["segments"]:
        lines.append(f"{segment['anchor']['cite']} {segment['text']}")
    if packet.get("frames"):
        lines.extend(
            [
                "",
                "## Frames",
                "",
                "Paths below are relative to visual/ in the bundle. "
                "Frame citations use #f<index>: `[t=MM:SS <12hex>#f<index>]`.",
                "",
            ]
        )
        for frame in packet["frames"]:
            path = frame["file"].removeprefix("visual/")
            lines.append(
                f"[t={format_time(frame['t_s'])}] {path} ({frame['reason']}); "
                f"index {frame['index']}; cite {frame['cite']}"
            )
    # Evidence may contain controls/newlines; keep each field on its own row.
    return "\n".join(" ".join(plain_field(str(line)).split()) for line in lines)


def check_citations(state: AutomationStateStore, text: str) -> dict[str, Any]:
    """Short tokens name current segments/frames, not historical snapshots."""

    cache: dict[tuple[str, bool], tuple[str | None, list[dict[str, Any]]]] = {}
    suffixes: dict[str, list[str]] | None = None
    citations: list[dict[str, Any]] = []
    for token in _CITATION_LIKE.finditer(text):
        match = _CITATION.fullmatch(token[0])
        if match is None:
            citations.append(
                {
                    "token": redact_paths(token[0]),
                    "bundle_id": None,
                    "segment_id": None,
                    "time": None,
                    "expected_time": None,
                    "status": "malformed",
                }
            )
            continue
        bundle_id = match["bundle"]
        is_frame = match["frame"] is not None
        identifier_field = "frame_index" if is_frame else "segment_id"
        if _SHORT_ID.fullmatch(bundle_id):
            if suffixes is None:
                suffixes = {}
                for registered_id in state.list_registered_bundle_ids():
                    suffix = registered_id.rsplit("-", 1)[-1]
                    suffixes.setdefault(suffix, []).append(registered_id)
            candidates = suffixes.get(bundle_id, [])
            if len(candidates) == 1:
                bundle_id = candidates[0]
            else:
                citations.append(
                    {
                        "token": redact_paths(match[0]),
                        "bundle_id": bundle_id,
                        identifier_field: int(match["segment"]),
                        "time": match["time"],
                        "expected_time": None,
                        "status": "ambiguous_bundle" if candidates else "unknown_bundle",
                    }
                )
                continue
        cache_key = bundle_id, is_frame
        if cache_key not in cache:
            try:
                bundle = state.get_library_bundle(bundle_id)
            except AutomationError:
                cache[cache_key] = ("unknown_bundle", [])
            else:
                try:
                    if bundle.status is not LibraryStatus.READY:
                        cache[cache_key] = ("not_ready", [])
                    else:
                        if is_frame:
                            frames = read_frames(
                                Path(bundle.bundle_path), state.get_library_manifest(bundle_id)
                            )
                            cache[cache_key] = (
                                None,
                                []
                                if frames is None
                                else [{"id": f.index, "start_s": f.t_s} for f in frames.root],
                            )
                        else:
                            evidence = state.get_registered_evidence(bundle_id)
                            cache[cache_key] = (
                                None,
                                [s.model_dump(mode="json") for s in evidence.segments.root],
                            )
                except (AutomationError, OSError, ValueError):
                    cache[cache_key] = ("not_ready", [])
        status, segments = cache[cache_key]
        segment_id = int(match["segment"])
        segment = next((s for s in segments if s["id"] == segment_id), None)
        expected_time = None
        if status is None:
            if segment is None:
                status = "unknown_frame" if is_frame else "unknown_segment"
            elif is_frame:
                expected_time = format_time(segment["start_s"])
                displayed_s = 0
                for part in match["time"].split(":"):
                    displayed_s = displayed_s * 60 + int(part)
                status = "ok" if displayed_s == int(segment["start_s"]) else "time_mismatch"
            else:
                anchor = make_anchor(bundle_id, segment_id, segment["start_s"], segment["text"])
                resolved = resolve_against_segments(anchor, segments)
                expected_time = format_time(resolved.start_s)
                displayed_s = 0
                for part in match["time"].split(":"):
                    displayed_s = displayed_s * 60 + int(part)
                if resolved.outcome is not AnchorResolution.EXACT or resolved.start_s is None:
                    status = "unknown_segment"
                else:
                    status = "ok" if displayed_s == int(resolved.start_s) else "time_mismatch"
        citations.append(
            {
                "token": redact_paths(match[0]),
                "bundle_id": redact_paths(bundle_id),
                identifier_field: segment_id,
                "time": match["time"],
                "expected_time": expected_time,
                "status": status,
            }
        )
    uncited = 0
    for paragraph in re.split(r"\n\s*\n", text):
        lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
        if not lines or _CITATION_LIKE.search(paragraph):
            continue
        if len(lines) == 2 and _SETEXT.fullmatch(lines[1]):
            continue
        if all(_HEADING.match(line) or _LIST_MARKER.fullmatch(line) for line in lines):
            continue
        uncited += 1
    ok = sum(item["status"] == "ok" for item in citations)
    return {
        "citations": citations,
        "ok": ok,
        "problems": len(citations) - ok,
        "uncited_paragraphs": uncited,
    }
