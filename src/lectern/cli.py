from __future__ import annotations

import json
import os
import shutil
import sqlite3
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from lectern import __version__
from lectern.agent_packet import check_citations, packet_markdown, talk_packet
from lectern.automation import (
    DEFAULT_STATE_PATH,
    DEFAULT_YOUTUBE_API_KEY_ENV,
    AutomationError,
    QueueState,
    SourceKind,
    SourcePolicy,
    YouTubePlaylistAdapter,
    open_state,
    preflight_local_folder,
    preflight_state_store,
    preflight_youtube_playlist,
)
from lectern.bundle import StageName, StageState, export_artifact_schemas, export_json_schema
from lectern.evidence import SelectedEvidence, read_frames
from lectern.ingest import IngestError
from lectern.keyframes import (
    DEFAULT_FRAME_INTERVAL,
    DEFAULT_MAX_FRAMES,
    KeyframeError,
    validate_frame_options,
)
from lectern.migrations import MigrationError, migrate_bundle
from lectern.readable import format_time, source_details
from lectern.readable import plain_field as _plain_search_field
from lectern.records import redact_paths

# Commands that return stored data to a caller, and therefore must emit no
# filesystem path in either rendering.
#
# This tuple is the CLI's own declaration of its outward surface, and it exists
# to be compared against the path-projection tests. Without it, that coverage is
# an allowlist of the commands someone thought of, and a command added later
# inherits no assertion at all — which is how four commands came to be returning
# absolute paths under a design that claimed none did.
#
# Commands that only accept input (`sources add-folder`), report on a path the
# caller just supplied (`doctor`, `sources preflight`), or write a file the
# caller named (`schema export`, `ingest`) are deliberately absent: they
# disclose nothing the caller did not already type.
OUTWARD_COMMANDS = (
    "sources list",
    "sources scan",
    "queue list",
    "queue show",
    "library list",
    "library show",
    "library search",
    "library cite",
    "transcript",
    "export",
    "check-citations",
    "migrate",
)

USAGE = """usage: lectern [--version] <command>

commands:
  check-citations check a review's citations against current ready evidence
  doctor         check required local tools and state-store access
  export         export an experimental talk packet for agents
  ingest         ingest local media into a bundle
  library        list, search, or cite ingested bundles from local state
  migrate        migrate one bundle to the current manifest schema
  queue          inspect and update discovery queue items
  schema export  print or write the manifest JSON Schema
  sources        manage local source registry and scans
  transcript     read a ready bundle's timestamped transcript
"""


def _doctor() -> int:
    ok = True
    print(f"lectern: OK ({__version__})")
    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"python: OK ({version})")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print("ffmpeg: MISSING")
        ok = False
    else:
        print(f"ffmpeg: OK ({ffmpeg})")
    state = preflight_state_store(DEFAULT_STATE_PATH)
    if state.ok:
        print(f"state: OK ({state.path})")
    else:
        detail = state.error or "state path is not writable"
        print(f"state: ERROR ({detail})")
        ok = False
    youtube_status = (
        f"configured via {DEFAULT_YOUTUBE_API_KEY_ENV}"
        if os.environ.get(DEFAULT_YOUTUBE_API_KEY_ENV)
        else f"not configured; set {DEFAULT_YOUTUBE_API_KEY_ENV} for YouTube discovery"
    )
    print(f"youtube: OPTIONAL ({youtube_status})")
    return 0 if ok else 1


def _export_schema(args: Sequence[str]) -> int:
    if len(args) == 2 and args[0] == "--all-into":
        # Every artifact type, not only the manifest. Written as a directory
        # because "schemas for every written artifact type" is a set, and
        # exporting them one at a time invites the set to fall out of date.
        directory = Path(args[1])
        directory.mkdir(parents=True, exist_ok=True)
        for name, schema_text in export_artifact_schemas().items():
            (directory / f"{name}.schema.json").write_text(schema_text, encoding="utf-8")
        print(f"wrote {len(export_artifact_schemas())} schemas to {directory}")
        return 0
    schema = export_json_schema()
    if not args:
        print(schema, end="")
        return 0
    if len(args) == 2 and args[0] == "--output":
        output = Path(args[1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(schema, encoding="utf-8")
        print(f"wrote {output}")
        return 0
    print(
        "usage: lectern schema export [--output PATH | --all-into DIR]",
        file=sys.stderr,
    )
    return 2


def _ingest(args: Sequence[str]) -> int:
    if not args or args[0].startswith("-"):
        print(
            "usage: lectern ingest SOURCE [--output DIR] [--state PATH] "
            "[--transcriber-command COMMAND] [--asr-model NAME] "
            "[--allow-model-download] [--language CODE] [--frames] "
            "[--frame-interval SECONDS] [--max-frames N] [--json]",
            file=sys.stderr,
        )
        return 2

    source = Path(args[0])
    output_root = Path("bundles")
    state_path = DEFAULT_STATE_PATH
    json_output = False
    transcriber_command: str | None = None
    asr_model = "small"
    allow_model_download = False
    language: str | None = None
    frames = False
    frame_interval = DEFAULT_FRAME_INTERVAL
    max_frames = DEFAULT_MAX_FRAMES
    rest = list(args[1:])
    index = 0
    while index < len(rest):
        token = rest[index]
        if token == "--output" and index + 1 < len(rest):
            output_root = Path(rest[index + 1])
            index += 2
        elif token == "--state" and index + 1 < len(rest):
            state_path = Path(rest[index + 1])
            index += 2
        elif token == "--transcriber-command" and index + 1 < len(rest):
            transcriber_command = rest[index + 1]
            index += 2
        elif token == "--asr-model" and index + 1 < len(rest):
            asr_model = rest[index + 1]
            index += 2
        elif token == "--frames":
            frames = True
            index += 1
        elif token in ("--frame-interval", "--max-frames") and index + 1 < len(rest):
            try:
                if token == "--frame-interval":
                    frame_interval = float(rest[index + 1])
                else:
                    max_frames = int(rest[index + 1])
                validate_frame_options(frame_interval, max_frames)
            except (ValueError, KeyframeError) as exc:
                print(f"ingest: invalid {token}: {exc}", file=sys.stderr)
                return 2
            index += 2
        elif token == "--allow-model-download":
            allow_model_download = True
            index += 1
        elif token == "--language" and index + 1 < len(rest):
            language = rest[index + 1]
            index += 2
        elif token == "--json":
            json_output = True
            index += 1
        else:
            print(
                "usage: lectern ingest SOURCE [--output DIR] [--state PATH] "
                "[--transcriber-command COMMAND] [--asr-model NAME] "
                "[--allow-model-download] [--language CODE] [--frames] "
                "[--frame-interval SECONDS] [--max-frames N] [--json]",
                file=sys.stderr,
            )
            return 2

    try:
        with open_state(state_path) as state:
            result = state.ingest_one_shot(
                source,
                output_root,
                transcriber_command=transcriber_command,
                asr_model=asr_model,
                allow_model_download=allow_model_download,
                language=language,
                frames=frames,
                frame_interval=frame_interval,
                max_frames=max_frames,
            )
    except IngestError as exc:
        print(f"ingest: {exc}", file=sys.stderr)
        return 1
    except AutomationError as exc:
        print(f"ingest: {exc}", file=sys.stderr)
        return 3
    except OSError as exc:
        print(f"ingest: {exc}", file=sys.stderr)
        return 1

    if json_output:
        _print_json(
            {
                "bundle_id": result.manifest.bundle_id,
                "bundle_dir": str(result.bundle_dir),
                "state_path": str(state_path),
            }
        )
    else:
        print(result.bundle_dir)
    return 0


def _sources(args: Sequence[str]) -> int:
    if not args:
        _sources_usage()
        return 2
    command = args[0]
    rest, state_path, json_output = _parse_common(args[1:])
    try:
        if command == "add-folder":
            return _sources_add_folder(rest, state_path, json_output)
        if command == "add-youtube-playlist":
            return _sources_add_youtube_playlist(rest, state_path, json_output)
        if command == "list":
            return _sources_list(rest, state_path, json_output)
        if command == "scan":
            return _sources_scan(rest, state_path, json_output)
        if command == "preflight":
            return _sources_preflight(rest, json_output)
        if command == "preflight-youtube":
            return _sources_preflight_youtube(rest, json_output)
    except AutomationError as exc:
        print(f"sources: {redact_paths(str(exc))}", file=sys.stderr)
        return 3
    except OSError as exc:
        print(f"sources: {redact_paths(str(exc))}", file=sys.stderr)
        return 1
    _sources_usage()
    return 2


def _sources_add_folder(args: Sequence[str], state_path: Path, json_output: bool) -> int:
    if len(args) not in (2, 4):
        _sources_usage()
        return 2
    name = args[0]
    root = Path(args[1])
    policy = SourcePolicy.REVIEW
    if len(args) == 4:
        if args[2] != "--policy":
            _sources_usage()
            return 2
        try:
            policy = SourcePolicy(args[3])
        except ValueError:
            print("sources: policy must be disabled, scan-only, or review", file=sys.stderr)
            return 2
    with open_state(state_path) as state:
        source = state.add_local_folder_source(name, root, policy)
    if json_output:
        _print_json(source.to_dict())
    else:
        print(f"{source.id}\t{source.policy.value}\t{source.name}")
    return 0


def _sources_add_youtube_playlist(args: Sequence[str], state_path: Path, json_output: bool) -> int:
    if len(args) not in (2, 4):
        _sources_usage()
        return 2
    name = args[0]
    playlist = args[1]
    policy = SourcePolicy.REVIEW
    if len(args) == 4:
        if args[2] != "--policy":
            _sources_usage()
            return 2
        try:
            policy = SourcePolicy(args[3])
        except ValueError:
            print("sources: policy must be disabled, scan-only, or review", file=sys.stderr)
            return 2
    with open_state(state_path) as state:
        source = state.add_youtube_playlist_source(name, playlist, policy)
    if json_output:
        _print_json(source.to_dict())
    else:
        print(f"{source.id}\t{source.policy.value}\t{source.name}")
    return 0


def _sources_list(args: Sequence[str], state_path: Path, json_output: bool) -> int:
    if args:
        _sources_usage()
        return 2
    with open_state(state_path) as state:
        sources = state.list_sources()
    if json_output:
        _print_json({"sources": [source.to_dict() for source in sources]})
    else:
        for source in sources:
            print(f"{source.id}\t{source.name}\t{source.policy.value}")
    return 0


def _sources_scan(args: Sequence[str], state_path: Path, json_output: bool) -> int:
    if not args:
        _sources_usage()
        return 2
    source_name = args[0]
    api_key_env: str | None = None
    max_pages: int | None = None
    rest = list(args[1:])
    index = 0
    while index < len(rest):
        token = rest[index]
        if token == "--api-key-env" and index + 1 < len(rest):
            api_key_env = rest[index + 1]
            index += 2
        elif token == "--max-pages" and index + 1 < len(rest):
            max_pages = _positive_int(rest[index + 1])
            if max_pages is None:
                print("sources: --max-pages must be a positive integer", file=sys.stderr)
                return 2
            index += 2
        else:
            _sources_usage()
            return 2

    with open_state(state_path) as state:
        source = state.get_source(source_name)
        adapter: YouTubePlaylistAdapter | None = None
        if source.kind is SourceKind.YOUTUBE_PLAYLIST:
            # Disabled sources scan as a state-layer no-op; requiring an API key
            # to construct an adapter that will never be called would break that.
            if source.policy is not SourcePolicy.DISABLED:
                adapter = YouTubePlaylistAdapter.from_environment(
                    api_key_env=api_key_env or DEFAULT_YOUTUBE_API_KEY_ENV,
                    max_pages=max_pages,
                )
        else:
            rejected = next(
                (
                    flag
                    for flag, value in (
                        ("--api-key-env", api_key_env),
                        ("--max-pages", max_pages),
                    )
                    if value is not None
                ),
                None,
            )
            if rejected is not None:
                print(
                    f"sources: {rejected} is only supported for "
                    f"{SourceKind.YOUTUBE_PLAYLIST.value} sources",
                    file=sys.stderr,
                )
                return 2
        delta = state.scan_source(source_name, adapter=adapter)
    payload = delta.to_dict()
    if json_output:
        _print_json(payload)
    else:
        counts = payload["counts"]
        print(
            f"added={counts['added']} changed={counts['changed']} "
            f"removed={counts['removed']} unchanged={counts['unchanged']} queued={counts['queued']}"
        )
    return 0


def _sources_preflight(args: Sequence[str], json_output: bool) -> int:
    if len(args) != 1:
        _sources_usage()
        return 2
    preflight = preflight_local_folder(Path(args[0]))
    if json_output:
        _print_json(preflight.to_dict())
    else:
        status = "OK" if preflight.ok else "FAIL"
        print(f"{status}\tmedia_files={preflight.media_files}\t{preflight.path}")
    return 0 if preflight.ok else 1


def _sources_preflight_youtube(args: Sequence[str], json_output: bool) -> int:
    if len(args) not in (1, 3):
        _sources_usage()
        return 2
    api_key_env = DEFAULT_YOUTUBE_API_KEY_ENV
    if len(args) == 3:
        if args[1] != "--api-key-env":
            _sources_usage()
            return 2
        api_key_env = args[2]
    preflight = preflight_youtube_playlist(args[0], api_key_env=api_key_env)
    if json_output:
        _print_json(preflight.to_dict())
    else:
        status = "OK" if preflight.ok else "FAIL"
        detail = preflight.error or f"estimated_units_consumed={preflight.estimated_units_consumed}"
        print(f"{status}\tplaylist={preflight.playlist_id}\t{detail}")
    return 0 if preflight.ok else 1


def _queue(args: Sequence[str]) -> int:
    if not args:
        _queue_usage()
        return 2
    command = args[0]
    rest, state_path, json_output = _parse_common(args[1:])
    try:
        if command == "list":
            return _queue_list(rest, state_path, json_output)
        if command == "show":
            return _queue_show(rest, state_path, json_output)
        if command == "approve":
            return _queue_transition(rest, state_path, json_output, command)
        if command == "skip":
            return _queue_transition(rest, state_path, json_output, command)
        if command == "retry":
            return _queue_transition(rest, state_path, json_output, command)
        if command == "ingest":
            return _queue_ingest(rest, state_path, json_output)
    except AutomationError as exc:
        print(f"queue: {exc}", file=sys.stderr)
        return 3
    except IngestError as exc:
        print(f"queue: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"queue: {exc}", file=sys.stderr)
        return 1
    _queue_usage()
    return 2


def _queue_list(args: Sequence[str], state_path: Path, json_output: bool) -> int:
    if len(args) not in (0, 2):
        _queue_usage()
        return 2
    state_filter: QueueState | None = None
    if args:
        if args[0] != "--queue-state":
            _queue_usage()
            return 2
        try:
            state_filter = QueueState(args[1])
        except ValueError:
            print("queue: unknown state", file=sys.stderr)
            return 2
    with open_state(state_path) as state:
        items = state.list_queue(state_filter)
    if json_output:
        _print_json({"queue": [item.to_dict() for item in items]})
    else:
        for item in items:
            print(f"{item.id}\t{item.state.value}\t{item.source_item_id}")
    return 0


def _queue_show(args: Sequence[str], state_path: Path, json_output: bool) -> int:
    if len(args) != 1:
        _queue_usage()
        return 2
    with open_state(state_path) as state:
        item = state.get_queue_item(args[0])
    if json_output:
        _print_json(item.to_dict())
    else:
        print(f"{item.id}\t{item.state.value}\t{item.source_item_id}")
    return 0


def _queue_transition(
    args: Sequence[str],
    state_path: Path,
    json_output: bool,
    command: str,
) -> int:
    if len(args) != 1:
        _queue_usage()
        return 2
    with open_state(state_path) as state:
        if command == "approve":
            item = state.approve_queue_item(args[0])
        elif command == "skip":
            item = state.skip_queue_item(args[0])
        else:
            item = state.retry_queue_item(args[0])
    if json_output:
        _print_json(item.to_dict())
    else:
        print(f"{item.id}\t{item.state.value}")
    return 0


def _queue_ingest(args: Sequence[str], state_path: Path, json_output: bool) -> int:
    if not args:
        _queue_usage()
        return 2
    queue_item_id = args[0]
    output_root = Path("bundles")
    transcriber_command: str | None = None
    asr_model = "small"
    allow_model_download = False
    language: str | None = None
    frames = False
    frame_interval = DEFAULT_FRAME_INTERVAL
    max_frames = DEFAULT_MAX_FRAMES
    rest = list(args[1:])
    index = 0
    while index < len(rest):
        token = rest[index]
        if token == "--output" and index + 1 < len(rest):
            output_root = Path(rest[index + 1])
            index += 2
        elif token == "--transcriber-command" and index + 1 < len(rest):
            transcriber_command = rest[index + 1]
            index += 2
        elif token == "--asr-model" and index + 1 < len(rest):
            asr_model = rest[index + 1]
            index += 2
        elif token == "--frames":
            frames = True
            index += 1
        elif token in ("--frame-interval", "--max-frames") and index + 1 < len(rest):
            try:
                if token == "--frame-interval":
                    frame_interval = float(rest[index + 1])
                else:
                    max_frames = int(rest[index + 1])
                validate_frame_options(frame_interval, max_frames)
            except (ValueError, KeyframeError) as exc:
                print(f"ingest: invalid {token}: {exc}", file=sys.stderr)
                return 2
            index += 2
        elif token == "--allow-model-download":
            allow_model_download = True
            index += 1
        elif token == "--language" and index + 1 < len(rest):
            language = rest[index + 1]
            index += 2
        else:
            _queue_usage()
            return 2
    with open_state(state_path) as state:
        result = state.ingest_queue_item(
            queue_item_id,
            output_root,
            transcriber_command=transcriber_command,
            asr_model=asr_model,
            allow_model_download=allow_model_download,
            language=language,
            frames=frames,
            frame_interval=frame_interval,
            max_frames=max_frames,
        )
    if json_output:
        _print_json(
            {
                "bundle_id": result.manifest.bundle_id,
                "bundle_dir": str(result.bundle_dir),
                "queue_item_id": queue_item_id,
            }
        )
    else:
        print(result.bundle_dir)
    return 0


def _library(args: Sequence[str]) -> int:
    if not args:
        _library_usage()
        return 2
    command = args[0]
    rest, state_path, json_output = _parse_common(args[1:])
    try:
        if command == "list":
            return _library_list(rest, state_path, json_output)
        if command == "show":
            return _library_show(rest, state_path, json_output)
        if command == "search":
            return _library_search(rest, state_path, json_output)
        if command == "cite":
            return _library_cite(rest, state_path, json_output)
    except AutomationError as exc:
        print(f"library: {redact_paths(str(exc))}", file=sys.stderr)
        return 3
    except OSError as exc:
        print(f"library: {redact_paths(str(exc))}", file=sys.stderr)
        return 1
    _library_usage()
    return 2


def _project_metadata_values(values: dict[str, Any]) -> dict[str, Any]:
    """Redact display metadata without changing stored provenance or identities."""

    return {
        key: (
            redact_paths(value)
            if isinstance(value, str)
            else _project_metadata_values(cast(dict[str, Any], value))
            if isinstance(value, dict)
            else value
        )
        for key, value in values.items()
    }


def _library_list(args: Sequence[str], state_path: Path, json_output: bool) -> int:
    grouped = bool(args) and args[0] == "--playlist"
    if args and (not grouped or len(args) > 2 or (len(args) == 2 and args[1].startswith("-"))):
        _library_usage()
        return 2
    playlist_id = args[1] if len(args) == 2 else None
    with open_state(state_path) as state:
        bundles = state.list_library()
        rows: list[dict[str, Any]] = []
        for bundle in bundles:
            title, external = "unknown", None
            try:
                source = state.get_library_manifest(bundle.bundle_id).source
                title = source.title or "Untitled talk"
                external = (
                    None if source.external is None else source.external.model_dump(mode="json")
                )
            except (AutomationError, OSError):
                pass
            rows.append({**bundle.to_dict(), "title": title, "external": external})
    if grouped:
        if playlist_id is not None:
            rows = [
                row
                for row in rows
                if row["external"] is not None and row["external"]["playlist_id"] == playlist_id
            ]
        rows.sort(key=_playlist_sort_key)
    if json_output:
        # Filtering and ordering use the original metadata; projection is display-only.
        for row in rows:
            row["title"] = redact_paths(row["title"])
            if row["external"] is not None:
                row["external"] = _project_metadata_values(row["external"])
        _print_json({"bundles": rows})
    else:
        previous_group: str | None = None
        for row in rows:
            title = _plain_search_field(redact_paths(row["title"]))
            if grouped:
                group_external: dict[str, Any] = row["external"] or {}
                group = group_external.get("playlist_id") or ""
                if group != previous_group:
                    playlist_title = group_external.get("playlist_title") or group
                    header = f"{playlist_title} ({group})" if group else "Without playlist"
                    print(_plain_search_field(redact_paths(header)))
                    previous_group = group
                index = group_external.get("playlist_index") if group else None
                print(f"{index if index is not None else ''}\t{row['bundle_id']}\t{title}")
            else:
                print(f"{row['bundle_id']}\t{row['created_at']}\t{row['status']}\t{title}")
    return 0


def _playlist_sort_key(row: dict[str, Any]) -> tuple[bool, str, bool, int, str]:
    external: dict[str, Any] = row["external"] or {}
    playlist = external.get("playlist_id") or ""
    index = external.get("playlist_index") if playlist else None
    return not bool(playlist), playlist, index is None, index or 0, row["bundle_id"]


def _library_show(args: Sequence[str], state_path: Path, json_output: bool) -> int:
    if len(args) != 1:
        _library_usage()
        return 2
    with open_state(state_path) as state:
        bundle = state.get_library_bundle(args[0])
        manifest = state.get_library_manifest(bundle.bundle_id)
        try:
            evidence = state.get_registered_evidence(bundle.bundle_id)
        except (AutomationError, OSError):
            # Show remains an inspection command for non-ready bundles. Never
            # derive transcript facts from evidence that failed strict checks.
            evidence = None
        source_kind = state.get_source(bundle.source_id).kind.value
    try:
        frames_document = read_frames(Path(bundle.bundle_path), manifest)
    except (ValueError, OSError):
        frames_document = None
    frame_count = None if frames_document is None else len(frames_document.root)
    segments = None if evidence is None else evidence.segments.root
    layers = {
        "transcript": manifest.stages[StageName.TRANSCRIBE].state is StageState.DONE,
        "visual": manifest.stages[StageName.VISUAL].state is StageState.DONE,
    }
    title = redact_paths(manifest.source.title or "Untitled talk")
    payload = {
        "bundle": bundle.to_dict(),
        "manifest": {
            **manifest.model_dump(mode="json"),
            "source": _project_metadata_values(manifest.source.model_dump(mode="json")),
        },
        "bundle_id": bundle.bundle_id,
        "title": title,
        **_project_metadata_values(source_details(manifest.source)),
        "external": (
            None
            if manifest.source.external is None
            else _project_metadata_values(manifest.source.external.model_dump(mode="json"))
        ),
        "status": bundle.status.value,
        "created_at": bundle.created_at,
        "source_kind": source_kind,
        "duration_s": manifest.source.duration_s,
        "transcript_method": None if evidence is None else evidence.source.transcript.method,
        "segment_count": None if segments is None else len(segments),
        "word_count": None if segments is None else sum(len(s.text.split()) for s in segments),
        "frame_count": frame_count,
        "layers": layers,
        "bundle_dir": redact_paths(bundle.bundle_path),
    }
    if json_output:
        _print_json(payload)
    else:
        details = "".join(
            f"{'URL' if key == 'url' else key.capitalize()}: "
            f"{_plain_search_field(redact_paths(value))}\n"
            for key, value in source_details(manifest.source).items()
            if value is not None
        )
        visual_description = "no"
        if layers["visual"]:
            visual_description = (
                f"yes ({frame_count} frames)" if frame_count is not None else "yes (frames unknown)"
            )
        print(
            f"Bundle: {bundle.bundle_id}\n"
            f"Title: {_plain_search_field(title)}\n"
            f"{details}"
            f"Status: {bundle.status.value}\n"
            f"Created: {bundle.created_at}\n"
            f"Source kind: {source_kind}\n"
            f"Duration: {format_time(manifest.source.duration_s, duration=True)}\n"
            f"Transcript method: {payload['transcript_method'] or 'unknown'}\n"
            f"Segments: {payload['segment_count'] if segments is not None else 'unknown'}\n"
            f"Words: {payload['word_count'] if segments is not None else 'unknown'}\n"
            f"Transcript: {'yes' if layers['transcript'] else 'no'}\n"
            f"Visual: {visual_description}\n"
            f"Bundle directory: {payload['bundle_dir']}"
        )
    return 0


def _transcript(args: Sequence[str]) -> int:
    rest, state_path, json_flag = _parse_common(args)
    output_format = "md"
    if len(rest) == 3 and rest[1] == "--format":
        output_format = rest[2]
        rest = rest[:1]
    if len(rest) != 1 or json_flag or output_format not in ("md", "txt", "json"):
        print(
            "usage: lectern transcript BUNDLE_ID [--format md|txt|json] [--state PATH]",
            file=sys.stderr,
        )
        return 2
    bundle_id = rest[0]
    try:
        with open_state(state_path) as state:
            evidence = state.get_library_evidence(bundle_id, require_ready=True)
    except AutomationError as exc:
        print(f"transcript: {redact_paths(str(exc))}", file=sys.stderr)
        return 3
    except OSError as exc:
        print(f"transcript: {redact_paths(str(exc))}", file=sys.stderr)
        return 1
    title = redact_paths(evidence.source.source.title or "Untitled talk")
    segments = evidence.segments.root
    if output_format == "json":
        _print_json(
            {
                "bundle_id": bundle_id,
                "title": title,
                "segments": [
                    {
                        "id": s.id,
                        "start_s": s.start_s,
                        "end_s": s.end_s,
                        "text": redact_paths(s.text),
                    }
                    for s in segments
                ],
            }
        )
    else:
        if output_format == "md":
            print(f"# {_plain_search_field(title)}\n")
        print(
            "\n\n".join(
                f"[{format_time(s.start_s)}] (#{s.id}) {_plain_search_field(redact_paths(s.text))}"
                for s in segments
            )
        )
    return 0


def _export(args: Sequence[str]) -> int:
    rest, state_path, json_flag = _parse_common(args)
    output_format = "json"
    if len(rest) == 3 and rest[1] == "--format":
        output_format = rest[2]
        rest = rest[:1]
    if (
        len(rest) != 1
        or rest[0].startswith("-")
        or json_flag
        or output_format not in ("json", "md")
    ):
        print(
            "usage: lectern export BUNDLE_ID [--format json|md] [--state PATH]",
            file=sys.stderr,
        )
        return 2
    try:
        with open_state(state_path) as state:
            packet = talk_packet(state, rest[0])
    except (AutomationError, ValueError, RecursionError) as exc:
        print(f"export: {redact_paths(str(exc))}", file=sys.stderr)
        return 3
    except OSError as exc:
        print(f"export: {redact_paths(str(exc))}", file=sys.stderr)
        return 1
    if output_format == "json":
        _print_json(packet)
    else:
        print(packet_markdown(packet))
    return 0


def _check_citations(args: Sequence[str]) -> int:
    rest, state_path, json_output = _parse_common(args)
    if len(rest) != 1 or rest[0].startswith("-"):
        print("usage: lectern check-citations FILE [--state PATH] [--json]", file=sys.stderr)
        return 2
    try:
        text = Path(rest[0]).read_text(encoding="utf-8")
        with open_state(state_path) as state:
            report = check_citations(state, text)
    except (AutomationError, ValueError) as exc:
        print(f"check-citations: {redact_paths(str(exc))}", file=sys.stderr)
        return 3
    except OSError as exc:
        print(f"check-citations: {redact_paths(str(exc))}", file=sys.stderr)
        return 1
    if json_output:
        _print_json(report)
    else:
        for citation in report["citations"]:
            if citation["status"] != "ok":
                print(f"{_plain_search_field(citation['token'])}: {citation['status']}")
        print(
            f"citations: {report['ok']} ok, {report['problems']} problems; "
            f"uncited paragraphs: {report['uncited_paragraphs']}"
        )
    return 1 if report["problems"] else 0


def _parse_common(args: Sequence[str]) -> tuple[list[str], Path, bool]:
    state_path = DEFAULT_STATE_PATH
    json_output = False
    rest: list[str] = []
    index = 0
    while index < len(args):
        token = args[index]
        if token == "--":
            rest.extend(args[index:])
            break
        if token == "--state" and index + 1 < len(args):
            state_path = Path(args[index + 1])
            index += 2
        elif token == "--json":
            json_output = True
            index += 1
        else:
            rest.append(token)
            index += 1
    return rest, state_path, json_output


def _positive_int(value: str) -> int | None:
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed >= 1 else None


def _print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _sources_usage() -> None:
    print(
        "usage: lectern sources "
        "{add-folder NAME PATH [--policy POLICY]|"
        "add-youtube-playlist NAME PLAYLIST [--policy POLICY]|list|"
        "scan SOURCE [--api-key-env ENV] [--max-pages N]|preflight PATH|"
        "preflight-youtube PLAYLIST [--api-key-env ENV]} "
        "[--state PATH] [--json]",
        file=sys.stderr,
    )


def _queue_usage() -> None:
    print(
        "usage: lectern queue "
        "{list [--queue-state STATE]|show ITEM|approve ITEM|skip ITEM|retry ITEM|"
        "ingest ITEM [--output DIR] [--transcriber-command COMMAND] "
        "[--asr-model NAME] [--allow-model-download] [--language CODE] "
        "[--frames] [--frame-interval SECONDS] [--max-frames N]} "
        "[--state PATH] [--json]",
        file=sys.stderr,
    )


def _library_search(args: Sequence[str], state_path: Path, json_output: bool) -> int:
    if not args:
        _library_usage()
        return 2
    rest = list(args)
    option_terminated = "--" in rest
    if option_terminated:
        terminator = rest.index("--")
        rest = [*rest[:terminator], *rest[terminator + 1 :]]
    literal = True
    if not option_terminated and rest and rest[-1] == "--operators":
        literal = False
        rest = rest[:-1]
    query = " ".join(rest)
    with open_state(state_path) as state:
        try:
            hits = state.search_segments(query, literal=literal)
        except ValueError as exc:
            print(f"library: {exc}", file=sys.stderr)
            return 2
        evidence_by_bundle: dict[str, SelectedEvidence | None] = {}
        results: list[dict[str, Any]] = []
        for hit in hits:
            if hit.bundle_id not in evidence_by_bundle:
                try:
                    evidence_by_bundle[hit.bundle_id] = state.get_registered_evidence(hit.bundle_id)
                except (AutomationError, OSError):
                    evidence_by_bundle[hit.bundle_id] = None
            evidence = evidence_by_bundle[hit.bundle_id]
            segment = (
                next((s for s in evidence.segments.root if s.id == hit.segment_id), None)
                if evidence is not None
                else None
            )
            results.append(
                {
                    **hit.to_dict(),
                    "start_s": None if segment is None else segment.start_s,
                    "title": (
                        None
                        if evidence is None or evidence.source.source.title is None
                        else redact_paths(evidence.source.source.title)
                    ),
                }
            )
    if json_output:
        _print_json({"results": results})
    else:
        # No results is an answer, not an error. Saying so explicitly keeps
        # "nothing found" distinguishable from "search is broken", which silence
        # would not.
        if not hits:
            print("no matches")
        for result in results:
            bundle_id = _plain_search_field(result["bundle_id"])
            snippet = _plain_search_field(result["snippet"])
            timestamp = format_time(result["start_s"])
            print(f"{bundle_id}\t{result['segment_id']}\t[{timestamp}]\t{snippet}")
    return 0


def _library_cite(args: Sequence[str], state_path: Path, json_output: bool) -> int:
    if len(args) != 2:
        _library_usage()
        return 2
    try:
        segment_id = int(args[1])
    except ValueError:
        print("library: SEGMENT_ID must be an integer", file=sys.stderr)
        return 2
    with open_state(state_path) as state:
        anchor, resolved = state.cite_segment(args[0], segment_id)
    if json_output:
        _print_json(
            {
                "rendered": anchor.rendered(),
                "anchor": anchor.to_dict(),
                "outcome": resolved.outcome.value,
                "resolution": resolved.to_dict(),
            }
        )
    else:
        serialized_anchor = json.dumps(anchor.to_dict(), sort_keys=True, separators=(",", ":"))
        print(f"{anchor.rendered()}\t{serialized_anchor}\t{resolved.outcome.value}")
    return 0


def _library_usage() -> None:
    print(
        "usage: lectern library {list [--playlist [ID]]|show BUNDLE_ID|"
        "search [--] QUERY [--operators]|"
        "cite BUNDLE_ID SEGMENT_ID} [--state PATH] [--json]",
        file=sys.stderr,
    )


def _migrate_bundle(args: Sequence[str]) -> int:
    if len(args) != 1 or args[0].startswith("-"):
        print("usage: lectern migrate BUNDLE", file=sys.stderr)
        return 2
    try:
        result = migrate_bundle(Path(args[0]))
    except MigrationError as exc:
        print(f"migrate: {exc}", file=sys.stderr)
        return 3
    _print_json(result.to_public_dict())
    if result.source_version == "1.0.0" and result.outcome == "already_current":
        print("migrate: already readable; 1.0.0 is a subset of 1.1.0", file=sys.stderr)
    return 0


def _usage() -> None:
    print(USAGE, file=sys.stderr)


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    try:
        if args == ["--version"]:
            print(f"lectern {__version__}")
            return 0
        if args == ["--help"]:
            print(USAGE, end="")
            return 0
        if args == ["doctor"]:
            return _doctor()
        if args and args[0] == "ingest":
            return _ingest(args[1:])
        if args and args[0] == "migrate":
            return _migrate_bundle(args[1:])
        if args and args[0] == "library":
            return _library(args[1:])
        if args and args[0] == "transcript":
            return _transcript(args[1:])
        if args and args[0] == "export":
            return _export(args[1:])
        if args and args[0] == "check-citations":
            return _check_citations(args[1:])
        if args and args[0] == "queue":
            return _queue(args[1:])
        if len(args) >= 2 and args[0] == "schema" and args[1] == "export":
            return _export_schema(args[2:])
        if args and args[0] == "sources":
            return _sources(args[1:])
    except sqlite3.Error as exc:
        print(f"state: {exc}", file=sys.stderr)
        return 3

    _usage()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
