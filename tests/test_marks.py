"""Synthetic acceptance tests for time-bound gestures, not interpreted annotations."""

from __future__ import annotations

import hashlib
import json
import shlex
import sqlite3
import sys
from pathlib import Path
from typing import Any

import pytest

from lectern import cli
from lectern.agent_packet import check_citations, packet_markdown, talk_packet
from lectern.automation import open_state
from lectern.bundle import Manifest
from lectern.ingest import IngestResult
from lectern.records import PATH_REDACTED

FIXTURE = Path(__file__).parent / "fixtures" / "synthetic_talk.wav"


@pytest.fixture
def marked_bundle(tmp_path: Path) -> tuple[Path, IngestResult]:
    media = tmp_path / "synthetic_talk.wav"
    media.write_bytes(FIXTURE.read_bytes())
    script = tmp_path / "transcriber.py"
    segments = [
        {"start_s": 1.25, "end_s": 2.0, "text": "A synthetic opening."},
        {"start_s": 10.0, "end_s": 12.0, "text": "A later synthetic point."},
        {"start_s": 3661.0, "end_s": 3663.0, "text": "An hour-later synthetic point."},
    ]
    script.write_text(f"print({json.dumps({'segments': segments})!r})\n", encoding="utf-8")
    state_path = tmp_path / "state.sqlite"
    with open_state(state_path) as state:
        result = state.ingest_one_shot(
            media,
            tmp_path / "bundles",
            transcriber_command=f"{shlex.quote(sys.executable)} {shlex.quote(str(script))}",
        )
    # Declare a long duration in synthetic metadata without creating long media.
    manifest = Manifest.load(result.bundle_dir)
    manifest.source.duration_s = 3670.0
    source_file = result.bundle_dir / "source.json"
    source = json.loads(source_file.read_text())
    source["source"]["duration_s"] = 3670.0
    payload = json.dumps(source).encode()
    source_file.write_bytes(payload)
    for stage in manifest.stages.values():
        for output in stage.outputs:
            if output.path == "source.json":
                output.sha256 = hashlib.sha256(payload).hexdigest()
                output.bytes = len(payload)
    manifest.save(result.bundle_dir)
    return state_path, result


def run_marks(state: Path, bundle_id: str, command: str, *args: str) -> int:
    return cli.main(["marks", command, bundle_id, *args, "--state", str(state)])


def list_marks(
    state: Path, bundle_id: str, capsys: pytest.CaptureFixture[str]
) -> list[dict[str, Any]]:
    assert run_marks(state, bundle_id, "list", "--json") == 0
    return json.loads(capsys.readouterr().out)["marks"]


def test_import_is_idempotent_across_files_and_preserves_anchors(
    marked_bundle: tuple[Path, IngestResult], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    state_path, result = marked_bundle
    bundle_id = result.manifest.bundle_id
    before = {
        p.relative_to(result.bundle_dir): p.read_bytes()
        for p in result.bundle_dir.rglob("*")
        if p.is_file()
    }
    file = tmp_path / "notes.txt"
    file.write_text("00:01.250 ? why\n00:02 !\n00:08 a thought\n", encoding="utf-8")
    assert run_marks(state_path, bundle_id, "import", str(file)) == 0
    assert capsys.readouterr().out.strip() == "added 3, already present 0, malformed 0"
    assert run_marks(state_path, bundle_id, "import", str(file)) == 0
    assert capsys.readouterr().out.strip() == "added 0, already present 3, malformed 0"
    other = tmp_path / "overlap.txt"
    other.write_text("00:01.250 ? why\n00:10 -> revisit\n00:10 → revisit\n", encoding="utf-8")
    assert run_marks(state_path, bundle_id, "import", str(other), "--json") == 0
    report = json.loads(capsys.readouterr().out)
    assert (report["added"], report["already_present"], report["malformed"]) == (1, 2, 0)
    rows = list_marks(state_path, bundle_id, capsys)
    assert [(r["t_s"], r["binding"], r["segment_id"]) for r in rows] == [
        (1.25, "exact", 0),
        (2.0, "nearest", 0),
        (8.0, "none", None),
        (10.0, "exact", 1),
    ]
    assert [r["kind"] for r in rows] == ["question", "important", None, "follow_up"]
    digest_input = f"{bundle_id}\x001250\x00question\x00why"
    assert rows[0]["mark_id"] == hashlib.sha256(digest_input.encode()).hexdigest()[:16]
    assert rows[0]["import_name"] == "notes.txt" and rows[0]["line_no"] == 1
    assert rows[0]["imported_at"]
    with open_state(state_path) as state:
        anchor, _ = state.cite_segment(bundle_id, 0)
    assert {key: rows[0][key] for key in anchor.to_dict()} == anchor.to_dict()
    assert rows[2]["start_s"] is rows[2]["text_sha256"] is rows[2]["canon_version"] is None
    assert before == {
        p.relative_to(result.bundle_dir): p.read_bytes()
        for p in result.bundle_dir.rglob("*")
        if p.is_file()
    }


def test_parser_reports_each_malformed_line_without_losing_valid_marks(
    marked_bundle: tuple[Path, IngestResult], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    state_path, result = marked_bundle
    bundle_id = result.manifest.bundle_id
    file = tmp_path / "mixed.txt"
    file.write_text(
        "00:01 ? valid\n12:99 bad\n1:01:11 too late\n00:01 @\n00:01\n", encoding="utf-8"
    )
    assert run_marks(state_path, bundle_id, "import", str(file), "--json") == 1
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert (report["added"], report["already_present"], report["malformed"]) == (1, 0, 4)
    assert [e["line_no"] for e in report["errors"]] == [2, 3, 4, 5]
    reasons = [e["reason"].lower() for e in report["errors"]]
    assert "time" in reasons[0] and "duration" in reasons[1]
    assert "kind" in reasons[2] and "text" in reasons[3]
    assert captured.err == ""
    assert run_marks(state_path, bundle_id, "import", str(file)) == 1
    captured = capsys.readouterr()
    assert captured.out.strip() == "added 0, already present 1, malformed 4"
    for line in (2, 3, 4, 5):
        assert f"line {line}:" in captured.err
    assert len(list_marks(state_path, bundle_id, capsys)) == 1


def test_bom_crlf_comments_fractions_hours_and_binding_boundaries(
    marked_bundle: tuple[Path, IngestResult], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    state_path, result = marked_bundle
    file = tmp_path / "encoded.txt"
    file.write_bytes(
        b"\xef\xbb\xbf  # comment\r\n\r\n  00:00 before speech  \r\n"
        b"00:01.249 before start\r\n00:01.25 ?\r\n00:07 !\r\n00:07.001 gap\r\n"
        b"1:01:01.125 -> later\r\n1:01:10 at duration\r\n"
    )
    assert run_marks(state_path, result.manifest.bundle_id, "import", str(file)) == 0
    capsys.readouterr()
    rows = list_marks(state_path, result.manifest.bundle_id, capsys)
    assert [r["t_s"] for r in rows] == [0.0, 1.249, 1.25, 7.0, 7.001, 3661.125, 3670.0]
    assert [r["binding"] for r in rows] == [
        "none",
        "none",
        "exact",
        "nearest",
        "none",
        "exact",
        "none",
    ]
    assert rows[0]["line_no"] == 3 and rows[0]["text"] == "before speech"


def test_packet_marks_survive_reopen_and_use_checkable_segment_times(
    marked_bundle: tuple[Path, IngestResult], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    state_path, result = marked_bundle
    bundle_id = result.manifest.bundle_id
    private = tmp_path / "Private Notes" / "lectures"
    private.mkdir(parents=True)
    file = private / "marks.txt"
    file.write_text(
        "00:01.500 ? why /Users/synthetic/private/secret.txt\n00:03 !\n00:08 → revisit\n"
    )
    assert run_marks(state_path, bundle_id, "import", str(file)) == 0
    assert str(private) not in capsys.readouterr().out
    with sqlite3.connect(state_path) as connection:
        connection.row_factory = sqlite3.Row
        stored = [dict(row) for row in connection.execute("SELECT * FROM marks")]
        assert [row["import_name"] for row in stored] == ["marks.txt"] * 3
        assert str(private) not in json.dumps(stored)
    rows = list_marks(state_path, bundle_id, capsys)
    assert str(private) not in json.dumps(rows) and PATH_REDACTED in rows[0]["text"]
    with open_state(state_path) as state:
        packet = talk_packet(state, bundle_id)
        markdown = packet_markdown(packet)
        tokens = [m["cite"] for m in packet["marks"] if m["cite"] is not None]
        assert check_citations(state, "\n".join(tokens))["ok"] == 2
        assert check_citations(state, "\n".join(tokens))["problems"] == 0
    short_id = bundle_id.rsplit("-", 1)[1]
    assert packet["marks"] == [
        {
            "t": 1.5,
            "kind": "question",
            "text": f"why {PATH_REDACTED}",
            "binding": "exact",
            "cite": f"[t=00:01 {short_id}#0]",
        },
        {
            "t": 3.0,
            "kind": "important",
            "text": "",
            "binding": "nearest",
            "cite": f"[t=00:01 {short_id}#0]",
        },
        {"t": 8.0, "kind": "follow_up", "text": "revisit", "binding": "none", "cite": None},
    ]
    assert markdown.index("## Marks") > markdown.index("## Segments")
    assert str(private) not in json.dumps(packet) + markdown
    assert f"[t=00:01 {short_id}#0] ? why" in markdown
    assert "00:08 -> revisit (unbound)" in markdown
    assert run_marks(state_path, bundle_id, "list") == 0
    plain = capsys.readouterr().out
    assert str(private) not in plain and "(unbound)" in plain
    assert "#0] !" in plain


@pytest.mark.parametrize("surface", ["listing", "packet"])
def test_complete_mark_output_has_no_unbound_citation_tokens(
    marked_bundle: tuple[Path, IngestResult],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    surface: str,
) -> None:
    state_path, result = marked_bundle
    bundle_id = result.manifest.bundle_id
    file = tmp_path / "notes.txt"
    file.write_text("00:01.500 ? exact\n00:03 ! nearest\n00:08 -> unbound\n")
    assert run_marks(state_path, bundle_id, "import", str(file)) == 0
    capsys.readouterr()
    assert run_marks(state_path, bundle_id, "list") == 0
    listing = capsys.readouterr().out
    short_id = bundle_id.rsplit("-", 1)[1]
    bound = [f"[t=00:01 {short_id}#0]"] * 2
    with open_state(state_path) as state:
        packet = talk_packet(state, bundle_id)
        assert [mark["cite"] for mark in packet["marks"]] == [*bound, None]
        text = listing if surface == "listing" else packet_markdown(packet)
        report = check_citations(state, text)
    valid = [row["token"] for row in report["citations"] if row["status"] == "ok"]
    assert valid == (
        bound
        if surface == "listing"
        else [
            f"[t=00:01 {short_id}#0]",
            f"[t=00:10 {short_id}#1]",
            f"[t=1:01:01 {short_id}#2]",
            *bound,
        ]
    )
    # The complete packet includes a literal format example in its instructions.
    # That existing non-citation is the only permitted checker problem.
    problems = [row["token"] for row in report["citations"] if row["status"] != "ok"]
    assert problems == ([] if surface == "listing" else ["[t=MM:SS <12hex>#<seg>]"])
    assert report["problems"] == (0 if surface == "listing" else 1)
    assert "00:08 -> unbound (unbound)" in text


@pytest.mark.parametrize("command", ["import", "list"])
def test_not_ready_and_unknown_bundles_are_refused_without_marks(
    marked_bundle: tuple[Path, IngestResult],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    command: str,
) -> None:
    state_path, result = marked_bundle
    file = tmp_path / "notes.txt"
    file.write_text("00:01 ? valid\n")
    (result.bundle_dir / "transcript/segments.json").write_text("[]")
    args = (str(file),) if command == "import" else ()
    for bundle_id in (result.manifest.bundle_id, "unknown", str(result.bundle_dir)):
        assert run_marks(state_path, bundle_id, command, *args) == 3
        captured = capsys.readouterr()
        assert captured.out == "" and str(tmp_path) not in captured.err
    with sqlite3.connect(state_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM marks").fetchone()[0] == 0


def test_empty_packet_and_v3_upgrade_keep_existing_evidence_searchable(
    marked_bundle: tuple[Path, IngestResult],
) -> None:
    state_path, result = marked_bundle
    with sqlite3.connect(state_path) as connection:
        connection.execute("DROP TABLE IF EXISTS marks")
        connection.execute("PRAGMA user_version = 3")
    with open_state(state_path) as state:
        packet = talk_packet(state, result.manifest.bundle_id)
        assert packet["marks"] == []
        assert "## Marks" not in packet_markdown(packet)
        assert state.indexed_segment_count() == 3
    with sqlite3.connect(state_path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 4
        assert connection.execute("SELECT COUNT(*) FROM marks").fetchone()[0] == 0


@pytest.mark.parametrize(
    "args",
    [
        [],
        ["import"],
        ["list"],
        ["unknown"],
        ["list", "id", "extra"],
        ["import", "id", "file", "--state"],
    ],
)
def test_marks_usage_errors(args: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["marks", *args]) == 2
    assert "usage:" in capsys.readouterr().err


def test_interrupted_v4_migration_keeps_v3_readable_and_retries(
    marked_bundle: tuple[Path, IngestResult], monkeypatch: pytest.MonkeyPatch
) -> None:
    from lectern.state import AutomationStateStore

    state_path, result = marked_bundle
    with sqlite3.connect(state_path) as connection:
        connection.execute("DROP TABLE marks")
        connection.execute("PRAGMA user_version = 2")
    migrate = AutomationStateStore._create_schema_v4  # pyright: ignore[reportPrivateUsage]

    def interrupt(store: AutomationStateStore) -> None:
        raise RuntimeError("synthetic interruption before marks schema")

    monkeypatch.setattr(AutomationStateStore, "_create_schema_v4", interrupt)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        open_state(state_path)
    with sqlite3.connect(state_path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 3
    monkeypatch.setattr(AutomationStateStore, "_create_schema_v4", migrate)
    with open_state(state_path) as state:
        assert state.list_marks(result.manifest.bundle_id) == []
        assert state.indexed_segment_count() == 3


def test_null_kind_identity_and_fraction_aliases(
    marked_bundle: tuple[Path, IngestResult], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    state_path, result = marked_bundle
    file = tmp_path / "plain.txt"
    file.write_text("00:01.5   ordinary prose  \n0:00:01.500 ordinary prose\n")
    assert run_marks(state_path, result.manifest.bundle_id, "import", str(file)) == 0
    assert capsys.readouterr().out.strip() == "added 1, already present 1, malformed 0"
    rows = list_marks(state_path, result.manifest.bundle_id, capsys)
    digest_input = f"{result.manifest.bundle_id}\x001500\x00\x00ordinary prose"
    assert rows[0]["mark_id"] == hashlib.sha256(digest_input.encode()).hexdigest()[:16]
    assert rows[0]["kind"] is None


def test_file_and_store_errors_are_path_free(
    marked_bundle: tuple[Path, IngestResult], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    state_path, result = marked_bundle
    file = tmp_path / "private" / "missing.txt"
    assert run_marks(state_path, result.manifest.bundle_id, "import", str(file)) == 1
    captured = capsys.readouterr()
    assert str(tmp_path) not in captured.out + captured.err
    file.parent.mkdir()
    file.write_bytes(b"\xff")
    assert run_marks(state_path, result.manifest.bundle_id, "import", str(file)) == 1
    captured = capsys.readouterr()
    assert str(tmp_path) not in captured.out + captured.err
    bad_store = tmp_path / "bad.sqlite"
    bad_store.write_bytes(b"not a database")
    assert run_marks(bad_store, result.manifest.bundle_id, "list") == 3
    captured = capsys.readouterr()
    assert str(tmp_path) not in captured.out + captured.err


@pytest.mark.parametrize("json_output", [False, True])
def test_list_uses_packet_cite_rule_for_non_ingest_bundle_ids(
    marked_bundle: tuple[Path, IngestResult],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    json_output: bool,
) -> None:
    state_path, result = marked_bundle
    bundle_id = "migrated with space"
    manifest = Manifest.load(result.bundle_dir)
    manifest.bundle_id = bundle_id
    manifest.save(result.bundle_dir)
    with sqlite3.connect(state_path) as connection:
        connection.execute("UPDATE library_bundles SET bundle_id = ?", (bundle_id,))
    file = tmp_path / "notes.txt"
    file.write_text("00:01.5 ? check\n")
    assert run_marks(state_path, bundle_id, "import", str(file)) == 0
    capsys.readouterr()
    with open_state(state_path) as state:
        packet = talk_packet(state, bundle_id)
        cite = packet["marks"][0]["cite"]
        assert cite == f"[t=00:01 {hashlib.sha256(bundle_id.encode()).hexdigest()[:12]}#0]"
        assert check_citations(state, cite)["ok"] == 1
    args = ["--json"] if json_output else []
    assert run_marks(state_path, bundle_id, "list", *args) == 0
    output = capsys.readouterr().out
    if json_output:
        assert json.loads(output)["marks"][0]["bundle_id"] == bundle_id
    else:
        assert output.strip() == f"{cite} ? check"


def test_unknown_duration_and_missing_end_do_not_invent_evidence() -> None:
    from lectern.bundle import TranscriptSegmentRecord
    from lectern.marks import bind_mark, parse_mark

    parsed = parse_mark("1:00:00 a late thought", duration_s=None)
    assert parsed is not None
    segment = TranscriptSegmentRecord(
        id=0, start_s=0.0, end_s=None, text="Synthetic untimed evidence.", source="synthetic"
    )
    mark = bind_mark("synthetic", parsed, [segment])
    assert mark["t_s"] == 3600.0
    assert mark["binding"] == "none" and mark["segment_id"] is None


@pytest.mark.parametrize(
    ("end_s", "time", "binding", "segment_id"),
    [
        (4.005, "00:09.005", "nearest", 0),
        (4.005, "00:09.004", "nearest", 0),
        (4.005, "00:09.006", "none", None),
        (4.005000001, "00:09.005", "nearest", 0),
        (4.004999999, "00:09.005", "none", None),
        (4.004999999999999, "00:09.005", "none", None),
    ],
)
def test_fractional_nearest_cutoff_uses_inclusive_decimal_times(
    end_s: float, time: str, binding: str, segment_id: int | None
) -> None:
    from lectern.bundle import TranscriptSegmentRecord
    from lectern.marks import bind_mark, parse_mark

    segment = TranscriptSegmentRecord(
        id=0, start_s=1.0, end_s=end_s, text="Synthetic cutoff evidence.", source="synthetic"
    )
    parsed = parse_mark(f"{time} ? cutoff", duration_s=20.0)
    assert parsed is not None
    mark = bind_mark("synthetic", parsed, [segment])
    assert (mark["binding"], mark["segment_id"]) == (binding, segment_id)


def test_fractional_future_segment_does_not_bind_backward() -> None:
    from lectern.bundle import TranscriptSegmentRecord
    from lectern.marks import bind_mark, parse_mark

    segment = TranscriptSegmentRecord(
        id=0,
        start_s=9.005000001,
        end_s=9.006,
        text="Synthetic future evidence.",
        source="synthetic",
    )
    parsed = parse_mark("00:09.005 ? before speech", duration_s=20.0)
    assert parsed is not None
    mark = bind_mark("synthetic", parsed, [segment])
    assert (mark["binding"], mark["segment_id"]) == ("none", None)


def test_fractional_cutoff_preserves_exact_priority_and_nearest_ties() -> None:
    from lectern.bundle import TranscriptSegmentRecord
    from lectern.marks import bind_mark, parse_mark

    segments = [
        TranscriptSegmentRecord(
            id=segment_id,
            start_s=start_s,
            end_s=4.005,
            text="Synthetic tied evidence.",
            source="synthetic",
        )
        for segment_id, start_s in [(2, 1.0), (0, 2.0), (1, 2.0)]
    ]
    cutoff = parse_mark("00:09.005 ? cutoff", duration_s=20.0)
    assert cutoff is not None
    nearest = bind_mark("synthetic", cutoff, segments)
    assert (nearest["binding"], nearest["segment_id"]) == ("nearest", 1)

    covered = parse_mark("00:03 ? covered", duration_s=20.0)
    assert covered is not None
    exact = bind_mark("synthetic", covered, segments)
    assert (exact["binding"], exact["segment_id"]) == ("exact", 2)


@pytest.mark.parametrize("time", ["-00:01", "00:60", "60:00", "00:01.1234", "9" * 5000 + ":00:00"])
def test_invalid_time_components_are_reportable(time: str) -> None:
    from lectern.marks import parse_mark

    with pytest.raises(ValueError, match="time"):
        parse_mark(f"{time} ? question", duration_s=3670.0)
