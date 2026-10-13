from __future__ import annotations

import json
import shlex
import sqlite3
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from lectern import cli
from lectern import state as state_module
from lectern.agent_packet import packet_markdown
from lectern.automation import open_state
from lectern.bundle import Manifest, StageName, StageState
from lectern.ingest import IngestResult

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "synthetic_talk.wav"
SEGMENTS = [
    {"start_s": 0.0, "end_s": 0.9, "text": "A synthetic opening."},
    {"start_s": 1.25, "end_s": 2.9, "text": "Evidence supports careful reading."},
    {"start_s": 3661.9, "end_s": 3662.5, "text": "A synthetic hour-later moment."},
]


@pytest.fixture
def agent_bundle(tmp_path: Path) -> tuple[Path, IngestResult]:
    media = tmp_path / "synthetic_talk.wav"
    media.write_bytes(FIXTURE.read_bytes())
    script = tmp_path / "transcriber.py"
    script.write_text(f"print({json.dumps({'segments': SEGMENTS})!r})\n", encoding="utf-8")
    state_path = tmp_path / "state.sqlite"
    with open_state(state_path) as state:
        result = state.ingest_one_shot(
            media,
            tmp_path / "bundles",
            transcriber_command=f"{shlex.quote(sys.executable)} {shlex.quote(str(script))}",
        )
    return state_path, result


def test_json_packet_preserves_existing_anchors_and_reports_layers(
    agent_bundle: tuple[Path, IngestResult], capsys: pytest.CaptureFixture[str]
) -> None:
    state, result = agent_bundle
    bundle_id = result.manifest.bundle_id
    args = ["export", bundle_id, "--format", "json", "--state", str(state)]
    assert cli.main(args) == 0
    packet = json.loads(capsys.readouterr().out)
    assert packet["packet"] == "lectern.talk-packet"
    assert packet["packet_version"] == "0.1"
    assert packet["stability"] == "experimental"
    assert packet["bundle_id"] == bundle_id
    assert packet["source"]["title"] == "Synthetic Talk"
    assert packet["source"]["external"] is None
    metadata = json.loads((result.bundle_dir / "transcript/metadata.json").read_text())
    assert packet["layers"]["transcript"] == {
        "present": True,
        "method": "local_command_json",
        "language": None,
        "evidence_limit": metadata["evidence_limit"],
        "segment_count": 3,
    }
    assert packet["layers"]["visual"] == {"present": False, "reason": "not requested"}
    for layer in ("speakers", "translation"):
        assert packet["layers"][layer] == {"present": False, "reason": "not processed"}
    for index, segment in enumerate(packet["segments"]):
        assert {key: segment[key] for key in SEGMENTS[index]} == SEGMENTS[index]
        assert (
            cli.main(["library", "cite", bundle_id, str(index), "--state", str(state), "--json"])
            == 0
        )
        citation = json.loads(capsys.readouterr().out)
        anchor = segment["anchor"]
        short_id = bundle_id.rsplit("-", 1)[1]
        assert anchor["short_id"] == short_id
        assert anchor["cite"] == f"[t={('00:00', '00:01', '1:01:01')[index]} {short_id}#{index}]"
        assert {key: anchor[key] for key in citation["anchor"]} == citation["anchor"]
        assert anchor["selectors"] == [
            {"type": "timecode_range", "start_s": segment["start_s"], "end_s": segment["end_s"]},
            {"type": "quote", "exact": segment["text"]},
        ]
    manifest = Manifest.load(result.bundle_dir)
    manifest.stages[StageName.DIARIZE].state = StageState.DONE
    manifest.save(result.bundle_dir)
    assert cli.main(args) == 0
    layers = json.loads(capsys.readouterr().out)["layers"]
    assert layers["visual"]["present"] is False
    assert layers["speakers"]["present"] is True

    # A migrated bundle's free-form segment `source` may hold a path; it must not leak.
    read_evidence = state_module.AutomationStateStore.get_registered_evidence

    def with_path_source(store: state_module.AutomationStateStore, wanted: str):  # noqa: ANN202
        evidence = read_evidence(store, wanted)
        evidence.segments.root[0].source = "/Users/someone/private/talk.wav"
        return evidence

    store_class = state_module.AutomationStateStore
    with patch.object(store_class, "get_registered_evidence", with_path_source):
        assert cli.main(args) == 0
    assert "/Users/someone" not in capsys.readouterr().out


def test_markdown_packet_instructs_agents_and_formats_citations(
    agent_bundle: tuple[Path, IngestResult], capsys: pytest.CaptureFixture[str]
) -> None:
    state, result = agent_bundle
    assert cli.main(["--help"]) == 0
    help_text = capsys.readouterr().out
    assert "export" in help_text and "check-citations" in help_text
    bundle_id = result.manifest.bundle_id
    assert cli.main(["export", bundle_id, "--format", "md", "--state", str(state)]) == 0
    output = capsys.readouterr().out
    assert output.startswith(
        "Cite moments as `[t=MM:SS <12hex>#<seg>]`; Lectern verifies these with "
        "`lectern check-citations`.\n"
    )
    assert "experimental" in output
    assert "# Synthetic Talk" in output
    assert "Layers" in output and "not processed" in output
    for index, (token, segment) in enumerate(
        zip(("00:00", "00:01", "1:01:01"), SEGMENTS, strict=True)
    ):
        assert f"[t={token} {bundle_id.rsplit('-', 1)[1]}#{index}] {segment['text']}" in output
    assert cli.main(["export", bundle_id, "--format", "json", "--state", str(state)]) == 0
    packet = json.loads(capsys.readouterr().out)
    # JSON evidence stays raw; only the Markdown renderer escapes controls.
    fields = [
        (packet["source"], "title"),
        (packet["source"], "channel"),
        (packet["layers"]["transcript"], "evidence_limit"),
        (packet["segments"][0], "text"),
    ]
    packet["source"]["external"] = {"url": "https://example.test/\x1b\x00\x85"}
    for owner, key in fields:
        owner[key] = "synthetic\x1b\x00\x85\nfield"
    output = packet_markdown(packet)
    assert output.count("synthetic\\u001b\\u0000\\u0085\\u000afield") == 4
    assert "External url: https://example.test/\\u001b\\u0000\\u0085" in output
    assert not any(ord(c) < 0x20 and c != "\n" or 0x7F <= ord(c) <= 0x9F for c in output)
    assert packet["segments"][0]["text"] == "synthetic\x1b\x00\x85\nfield"


def test_citation_checker_counts_problems_and_uncited_paragraphs(
    agent_bundle: tuple[Path, IngestResult], capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    state, result = agent_bundle
    bundle_id = result.manifest.bundle_id
    short_id = bundle_id.rsplit("-", 1)[1]
    review = tmp_path / "review.md"
    content = (
        "# Review\n\n-\n\n1.\n\n"
        f"Opening [t=00:00 {bundle_id}#0].\n\n"
        f"Later [t=1:01:01 {short_id}#2].\n\n"
        f"Wrong time [t=00:02 {bundle_id}#1].\n\n"
        f"Missing segment [t=00:00 {bundle_id}#99].\n\n"
        f"Impossible time [t=09:99 {short_id}#100].\n\n"
        f"Invalid hour minutes [t=1:99:01 {short_id}#2].\n\n"
        f"Invalid segment [t=00:00 {short_id}#nope].\n\n"
        "An unsupported claim.\n"
    )
    review.write_text(content, encoding="utf-8")
    args = ["check-citations", str(review), "--state", str(state)]
    with patch(
        "lectern.state._library_status_from_row",
        wraps=state_module._library_status_from_row,  # pyright: ignore[reportPrivateUsage]
    ) as derive_status:
        assert cli.main([*args, "--json"]) == 1
        assert derive_status.call_count == 1
    report = json.loads(capsys.readouterr().out)
    assert [item["status"] for item in report["citations"]] == [
        "ok",
        "ok",
        "time_mismatch",
        "unknown_segment",
        "malformed",
        "malformed",
        "malformed",
    ]
    assert report["ok"] == 2 and report["problems"] == 5
    assert report["citations"][1]["bundle_id"] == bundle_id
    assert report["uncited_paragraphs"] == 1
    assert cli.main(args) == 1
    output = capsys.readouterr().out
    assert len(output.splitlines()) == 6
    assert output.endswith("citations: 2 ok, 5 problems; uncited paragraphs: 1\n")
    review.write_text(
        content.replace(f"00:02 {bundle_id}#1", f"00:01 {bundle_id}#1")
        .replace(f"00:00 {bundle_id}#99", f"00:00 {bundle_id}#0")
        .replace(f"09:99 {short_id}#100", f"00:00 {short_id}#0")
        .replace(f"1:99:01 {short_id}#2", f"1:01:01 {short_id}#2")
        .replace(f"00:00 {short_id}#nope", f"00:00 {short_id}#0"),
        encoding="utf-8",
    )
    assert cli.main(args) == 0
    assert capsys.readouterr().out == "citations: 7 ok, 0 problems; uncited paragraphs: 1\n"
    review.write_text(
        "Unknown [t=00:00 missing-bundle#0].\n\nUnknown suffix [t=00:00 ffffffffffff#0].",
        encoding="utf-8",
    )
    assert cli.main([*args, "--json"]) == 1
    assert [c["status"] for c in json.loads(capsys.readouterr().out)["citations"]] == [
        "unknown_bundle",
        "unknown_bundle",
    ]
    # A collision need not have ready evidence: ambiguity is over registered IDs.
    with sqlite3.connect(state) as connection:
        connection.execute(
            """INSERT INTO library_bundles
               SELECT ?, ?, source_id, source_item_id, queue_item_id, created_at, kind
               FROM library_bundles WHERE bundle_id = ?""",
            (f"other-{short_id}", str(tmp_path / "unavailable-bundle"), bundle_id),
        )
    review.write_text(
        f"Ambiguous [t=00:00 {short_id}#0].\n\nFull ID [t=00:00 {bundle_id}#0].",
        encoding="utf-8",
    )
    assert cli.main([*args, "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert [c["status"] for c in report["citations"]] == ["ambiguous_bundle", "ok"]
    assert report["ok"] == 1 and report["problems"] == 1
    assert report["uncited_paragraphs"] == 0


def test_changed_transcript_invalidates_citation_and_refuses_export(
    agent_bundle: tuple[Path, IngestResult], capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    state, result = agent_bundle
    bundle_id = result.manifest.bundle_id
    review = tmp_path / "review.md"
    review.write_text(f"Opening [t=00:00 {bundle_id}#0].", encoding="utf-8")
    args = ["check-citations", str(review), "--state", str(state), "--json"]
    assert cli.main(args) == 0
    assert json.loads(capsys.readouterr().out)["citations"][0]["status"] == "ok"
    transcript = result.bundle_dir / "transcript/segments.json"
    segments = json.loads(transcript.read_text())
    segments[0]["text"] = "Changed synthetic evidence."
    transcript.write_text(json.dumps(segments), encoding="utf-8")
    assert cli.main(args) == 1
    assert json.loads(capsys.readouterr().out)["citations"][0]["status"] != "ok"
    assert cli.main(["export", bundle_id, "--state", str(state)]) == 3
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "requires ready" in captured.err
    assert str(tmp_path) not in captured.err
