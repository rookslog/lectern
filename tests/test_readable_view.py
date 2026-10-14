from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path

import pytest

from lectern import cli
from lectern.automation import open_state
from lectern.bundle import Manifest, StageName, StageState
from lectern.ingest import IngestResult

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "synthetic_talk.wav"
SEGMENTS = [
    {"start_s": 0.0, "end_s": 1.0, "text": "A synthetic opening sentence."},
    {"start_s": 1.25, "end_s": 2.0, "text": "Evidence anchors support careful reading."},
    {"start_s": 2.5, "end_s": 2.9, "text": "An agent writes the summary."},
]


@pytest.fixture
def readable_bundle(tmp_path: Path) -> tuple[Path, IngestResult]:
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


def test_ingest_overview_contains_facts_not_transcript_sentences(
    readable_bundle: tuple[Path, IngestResult],
) -> None:
    _, result = readable_bundle
    overview = (result.bundle_dir / "analysis/summary.md").read_text(encoding="utf-8")
    assert overview.startswith("# Synthetic Talk\n")
    assert "Duration: 0:00:02" in overview
    assert "Transcript method: local_command_json" in overview
    assert "Segments: 3" in overview
    assert "Words: 14" in overview
    assert "Time span: 00:00–00:02" in overview
    assert "Lectern does not summarize." in overview
    assert "lectern transcript <bundle-id>" in overview
    assert all(str(segment["text"]) not in overview for segment in SEGMENTS)


def test_library_show_exposes_readable_facts_and_preserves_json(
    readable_bundle: tuple[Path, IngestResult], capsys: pytest.CaptureFixture[str]
) -> None:
    state_path, result = readable_bundle
    bundle_id = result.manifest.bundle_id
    args = ["library", "show", bundle_id, "--state", str(state_path)]
    assert cli.main(args) == 0
    plain = capsys.readouterr().out
    for field in (
        f"Bundle: {bundle_id}",
        "Title: Synthetic Talk",
        "Status: ready",
        "Created:",
        "Source kind: one-shot",
        "Duration: 0:00:02",
        "Transcript method: local_command_json",
        "Segments: 3",
        "Words: 14",
        "Transcript: yes",
        "Visual: no",
        "Bundle directory: <path>",
    ):
        assert field in plain
    assert str(state_path.parent) not in plain
    assert cli.main([*args, "--json"]) == 0
    output = capsys.readouterr().out
    payload = json.loads(output)
    with open_state(state_path) as state:
        assert payload["bundle"] == state.get_library_bundle(bundle_id).to_dict()
    assert payload["manifest"] == result.manifest.model_dump(mode="json")
    assert payload["bundle_id"] == bundle_id
    assert payload["title"] == "Synthetic Talk"
    assert payload["source_kind"] == "one-shot"
    assert payload["duration_s"] == 2.9676875
    assert payload["transcript_method"] == "local_command_json"
    assert payload["segment_count"] == 3
    assert payload["word_count"] == 14
    assert payload["layers"] == {"transcript": True, "visual": False, "speakers": False}
    assert payload["bundle_dir"] == "<path>"
    assert str(state_path.parent) not in output
    assert cli.main(["library", "list", "--state", str(state_path)]) == 0
    assert capsys.readouterr().out.rstrip().split("\t") == [
        bundle_id,
        payload["created_at"],
        "ready",
        "Synthetic Talk",
    ]
    manifest = Manifest.load(result.bundle_dir)
    manifest.stages[StageName.VISUAL].state = StageState.DONE
    manifest.save(result.bundle_dir)
    assert cli.main([*args, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["layers"]["visual"] is True


def test_transcript_renders_timestamped_segments_and_refuses_non_ready_bundles(
    readable_bundle: tuple[Path, IngestResult], capsys: pytest.CaptureFixture[str]
) -> None:
    state_path, result = readable_bundle
    assert cli.main(["--help"]) == 0
    assert "transcript" in capsys.readouterr().out
    args = ["transcript", result.manifest.bundle_id, "--state", str(state_path)]
    paragraphs = (
        "[00:00] (#0) A synthetic opening sentence.\n\n"
        "[00:01] (#1) Evidence anchors support careful reading.\n\n"
        "[00:02] (#2) An agent writes the summary.\n"
    )
    assert cli.main(args) == 0
    assert capsys.readouterr().out == "# Synthetic Talk\n\n" + paragraphs
    assert cli.main([*args, "--format", "txt"]) == 0
    assert capsys.readouterr().out == paragraphs
    assert cli.main([*args, "--format", "json"]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "bundle_id": result.manifest.bundle_id,
        "title": "Synthetic Talk",
        "segments": [{"id": index, **segment} for index, segment in enumerate(SEGMENTS)],
    }
    manifest = Manifest.load(result.bundle_dir)
    manifest.stages[StageName.SYNTHESIZE].state = StageState.FAILED
    manifest.stages[StageName.SYNTHESIZE].error = "synthetic failure"
    manifest.save(result.bundle_dir)
    assert cli.main(args) == 3
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "failed" in captured.err
    assert str(state_path.parent) not in captured.err


def test_search_hits_include_title_and_segment_time(
    readable_bundle: tuple[Path, IngestResult], capsys: pytest.CaptureFixture[str]
) -> None:
    state_path, result = readable_bundle
    args = ["library", "search", "Evidence anchors", "--state", str(state_path)]
    assert cli.main([*args, "--json"]) == 0
    hit = json.loads(capsys.readouterr().out)["results"][0]
    assert hit["bundle_id"] == result.manifest.bundle_id
    assert hit["segment_id"] == 1
    assert hit["start_s"] == 1.25
    assert hit["title"] == "Synthetic Talk"
    assert cli.main(args) == 0
    assert capsys.readouterr().out == (
        f"{result.manifest.bundle_id}\t1\t[00:01]\tEvidence anchors support careful reading.\n"
    )
