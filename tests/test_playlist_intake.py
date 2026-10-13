"""Playlist intake acceptance checks: invented metadata and synthetic audio only."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from lectern import cli
from lectern.automation import AutomationError, open_state
from lectern.bundle import SourceDocument, StageName
from lectern.ingest import IngestResult, ingest_local, plan_local_bundle_id

FIXTURE = Path(__file__).parent / "fixtures" / "synthetic_talk.wav"


def _media(inbox: Path, stem: str) -> Path:
    inbox.mkdir(parents=True, exist_ok=True)
    media = inbox / f"{stem}.wav"
    shutil.copyfile(FIXTURE, media)
    media.with_suffix(".transcript.txt").write_text("An invented lecture about maps.\n")
    return media


def test_folder_playlist_intake_orders_titles_and_keeps_only_allowed_metadata(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    inbox = tmp_path / "inbox"
    second = _media(inbox, "a_second")
    first = _media(inbox, "z_first")
    first.with_suffix(".fr.vtt").write_text(
        "WEBVTT\n\n00:00.000 --> 00:01.000\nUne carte inventée.\n"
    )
    for media, index, title in ((second, 2, "Paths through maps"), (first, 1, "Making maps")):
        media.with_suffix(".info.json").write_text(
            json.dumps(
                {
                    "id": f"invented-talk-{index}",
                    "title": title,
                    "uploader": "Invented Map Club",
                    "upload_date": "20260907",
                    "duration": 999,
                    "webpage_url": f"https://example.invalid/talks/{index}",
                    "language": "ja",
                    "playlist_id": "invented-series",
                    "playlist": "Map lectures",
                    "playlist_index": index,
                    "extractor_key": "Youtube",
                    "http_headers": {"Authorization": "PRIVATE-SENTINEL"},
                    "_filename": "/private/PRIVATE-SENTINEL.wav",
                }
            )
        )
    state_path = tmp_path / "state.sqlite"
    with open_state(state_path) as state:
        source = state.add_local_folder_source("inbox", inbox)
        queued = state.scan_source(source.id).queued
        assert len(queued) == 2
        results: list[IngestResult] = []
        for item in queued:
            state.approve_queue_item(item.id)
            results.append(state.ingest_queue_item(item.id, tmp_path / "bundles"))
    args = ["library", "list", "--state", str(state_path)]
    assert cli.main([*args, "--playlist", "--json"]) == 0
    rows: list[dict[str, Any]] = json.loads(capsys.readouterr().out)["bundles"]
    assert [row["title"] for row in rows] == ["Making maps", "Paths through maps"]
    assert [row["external"]["playlist_index"] for row in rows] == [1, 2]
    assert cli.main([*args, "--json"]) == 0
    assert {row["title"] for row in json.loads(capsys.readouterr().out)["bundles"]} == {
        "Making maps",
        "Paths through maps",
    }
    assert cli.main([*args, "--playlist", "invented-series", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["bundles"] == rows
    assert cli.main([*args, "--playlist", "missing-series", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["bundles"] == []
    assert cli.main([*args, "--playlist"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "Map lectures (invented-series)",
        f"1\t{rows[0]['bundle_id']}\tMaking maps",
        f"2\t{rows[1]['bundle_id']}\tPaths through maps",
    ]
    for result in results:
        record = SourceDocument.model_validate_json((result.bundle_dir / "source.json").read_text())
        assert record.source == result.manifest.source
        external = record.source.external
        assert external is not None
        assert external.model_dump() == {
            "provider": "Youtube",
            "id": f"invented-talk-{external.playlist_index}",
            "url": f"https://example.invalid/talks/{external.playlist_index}",
            "playlist_id": "invented-series",
            "playlist_title": "Map lectures",
            "playlist_index": external.playlist_index,
            "language_hint": "ja",
        }
        assert record.source.channel == "Invented Map Club"
        assert record.source.published is not None
        assert record.source.published.date().isoformat() == "2026-09-07"
        assert record.source.duration_s == 2.9676875
        metadata = json.loads((result.bundle_dir / "transcript/metadata.json").read_text())
        assert metadata["language"] == ("fr" if external.playlist_index == 1 else None)
        facts = ["Invented Map Club", "2026-09-07", external.url, "Map lectures #"]
        overview = (result.bundle_dir / "analysis/summary.md").read_text()
        assert all(fact is not None and fact in overview for fact in facts)
        show = ["library", "show", result.manifest.bundle_id, "--state", str(state_path)]
        assert cli.main(show) == 0
        plain = capsys.readouterr().out
        assert all(fact is not None and fact in plain for fact in facts)
        assert cli.main([*show, "--json"]) == 0
        shown = json.loads(capsys.readouterr().out)
        assert shown["channel"] == "Invented Map Club"
        assert shown["published"] == "2026-09-07"
        assert shown["url"] == external.url
        assert shown["playlist"] == f"Map lectures #{external.playlist_index}"
        for path in result.bundle_dir.rglob("*"):
            if path.is_file():
                payload = path.read_bytes()
                assert all(
                    key not in payload
                    for key in (b'"http_headers"', b'"_filename"', b"PRIVATE-SENTINEL")
                )


def test_changed_info_json_refuses_approved_queue_ingest(tmp_path: Path) -> None:
    media = _media(tmp_path / "inbox", "talk")
    info = media.with_suffix(".info.json")
    info.write_text('{"title": "An invented original title"}')
    old_id = plan_local_bundle_id(media)
    with open_state(tmp_path / "state.sqlite") as state:
        source = state.add_local_folder_source("inbox", media.parent)
        item = state.scan_source(source.id).queued[0]
        state.approve_queue_item(item.id)
        info.write_text('{"title": "A changed invented title"}')
        with pytest.raises(AutomationError, match="changed since queue approval"):
            state.ingest_queue_item(item.id, tmp_path / "bundles")
        assert not (tmp_path / "bundles").exists()
        assert plan_local_bundle_id(media) != old_id
        rescanned = state.scan_source(source.id).queued[0]
        state.approve_queue_item(rescanned.id)
        result = state.ingest_queue_item(rescanned.id, tmp_path / "bundles")
        assert result.manifest.source.title == "A changed invented title"
        info.unlink()
        with pytest.raises(AutomationError, match="changed since queue approval"):
            state.approve_queue_item(rescanned.id)
            state.ingest_queue_item(rescanned.id, tmp_path / "bundles")


def test_malformed_info_json_warns_and_keeps_filename_title(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    media = _media(tmp_path / "inbox", "broken_metadata")
    media.with_suffix(".info.json").write_text('{"http_headers": "PRIVATE-SENTINEL", bad json')
    result = ingest_local(media, tmp_path / "direct-bundles")
    assert result.manifest.source.title == "Broken Metadata"
    assert result.manifest.source.external is None
    assert result.warnings
    assert "info.json" in result.warnings[0]
    assert "PRIVATE-SENTINEL" not in result.warnings[0]
    assert result.manifest.stages[StageName.ACQUIRE].warnings == list(result.warnings)
    state_path = tmp_path / "state.sqlite"
    with open_state(state_path) as state:
        queued = state.ingest_one_shot(media, tmp_path / "bundles")
        replay = state.ingest_one_shot(media, tmp_path / "bundles")
    assert queued.warnings == replay.warnings == result.warnings
    assert cli.main(["library", "list", "--playlist", "--json", "--state", str(state_path)]) == 0
    row = json.loads(capsys.readouterr().out)["bundles"][0]
    assert row["title"] == "Broken Metadata"
    assert row["external"] is None
    assert cli.main(["library", "list", "--playlist", "--state", str(state_path)]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "Without playlist",
        f"\t{queued.manifest.bundle_id}\tBroken Metadata",
    ]
