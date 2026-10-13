"""Six acceptance checks using invented captions and synthetic local audio."""

from __future__ import annotations

import hashlib
import json
import shutil
import wave
from pathlib import Path
from typing import Any

import pytest

from lectern import cli
from lectern.asr import ASRResult, ASRSegment
from lectern.automation import AutomationError, open_state
from lectern.bundle import TranscriptMetadataDocument, TranscriptSegmentsDocument
from lectern.ingest import ingest_local, plan_local_bundle_id

FIXTURES = Path(__file__).parent / "fixtures"


def _media(tmp_path: Path, caption: str | None = None) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    media = tmp_path / "talk.wav"
    shutil.copyfile(FIXTURES / "synthetic_talk.wav", media)
    if caption is not None:
        suffix = caption.removeprefix("timed_plain").removeprefix("timed_rolling")
        shutil.copyfile(FIXTURES / caption, media.with_suffix(suffix))
    return media


def _metadata(bundle: Path) -> dict[str, Any]:
    payload = json.loads((bundle / "transcript/metadata.json").read_text())
    TranscriptMetadataDocument.model_validate(payload)
    return payload


def _segments(bundle: Path) -> list[dict[str, Any]]:
    payload = json.loads((bundle / "transcript/segments.json").read_text())
    TranscriptSegmentsDocument.model_validate(payload)
    return payload


def test_webvtt_sidecar_records_times_and_filename_language(tmp_path: Path) -> None:
    media = _media(tmp_path, "timed_plain.fr.vtt")
    media.with_suffix(".transcript.txt").write_text("Not the chosen transcript.")
    planned = plan_local_bundle_id(media)
    result = ingest_local(media, tmp_path / "bundles")
    assert result.manifest.bundle_id == planned
    assert _segments(result.bundle_dir) == [
        {
            "id": 0,
            "start_s": 0.25,
            "end_s": 1.5,
            "text": "Bonjour & bienvenue.",
            "source": "caption_vtt",
        },
        {
            "id": 1,
            "start_s": 1.5,
            "end_s": 3.0,
            "text": "Deux > un. Voilà.",
            "source": "caption_vtt",
        },
    ]
    metadata = _metadata(result.bundle_dir)
    assert metadata["language"] == "fr"
    assert metadata["method"] == "caption_sidecar_vtt"
    assert metadata["backend"]["sidecar_filename"] == "talk.fr.vtt"


def test_rolling_webvtt_emits_spoken_phrases_once(tmp_path: Path) -> None:
    result = ingest_local(_media(tmp_path, "timed_rolling.en.vtt"), tmp_path / "bundles")
    assert (result.bundle_dir / "transcript/transcript.md").read_text() == (
        "We build maps.\nThen we test them.\nFinally we share results.\n"
    )
    assert [row["text"] for row in _segments(result.bundle_dir)] == [
        "We build maps.",
        "Then we test them.",
        "Finally we share results.",
    ]


def test_srt_sidecar_records_timed_segments(tmp_path: Path) -> None:
    media = _media(tmp_path, "timed_plain.srt")
    assert (
        cli.main(
            [
                "ingest",
                str(media),
                "--output",
                str(tmp_path / "bundles"),
                "--state",
                str(tmp_path / "state.sqlite"),
                "--language",
                "de",
            ]
        )
        == 0
    )
    bundle = next((tmp_path / "bundles").iterdir())
    assert _segments(bundle) == [
        {"id": 0, "start_s": 0.25, "end_s": 1.5, "text": "A small map.", "source": "caption_srt"},
        {
            "id": 1,
            "start_s": 1.5,
            "end_s": 3.0,
            "text": "Two paths & one destination.",
            "source": "caption_srt",
        },
    ]
    assert _metadata(bundle)["method"] == "caption_sidecar_srt"
    assert _metadata(bundle)["language"] == "de"


def test_builtin_asr_records_fake_backend_and_detected_language(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeBackend:
        def __init__(self, model: str) -> None:
            self.model = model

        def transcribe(self, audio: Path, *, language: str | None = None) -> ASRResult:
            with wave.open(str(audio)) as wav:
                assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (16000, 1, 2)
            return ASRResult(
                segments=(ASRSegment(0.25, 1.5, "Une carte locale."),),
                language=language or "fr",
                language_probability=0.98,
                model=self.model,
                library_version="test-version",
                compute_type="int8",
            )

    monkeypatch.setattr("lectern.ingest.create_asr_backend", FakeBackend)
    media = _media(tmp_path)
    assert (
        cli.main(
            [
                "ingest",
                str(media),
                "--output",
                str(tmp_path / "bundles"),
                "--state",
                str(tmp_path / "state.sqlite"),
            ]
        )
        == 0
    )
    bundle = next((tmp_path / "bundles").iterdir())
    assert _segments(bundle) == [
        {
            "id": 0,
            "start_s": 0.25,
            "end_s": 1.5,
            "text": "Une carte locale.",
            "source": "faster_whisper",
        }
    ]
    metadata = _metadata(bundle)
    assert metadata["language"] == "fr"
    assert metadata["backend"] == {
        "kind": "faster_whisper",
        "model": "small",
        "library_version": "test-version",
        "compute_type": "int8",
        "detected_language": "fr",
        "language_probability": 0.98,
    }
    assert metadata["remote_services"]["lectern_invoked"] is False
    assert metadata["remote_services"]["allowed"] is False
    # A second invocation exercises model selection, language override and
    # non-plannable ASR replay through the same end-to-end path.
    assert (
        cli.main(
            [
                "ingest",
                str(media),
                "--output",
                str(tmp_path / "bundles"),
                "--state",
                str(tmp_path / "state.sqlite"),
                "--asr-model",
                "tiny",
                "--language",
                "es",
            ]
        )
        == 0
    )
    alternate = next(path for path in (tmp_path / "bundles").iterdir() if path != bundle)
    assert _metadata(alternate)["language"] == "es"
    assert _metadata(alternate)["backend"]["model"] == "tiny"


def test_missing_asr_extra_names_install_fix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def unavailable(model: str) -> None:
        raise ImportError("faster_whisper unavailable")

    monkeypatch.setattr("lectern.ingest.create_asr_backend", unavailable, raising=False)
    assert (
        cli.main(
            [
                "ingest",
                str(_media(tmp_path)),
                "--output",
                str(tmp_path / "bundles"),
                "--state",
                str(tmp_path / "state.sqlite"),
            ]
        )
        == 1
    )
    assert "lectern[asr]" in capsys.readouterr().err


def test_queue_caption_bytes_are_bound_to_approval(tmp_path: Path) -> None:
    media = _media(tmp_path / "inbox", "timed_plain.fr.vtt")
    caption = media.with_suffix(".fr.vtt")
    state_path = tmp_path / "state.sqlite"
    with open_state(state_path) as state:
        source = state.add_local_folder_source("talks", media.parent)
        queue = state.scan_source(source.id).queued[0]
        approved = state.approve_queue_item(queue.id)
    assert (
        cli.main(
            [
                "queue",
                "ingest",
                approved.id,
                "--output",
                str(tmp_path / "bundles"),
                "--state",
                str(state_path),
                "--asr-model",
                "tiny",
                "--language",
                "fr",
            ]
        )
        == 0
    )
    bundle = next((tmp_path / "bundles").iterdir())
    record = json.loads((bundle / "source.json").read_text())
    assert record["transcript_sidecar"] == {
        "sha256": hashlib.sha256(caption.read_bytes()).hexdigest(),
        "bytes": len(caption.read_bytes()),
    }
    assert _metadata(bundle)["language"] == "fr"
    with open_state(state_path) as state:
        state.approve_queue_item(queue.id)
        caption.write_text("WEBVTT\n\n00:00.000 --> 00:01.000\nChanged words.\n")
        with pytest.raises(AutomationError, match="changed since queue approval"):
            state.ingest_queue_item(queue.id, tmp_path / "bundles")
