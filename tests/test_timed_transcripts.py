"""Acceptance checks using invented captions and synthetic local audio."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import wave
from pathlib import Path
from typing import Any

import pytest

from lectern import asr, cli
from lectern.asr import ASRResult, ASRSegment
from lectern.automation import AutomationError, open_state
from lectern.bundle import TranscriptMetadataDocument, TranscriptSegmentsDocument
from lectern.ingest import IngestError, ingest_local, plan_local_bundle_id, prepare_local_ingest
from lectern.sources.local import approval_digest_and_media_size

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
    media = _media(tmp_path / "prefix")
    media.with_suffix(".vtt").write_text(
        "WEBVTT\n\n00:00.000 --> 00:01.000\nI\n\n00:01.000 --> 00:02.000\nIt works.\n"
    )
    prefix_result = ingest_local(media, tmp_path / "bundles")
    assert [row["text"] for row in _segments(prefix_result.bundle_dir)] == ["I", "It works."]


def test_plain_webvtt_keeps_separately_timed_identical_cues(tmp_path: Path) -> None:
    media = _media(tmp_path)
    media.with_suffix(".vtt").write_text(
        "WEBVTT\n\n00:00.000 --> 00:01.000\nNo.\n\n00:01.000 --> 00:02.000\nNo.\n"
    )
    result = ingest_local(media, tmp_path / "bundles")
    assert [
        (row["start_s"], row["end_s"], row["text"]) for row in _segments(result.bundle_dir)
    ] == [
        (0.0, 1.0, "No."),
        (1.0, 2.0, "No."),
    ]


def test_rolling_webvtt_without_inline_times_deduplicates_carried_lines(tmp_path: Path) -> None:
    media = _media(tmp_path)
    media.with_suffix(".vtt").write_text(
        "WEBVTT\n\n00:00.000 --> 00:01.000\nWe build maps.\n\n"
        "00:01.000 --> 00:02.000\nWe build maps.\nThen we test them.\n\n"
        "00:02.000 --> 00:02.500\nThen we test them.\n"
    )
    result = ingest_local(media, tmp_path / "bundles")
    assert [row["text"] for row in _segments(result.bundle_dir)] == [
        "We build maps.",
        "Then we test them.",
    ]


def test_command_transcript_does_not_inherit_caption_language(tmp_path: Path) -> None:
    media = _media(tmp_path, "timed_plain.fr.vtt")
    transcriber = tmp_path / "transcriber.py"
    transcriber.write_text('print(\'{"text": "Command transcript."}\')\n')
    result = ingest_local(
        media,
        tmp_path / "bundles",
        transcriber_command=f'"{sys.executable}" "{transcriber}"',
    )
    assert _metadata(result.bundle_dir)["language"] is None


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
    download_permissions: list[bool] = []

    class FakeBackend:
        def __init__(self, model: str, *, allow_model_download: bool = False) -> None:
            self.model = model
            download_permissions.append(allow_model_download)

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

    monkeypatch.setattr("lectern.asr.create_asr_backend", FakeBackend)
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
    assert metadata["remote_services"]["transcriber_network_posture"] == (
        "local_inference_cached_model_only"
    )
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
                "--allow-model-download",
                "--language",
                "es",
            ]
        )
        == 0
    )
    alternate = next(path for path in (tmp_path / "bundles").iterdir() if path != bundle)
    assert _metadata(alternate)["language"] == "es"
    assert _metadata(alternate)["backend"]["model"] == "tiny"
    assert _metadata(alternate)["remote_services"]["lectern_invoked"] is True
    assert _metadata(alternate)["remote_services"]["transcriber_network_posture"] == (
        "local_inference_model_download_allowed"
    )
    with open_state(tmp_path / "state.sqlite") as state:
        approved = state.approve_queue_item(state.list_queue()[0].id)
    assert (
        cli.main(
            [
                "queue",
                "ingest",
                approved.id,
                "--output",
                str(tmp_path / "bundles"),
                "--state",
                str(tmp_path / "state.sqlite"),
                "--asr-model",
                "small",
                "--allow-model-download",
            ]
        )
        == 0
    )
    downloaded = next(
        path for path in (tmp_path / "bundles").iterdir() if path not in (bundle, alternate)
    )
    assert _metadata(downloaded)["remote_services"]["lectern_invoked"] is True
    assert _metadata(downloaded)["remote_services"]["transcriber_network_posture"] == (
        "local_inference_model_download_allowed"
    )
    (tmp_path / "converted-directory").mkdir()
    (tmp_path / "converted-directory" / "tokenizer.json").write_text("{}")
    monkeypatch.chdir(tmp_path)
    for index, model in enumerate(
        (str(tmp_path / "converted-model"), r"models\converted-model", "converted-directory")
    ):
        local = ingest_local(media, tmp_path / f"local-bundles-{index}", asr_model=model)
        assert _metadata(local.bundle_dir)["backend"]["model"] == (
            "local:converted-directory"
            if model == "converted-directory"
            else "local:converted-model"
        )
    assert download_permissions == [False, True, True, False, False, False]


def test_missing_asr_extra_names_install_fix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def unavailable(model: str, *, allow_model_download: bool = False) -> None:
        raise ImportError("faster_whisper unavailable")

    monkeypatch.setattr("lectern.asr.create_asr_backend", unavailable, raising=False)
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

    def uncached(model: str, *, allow_model_download: bool = False) -> None:
        raise RuntimeError("model weights are not cached")

    monkeypatch.setattr("lectern.asr.create_asr_backend", uncached)
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
    assert "--allow-model-download" in capsys.readouterr().err


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


@pytest.mark.parametrize("renamed_suffix", [".fr.vtt", ".en.srt"])
def test_caption_rename_invalidates_queue_approval(tmp_path: Path, renamed_suffix: str) -> None:
    media = _media(tmp_path / "inbox", "timed_rolling.en.vtt")
    caption = media.with_suffix(".en.vtt")
    with open_state(tmp_path / "state.sqlite") as state:
        source = state.add_local_folder_source("talks", media.parent)
        queue = state.scan_source(source.id).queued[0]
        approved = state.approve_queue_item(queue.id)
        with prepare_local_ingest(
            media, expected_approval_sha256=approved.content_sha256, source_root=media.parent
        ):
            pass
        caption.rename(media.with_suffix(renamed_suffix))
        with pytest.raises(AutomationError, match="rescan before ingest"):
            state.ingest_queue_item(queue.id, tmp_path / "bundles")
        rescanned = state.scan_source(source.id)
        assert len(rescanned.changed) == 1
        assert rescanned.queued[0].content_sha256 != approved.content_sha256


def test_untagged_text_sidecar_preserves_baseline_approval_digest(tmp_path: Path) -> None:
    media = _media(tmp_path)
    shutil.copyfile(
        FIXTURES / "synthetic_talk.transcript.txt", media.with_suffix(".transcript.txt")
    )
    # Computed on 183e8a5 from these synthetic media/sidecar bytes.
    baseline_digest = "71081ac54e089f9f6b5d4f6148c9cd64b4e7a85f039e0341aad6cd35905679db"
    assert approval_digest_and_media_size(media)[0] == baseline_digest
    with prepare_local_ingest(media, expected_approval_sha256=baseline_digest) as prepared:
        assert prepared.approval_sha256 == baseline_digest


def test_untagged_caption_preserves_baseline_approval_digest(tmp_path: Path) -> None:
    media = _media(tmp_path)
    media.with_suffix(".vtt").write_text(
        "WEBVTT\n\n00:00.000 --> 00:01.000\nNo.\n\n00:01.000 --> 00:02.000\nNo.\n"
    )
    # Computed on 183e8a5; no language tag means no added approval component.
    baseline_digest = "cf3f96278d6a24d91f2c1ba278e89df7e68aa6d3df81b7812bfd0ebbe086ae6f"
    assert approval_digest_and_media_size(media)[0] == baseline_digest
    with prepare_local_ingest(media, expected_approval_sha256=baseline_digest) as prepared:
        assert prepared.approval_sha256 == baseline_digest


def test_caption_approval_binds_derived_metadata_not_raw_filename(tmp_path: Path) -> None:
    media = _media(tmp_path, "timed_rolling.en.vtt")
    baseline_digest = approval_digest_and_media_size(media)[0]
    renamed = media.with_name("renamed.wav")
    media.rename(renamed)
    media.with_suffix(".en.vtt").rename(renamed.with_suffix(".en.vtt"))
    assert approval_digest_and_media_size(renamed)[0] == baseline_digest
    with prepare_local_ingest(renamed, expected_approval_sha256=baseline_digest) as prepared:
        assert prepared.approval_sha256 == baseline_digest


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("org/custom-whisper", "org/custom-whisper"),
        ("models/nonexistent-relative-model", "models/nonexistent-relative-model"),
        ("/Users/x/models/m", "local:m"),
        ("~/models/m", "local:m"),
        ("C:/models/m", "local:m"),
        ("C:m", "local:m"),
        (r"C:\models\m", "local:m"),
        (r"models\m", "local:m"),
    ],
)
def test_model_identifier_preserves_hub_ids_and_redacts_paths(model: str, expected: str) -> None:
    assert asr.model_identifier(model) == expected


def test_model_identifier_redacts_existing_directory_after_expanduser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "converted-model"
    model.mkdir()
    monkeypatch.chdir(tmp_path)
    tilde_path = f"~/{os.path.relpath(model, Path.home())}"
    for value in (str(model), tilde_path, "converted-model"):
        assert asr.model_identifier(value) == "local:converted-model"


@pytest.mark.parametrize(
    ("allow_download", "has_tokenizer"), [(False, False), (True, False), (False, True)]
)
def test_local_model_tokenizer_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    allow_download: bool,
    has_tokenizer: bool,
) -> None:
    model = tmp_path / "converted-model"
    model.mkdir()
    if has_tokenizer:
        (model / "tokenizer.json").write_text("{}")
    constructed: list[bool] = []

    class FakeBackend:
        def __init__(self, model: str, *, allow_model_download: bool = False) -> None:
            constructed.append(allow_model_download)

        def transcribe(self, audio: Path, *, language: str | None = None) -> ASRResult:
            return ASRResult(
                (ASRSegment(0.0, 1.0, "A local test."),), "en", 1.0, str(model), "test", "int8"
            )

    monkeypatch.setattr(asr, "create_asr_backend", FakeBackend)
    media = _media(tmp_path / "inbox")
    if not allow_download and not has_tokenizer:
        with pytest.raises(IngestError, match=r"tokenizer\.json.*--allow-model-download"):
            ingest_local(media, tmp_path / "bundles", asr_model=str(model))
        assert constructed == []
    else:
        result = ingest_local(
            media, tmp_path / "bundles", asr_model=str(model), allow_model_download=allow_download
        )
        assert _metadata(result.bundle_dir)["language"] == "en"
        assert constructed == [allow_download]


@pytest.mark.parametrize("queued", [False, True])
@pytest.mark.parametrize("requested_language", [None, "en", "fr"])
def test_asr_keeps_effective_language_and_warns_on_override_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    queued: bool,
    requested_language: str | None,
) -> None:
    requested_languages: list[str | None] = []

    class EnglishOnlyBackend:
        def __init__(self, model: str, *, allow_model_download: bool = False) -> None:
            self.model = model

        def transcribe(self, audio: Path, *, language: str | None = None) -> ASRResult:
            requested_languages.append(language)
            return ASRResult(
                (ASRSegment(0.0, 1.0, "An English test."),), "en", 1.0, self.model, "test", "int8"
            )

    monkeypatch.setattr(asr, "create_asr_backend", EnglishOnlyBackend)
    media = _media(tmp_path / "inbox")
    state_path = tmp_path / "state.sqlite"
    if queued:
        with open_state(state_path) as state:
            source = state.add_local_folder_source("talks", media.parent)
            queue = state.scan_source(source.id).queued[0]
            state.approve_queue_item(queue.id)
        args = ["queue", "ingest", queue.id]
    else:
        args = ["ingest", str(media)]
    assert (
        cli.main(
            [
                *args,
                "--output",
                str(tmp_path / "bundles"),
                "--state",
                str(state_path),
                "--asr-model",
                "tiny.en",
                *(["--language", requested_language] if requested_language is not None else []),
            ]
        )
        == 0
    )
    bundle = next((tmp_path / "bundles").iterdir())
    metadata = _metadata(bundle)
    assert metadata["language"] == metadata["backend"]["detected_language"] == "en"
    warning_lines = capsys.readouterr().err.splitlines()
    if requested_language == "fr":
        assert len(warning_lines) == 1
        assert "warning" in warning_lines[0].lower()
        assert "'fr'" in warning_lines[0] and "'en'" in warning_lines[0]
    else:
        assert warning_lines == []
    assert requested_languages == [requested_language]


@pytest.mark.parametrize("method", ["sidecar", "command"])
def test_explicit_language_still_overrides_sidecar_and_command(tmp_path: Path, method: str) -> None:
    media = _media(tmp_path, "timed_rolling.en.vtt")
    command: str | None = None
    if method == "command":
        transcriber = tmp_path / "transcriber.py"
        transcriber.write_text('print(\'{"text": "Command transcript."}\')\n')
        command = f'"{sys.executable}" "{transcriber}"'
    result = ingest_local(media, tmp_path / "bundles", transcriber_command=command, language="fr")
    assert _metadata(result.bundle_dir)["language"] == "fr"
