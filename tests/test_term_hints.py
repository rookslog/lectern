"""Named-term pass-through contracts, using only invented text and fake ASR."""

from __future__ import annotations

import hashlib
import json
import shlex
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from lectern import asr, cli
from lectern.agent_packet import talk_packet
from lectern.automation import open_state
from lectern.bundle import TranscriptMetadataDocument
from lectern.ingest import (
    _sidecar_transcript,  # pyright: ignore[reportPrivateUsage]
    _transcribe_with_asr,  # pyright: ignore[reportPrivateUsage]
    _transcribe_with_local_command,  # pyright: ignore[reportPrivateUsage]
    ingest_local,
    plan_local_bundle_id,
)

FIXTURE = Path(__file__).parent / "fixtures/synthetic_talk.wav"


def _media(tmp_path: Path, sidecar: str | None = None) -> Path:
    media = tmp_path / "talk.wav"
    media.write_bytes(FIXTURE.read_bytes())
    if sidecar is not None:
        media.with_suffix(sidecar).write_text(
            "Synthetic text.\n"
            if sidecar == ".transcript.txt"
            else "WEBVTT\n\n00:00.000 --> 00:01.000\nSynthetic text.\n",
            encoding="utf-8",
        )
    return media


def _metadata(bundle: Path) -> dict[str, Any]:
    raw = (bundle / "transcript/metadata.json").read_text(encoding="utf-8")
    TranscriptMetadataDocument.model_validate_json(raw)
    return json.loads(raw)


@pytest.fixture
def whisper_calls(monkeypatch: pytest.MonkeyPatch) -> list[str | None]:
    calls: list[str | None] = []

    class FakeWhisperModel:
        def __init__(
            self, model: str, *, device: str, compute_type: str, local_files_only: bool
        ) -> None:
            assert local_files_only

        def transcribe(
            self, audio: object, *, language: str | None = None, hotwords: str | None = None
        ) -> tuple[list[SimpleNamespace], SimpleNamespace]:
            calls.append(hotwords)
            text = "Schmitt" if hotwords and "Schmitt" in hotwords else "Schmidt"
            return [SimpleNamespace(start=0.0, end=1.0, text=text)], SimpleNamespace(
                language="de", language_probability=1.0
            )

    module = ModuleType("faster_whisper")

    def samples(audio: Path) -> object:
        return object()

    def library_version(name: str) -> str:
        return "synthetic-1"

    monkeypatch.setattr(module, "WhisperModel", FakeWhisperModel, raising=False)
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    monkeypatch.setattr(asr, "pcm16_mono_samples", samples)
    monkeypatch.setattr(asr, "version", library_version)
    monkeypatch.setattr(asr, "create_asr_backend", asr.FasterWhisperBackend)
    return calls


def test_hotwords_reach_whisper_and_change_synthetic_transcript(
    tmp_path: Path, whisper_calls: list[str | None]
) -> None:
    media = _media(tmp_path)
    baseline = ingest_local(media, tmp_path / "bundles")
    # Identical fake output with Kelsen alone must still have a different identity.
    kelsen = ingest_local(media, tmp_path / "bundles", terms=["Kelsen"])
    hinted = ingest_local(media, tmp_path / "bundles", terms=["Schmitt", "Kelsen"])
    assert whisper_calls == [None, "Kelsen", "Schmitt Kelsen"]
    assert (baseline.bundle_dir / "transcript/transcript.md").read_text() == "Schmidt\n"
    assert (kelsen.bundle_dir / "transcript/transcript.md").read_text() == "Schmidt\n"
    assert (hinted.bundle_dir / "transcript/transcript.md").read_text() == "Schmitt\n"
    assert len({r.manifest.bundle_id for r in (baseline, kelsen, hinted)}) == 3
    backend = _metadata(hinted.bundle_dir)["backend"]
    assert backend["terms"] == ["Schmitt", "Kelsen"]
    assert backend["terms_applied"] is True


@pytest.mark.parametrize("queued", [False, True])
def test_cli_combines_terms_in_order_and_exports_them(
    tmp_path: Path,
    whisper_calls: list[str | None],
    capsys: pytest.CaptureFixture[str],
    queued: bool,
) -> None:
    media = _media(tmp_path)
    # Terms must not be inferred from the media's metadata.
    media.with_suffix(".info.json").write_text('{"title":"NotAnAutomaticTerm"}')
    terms_file = tmp_path / "terms.txt"
    terms_file.write_text("# Names\n\n Schmitt \nKelsen\nKelsen\nweimar\n", encoding="utf-8")
    state_path = tmp_path / "state.sqlite"
    if queued:
        with open_state(state_path) as state:
            source = state.add_local_folder_source("talks", tmp_path)
            item = state.scan_source(source.id).queued[0]
            state.approve_queue_item(item.id)
        args = ["queue", "ingest", item.id]
    else:
        args = ["ingest", str(media)]
    args += [
        "--output",
        str(tmp_path / "bundles"),
        "--state",
        str(state_path),
        "--terms",
        str(terms_file),
        "--term",
        " Schmitt ",
        "--term",
        "Weimar",
        "--term",
        "Schmitt",
    ]
    assert cli.main(args) == 0
    capsys.readouterr()
    assert whisper_calls == ["Schmitt Weimar Kelsen weimar"]
    bundle = next((tmp_path / "bundles").iterdir())
    terms = ["Schmitt", "Weimar", "Kelsen", "weimar"]
    assert _metadata(bundle)["backend"]["terms"] == terms
    with open_state(state_path) as state:
        packet = talk_packet(state, bundle.name)
    assert packet["layers"]["transcript"]["terms"] == terms
    assert cli.main(["library", "show", bundle.name, "--state", str(state_path)]) == 0
    assert "Terms: 4 (passed to faster_whisper)" in capsys.readouterr().out


def test_command_receives_newline_terms_without_argv_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media = _media(tmp_path)
    receipt = tmp_path / "receipt.json"
    script = tmp_path / "transcriber.py"
    script.write_text(
        "import json, os, sys\nfrom pathlib import Path\n"
        f"Path({str(receipt)!r}).write_text(json.dumps("
        "{'terms': os.environ.get('LECTERN_TERMS'), 'args': sys.argv[1:]}))\n"
        'print(\'{"text":"Synthetic text."}\')\n'
    )
    monkeypatch.setenv("LECTERN_TERMS", "Must not leak from parent")
    command = f"{shlex.quote(sys.executable)} {shlex.quote(str(script))} sentinel {{input}}"
    for terms, expected in [(["Schmitt", "Kelsen"], "Schmitt\nKelsen"), ([], None)]:
        result = ingest_local(media, tmp_path / "bundles", transcriber_command=command, terms=terms)
        seen = json.loads(receipt.read_text())
        assert seen["terms"] == expected
        assert seen["args"][0] == "sentinel"
        assert len(seen["args"]) == 2
        assert Path(seen["args"][1]).name == "audio.wav"
        backend = _metadata(result.bundle_dir)["backend"]
        if terms:
            assert backend["terms"] == terms
            assert backend["terms_applied"] is True
        else:
            assert "terms" not in backend
            assert "terms_applied" not in backend


@pytest.mark.parametrize("suffix", [".vtt", ".transcript.txt"])
def test_sidecar_records_unapplied_terms_and_warns_once(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], suffix: str
) -> None:
    media = _media(tmp_path, suffix)
    baseline_id = plan_local_bundle_id(media)
    planned = plan_local_bundle_id(media, terms=["Schmitt"])
    assert capsys.readouterr().err == ""
    result = ingest_local(media, tmp_path / "bundles", terms=["Schmitt"])
    assert result.manifest.bundle_id == planned != baseline_id
    backend = _metadata(result.bundle_dir)["backend"]
    assert backend["terms"] == ["Schmitt"]
    assert backend["terms_applied"] is False
    warnings = capsys.readouterr().err.splitlines()
    assert len(warnings) == 1
    assert "warning" in warnings[0] and "terms" in warnings[0] and "sidecar" in warnings[0]


@pytest.mark.parametrize("queued", [False, True])
@pytest.mark.parametrize(
    ("terms", "position"),
    [
        (["x" * 65], 1),
        (["valid", "bad\x00name"], 2),
        (["bad\x85name"], 1),
        (["bad\nname"], 1),
        (["\tbad"], 1),
        (["   "], 1),
        ([f"name{i}" for i in range(51)], 51),
    ],
)
def test_invalid_terms_are_usage_errors_before_ingest(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    queued: bool,
    terms: list[str],
    position: int,
) -> None:
    args = ["queue", "ingest", "unused"] if queued else ["ingest", "missing.wav"]
    args += ["--state", str(tmp_path / "state.sqlite")]
    for term in terms:
        args += ["--term", term]
    assert cli.main(args) == 2
    error = capsys.readouterr().err.lower()
    assert f"term {position}" in error
    assert not (tmp_path / "state.sqlite").exists()


def test_file_validation_uses_combined_position(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    terms_file = tmp_path / "terms.txt"
    terms_file.write_text("\n# comment\nKelsen\nbad\x01name\n")
    assert (
        cli.main(
            [
                "ingest",
                "missing.wav",
                "--term",
                "Schmitt",
                "--terms",
                str(terms_file),
            ]
        )
        == 2
    )
    assert "term 3" in capsys.readouterr().err.lower()


def test_blank_file_lines_and_duplicates_do_not_use_term_limit(
    tmp_path: Path, whisper_calls: list[str | None]
) -> None:
    media = _media(tmp_path)
    terms_file = tmp_path / "terms.txt"
    terms_file.write_text("\n# comment\n" + "Schmitt\n" * 60, encoding="utf-8")
    assert (
        cli.main(
            [
                "ingest",
                str(media),
                "--terms",
                str(terms_file),
                "--output",
                str(tmp_path / "bundles"),
                "--state",
                str(tmp_path / "state.sqlite"),
            ]
        )
        == 0
    )
    assert whisper_calls == ["Schmitt"]


def test_maximum_terms_and_unicode_are_passed_exactly(
    tmp_path: Path, whisper_calls: list[str | None]
) -> None:
    terms = ["é" * 64, "Carl Schmitt", *[f"name{i}" for i in range(48)]]
    result = ingest_local(_media(tmp_path), tmp_path / "bundles", terms=terms)
    assert _metadata(result.bundle_dir)["backend"]["terms"] == terms
    assert whisper_calls == [" ".join(terms)]


def test_metadata_does_not_supply_automatic_terms(
    tmp_path: Path, whisper_calls: list[str | None]
) -> None:
    media = _media(tmp_path)
    media.with_suffix(".info.json").write_text('{"title":"Schmitt Kelsen"}')
    result = ingest_local(media, tmp_path / "bundles")
    assert whisper_calls == [None]
    assert "terms" not in _metadata(result.bundle_dir)["backend"]


def test_transcript_identity_records_the_exact_passed_terms(
    whisper_calls: list[str | None],
) -> None:
    transcript = _transcribe_with_asr("small", Path("audio.wav"), None, terms=["Schmitt", "Kelsen"])
    assert transcript.identity["terms"] == ["Schmitt", "Kelsen"]
    assert transcript.identity["terms_applied"] is True
    assert transcript.backend["terms"] == ["Schmitt", "Kelsen"]


@pytest.mark.parametrize("payload", [None, b"\xff"])
def test_unreadable_terms_files_are_usage_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], payload: bytes | None
) -> None:
    terms_file = tmp_path / "terms.txt"
    if payload is not None:
        terms_file.write_bytes(payload)
    assert cli.main(["ingest", "missing.wav", "--terms", str(terms_file)]) == 2
    assert "terms file" in capsys.readouterr().err


@pytest.mark.parametrize("kind", ["sidecar", "asr", "command"])
def test_no_terms_preserves_base_identity_and_provenance_bytes(
    whisper_calls: list[str | None], monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    # SHA-256 literals computed on 96d3471 before production edits. Identity uses
    # the existing canonical bytes; provenance uses the JSON writer's exact format.
    digests = {
        "sidecar": (
            "b83a3aa3f1ab7aa9f6cc5ac28cbc381c41dde51c44013fe2cd1ea51759504710",
            "351e936fd80bb0fd9a2ee476ca69c2b55aef355c4f425d668cc23f411e54e133",
        ),
        "asr": (
            "98c2925b85e85739a9debcd3c6df6eccbc30bc64447f7bfc099185d406718ed2",
            "cf04dfb24214d2148ce8665112cadb03885374e47a9f73bf8a0a37a264580c71",
        ),
        "command": (
            "a3781b19f361cc9bf4c8e3fb0460852c295c0920275c80c8c3964aba0b7f3585",
            "e5a601d09573f27eeea529da1fad4a65669b8db3b8f82712838d237316eb6c9d",
        ),
    }
    if kind == "sidecar":
        transcript = _sidecar_transcript(Path("talk.wav"), b"Synthetic text.\n")
    elif kind == "asr":
        transcript = _transcribe_with_asr("small", Path("audio.wav"), None)
    else:

        def command_result(*args: Any, **kwargs: Any) -> SimpleNamespace:
            return SimpleNamespace(stdout=b'{"text":"Synthetic text."}', stderr=b"", returncode=0)

        monkeypatch.setattr("lectern.ingest.subprocess.run", command_result)
        transcript = _transcribe_with_local_command("synthetic-asr", Path("audio.wav"), 3.0)
    assert transcript.identity_component == digests[kind][0]
    provenance = (json.dumps(transcript.backend, indent=2) + "\n").encode("utf-8")
    assert hashlib.sha256(provenance).hexdigest() == digests[kind][1]
