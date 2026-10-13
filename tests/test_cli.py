"""CLI scaffold behavior for M0+."""

import json
import sys
from pathlib import Path

from pytest import CaptureFixture, MonkeyPatch

from lectern import __version__, cli
from lectern.bundle import Manifest, StageName, StageState, export_json_schema

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
SYNTHETIC_TALK = FIXTURE_DIR / "synthetic_talk.wav"
SYNTHETIC_TRANSCRIPT = FIXTURE_DIR / "synthetic_talk.transcript.txt"


def copy_fixture(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    media = directory / "synthetic_talk.wav"
    media.write_bytes(SYNTHETIC_TALK.read_bytes())
    media.with_suffix(".transcript.txt").write_text(
        SYNTHETIC_TRANSCRIPT.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    return media


def copy_media_without_sidecar(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    media = directory / "local_talk.wav"
    media.write_bytes(SYNTHETIC_TALK.read_bytes())
    return media


def test_version_flag(capsys: CaptureFixture[str]) -> None:
    assert cli.main(["--version"]) == 0
    captured = capsys.readouterr()
    assert captured.out == f"lectern {__version__}\n"
    assert captured.err == ""


def test_doctor_reports_required_local_tools(
    capsys: CaptureFixture[str], monkeypatch: MonkeyPatch
) -> None:
    def fake_which(name: str) -> str | None:
        if name == "ffmpeg":
            return "/usr/bin/ffmpeg"
        return None

    monkeypatch.setattr(cli.shutil, "which", fake_which)

    assert cli.main(["doctor"]) == 0
    captured = capsys.readouterr()
    assert "python: OK" in captured.out
    assert "ffmpeg: OK (/usr/bin/ffmpeg)" in captured.out
    assert captured.err == ""


def test_doctor_fails_when_required_tool_is_missing(
    capsys: CaptureFixture[str], monkeypatch: MonkeyPatch
) -> None:
    def fake_which(name: str) -> str | None:
        return None

    monkeypatch.setattr(cli.shutil, "which", fake_which)

    assert cli.main(["doctor"]) == 1
    captured = capsys.readouterr()
    assert "ffmpeg: MISSING" in captured.out
    assert captured.err == ""


def test_doctor_does_not_create_state_store(
    tmp_path: Path,
    capsys: CaptureFixture[str],
    monkeypatch: MonkeyPatch,
) -> None:
    def fake_which(name: str) -> str | None:
        if name == "ffmpeg":
            return "/usr/bin/ffmpeg"
        return None

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli.shutil, "which", fake_which)

    assert cli.main(["doctor"]) == 0
    captured = capsys.readouterr()
    assert "state: OK (.lectern/state.sqlite)" in captured.out
    assert not (tmp_path / ".lectern").exists()


def test_doctor_reports_existing_read_only_state_store(
    tmp_path: Path,
    capsys: CaptureFixture[str],
    monkeypatch: MonkeyPatch,
) -> None:
    def fake_which(name: str) -> str | None:
        if name == "ffmpeg":
            return "/usr/bin/ffmpeg"
        return None

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli.shutil, "which", fake_which)
    state_dir = tmp_path / ".lectern"
    state_dir.mkdir()
    state = state_dir / "state.sqlite"
    state.write_bytes(b"")
    state.chmod(0o444)

    try:
        assert cli.main(["doctor"]) == 1
        captured = capsys.readouterr()
        assert "state: ERROR" in captured.out
    finally:
        state.chmod(0o644)


def test_doctor_rejects_state_parent_that_is_a_file(
    tmp_path: Path,
    capsys: CaptureFixture[str],
    monkeypatch: MonkeyPatch,
) -> None:
    def fake_which(name: str) -> str | None:
        if name == "ffmpeg":
            return "/usr/bin/ffmpeg"
        return None

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli.shutil, "which", fake_which)
    (tmp_path / ".lectern").write_text("not a directory", encoding="utf-8")

    assert cli.main(["doctor"]) == 1
    captured = capsys.readouterr()
    assert "state: ERROR" in captured.out


def test_schema_export_writes_manifest_schema(tmp_path: Path) -> None:
    output = tmp_path / "manifest.schema.json"

    assert cli.main(["schema", "export", "--output", str(output)]) == 0

    assert output.read_text() == export_json_schema()


def test_ingest_command_writes_bundle(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    source = Path("tests/fixtures/synthetic_talk.wav")

    assert (
        cli.main(
            [
                "ingest",
                str(source),
                "--output",
                str(tmp_path),
                "--state",
                str(tmp_path / "state.sqlite"),
            ]
        )
        == 0
    )

    captured = capsys.readouterr()
    assert captured.err == ""
    bundle_dir = Path(captured.out.strip())
    assert bundle_dir.is_dir()
    manifest = Manifest.load(bundle_dir)
    assert manifest.stages[StageName.TRANSCRIBE].state is StageState.DONE


def test_ingest_command_accepts_local_transcriber_command_json(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    source = copy_media_without_sidecar(tmp_path / "source")
    transcriber = _write_transcriber_script(
        tmp_path / "transcriber.py",
        json.dumps({"segments": [{"start_s": 3.0, "text": "CLI transcript."}]}),
    )

    assert (
        cli.main(
            [
                "ingest",
                str(source),
                "--output",
                str(tmp_path / "bundles"),
                "--state",
                str(tmp_path / "state.sqlite"),
                "--transcriber-command",
                f"{sys.executable} {transcriber}",
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    bundle_dir = Path(payload["bundle_dir"])
    source_json = json.loads((bundle_dir / "source.json").read_text(encoding="utf-8"))
    assert payload["bundle_id"]
    assert source_json["transcript"]["method"] == "local_command_json"
    overview = (bundle_dir / "analysis" / "summary.md").read_text(encoding="utf-8")
    assert "Transcript method: local_command_json" in overview
    assert "CLI transcript." not in overview


def test_ingest_usage_error_mentions_transcriber_command(
    capsys: CaptureFixture[str],
) -> None:
    assert cli.main(["ingest", "file.wav", "--bad-flag"]) == 2

    captured = capsys.readouterr()
    assert "--transcriber-command COMMAND" in captured.err


def test_source_queue_library_commands_emit_json(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    source_dir = tmp_path / "source"
    copy_fixture(source_dir)
    state = tmp_path / "state.sqlite"
    bundles = tmp_path / "bundles"

    assert (
        cli.main(
            [
                "sources",
                "add-folder",
                "talks",
                str(source_dir),
                "--state",
                str(state),
                "--json",
            ]
        )
        == 0
    )
    source_payload = json.loads(capsys.readouterr().out)
    assert source_payload["name"] == "talks"
    source_id = source_payload["id"]

    assert cli.main(["sources", "scan", source_id, "--state", str(state), "--json"]) == 0
    scan_payload = json.loads(capsys.readouterr().out)
    assert scan_payload["counts"]["added"] == 1
    queue_item_id = scan_payload["queued"][0]["id"]

    assert (
        cli.main(
            [
                "queue",
                "list",
                "--queue-state",
                "discovered",
                "--state",
                str(state),
                "--json",
            ]
        )
        == 0
    )
    queue_payload = json.loads(capsys.readouterr().out)
    assert [item["id"] for item in queue_payload["queue"]] == [queue_item_id]

    assert cli.main(["queue", "approve", queue_item_id, "--state", str(state), "--json"]) == 0
    approved_payload = json.loads(capsys.readouterr().out)
    assert approved_payload["state"] == "approved"

    assert (
        cli.main(
            [
                "queue",
                "ingest",
                queue_item_id,
                "--output",
                str(bundles),
                "--state",
                str(state),
                "--json",
            ]
        )
        == 0
    )
    ingest_payload = json.loads(capsys.readouterr().out)
    assert ingest_payload["queue_item_id"] == queue_item_id

    assert cli.main(["library", "list", "--state", str(state), "--json"]) == 0
    library_payload = json.loads(capsys.readouterr().out)
    assert [bundle["bundle_id"] for bundle in library_payload["bundles"]] == [
        ingest_payload["bundle_id"]
    ]


def test_cli_reports_usage_errors(capsys: CaptureFixture[str]) -> None:
    assert cli.main(["sources", "scan"]) == 2
    captured = capsys.readouterr()
    assert "usage: lectern sources" in captured.err


def test_sources_scan_rejects_youtube_only_flags_for_local_folder(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    source_dir = tmp_path / "source"
    copy_fixture(source_dir)
    state = tmp_path / "state.sqlite"

    assert (
        cli.main(
            [
                "sources",
                "add-folder",
                "talks",
                str(source_dir),
                "--state",
                str(state),
                "--json",
            ]
        )
        == 0
    )
    capsys.readouterr()

    assert cli.main(["sources", "scan", "talks", "--max-pages", "1", "--state", str(state)]) == 2
    assert "--max-pages" in capsys.readouterr().err

    assert (
        cli.main(["sources", "scan", "talks", "--api-key-env", "OTHER_KEY", "--state", str(state)])
        == 2
    )
    assert "--api-key-env" in capsys.readouterr().err

    assert cli.main(["sources", "scan", "talks", "--state", str(state), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["counts"]["added"] == 1


def test_cli_reports_unknown_source_as_domain_error(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    assert cli.main(["sources", "scan", "missing", "--state", str(tmp_path / "state.sqlite")]) == 3
    captured = capsys.readouterr()
    assert "source not found" in captured.err


def test_cli_rejects_unapproved_queue_ingest(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    source_dir = tmp_path / "source"
    copy_fixture(source_dir)
    state = tmp_path / "state.sqlite"
    bundles = tmp_path / "bundles"

    assert (
        cli.main(
            [
                "sources",
                "add-folder",
                "talks",
                str(source_dir),
                "--state",
                str(state),
                "--json",
            ]
        )
        == 0
    )
    source_id = json.loads(capsys.readouterr().out)["id"]
    assert cli.main(["sources", "scan", source_id, "--state", str(state), "--json"]) == 0
    queue_item_id = json.loads(capsys.readouterr().out)["queued"][0]["id"]

    assert (
        cli.main(
            [
                "queue",
                "ingest",
                queue_item_id,
                "--output",
                str(bundles),
                "--state",
                str(state),
            ]
        )
        == 3
    )
    captured = capsys.readouterr()
    assert "requires explicit approval" in captured.err


def test_cli_reports_corrupt_state_store_cleanly(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    state = tmp_path / "state.sqlite"
    state.write_text("not sqlite", encoding="utf-8")

    assert cli.main(["sources", "list", "--state", str(state)]) == 3
    captured = capsys.readouterr()
    assert "state database error" in captured.err
    assert "Traceback" not in captured.err


def _write_transcriber_script(path: Path, stdout: str, *, exit_code: int = 0) -> Path:
    path.write_text(
        f"import sys\nsys.stdout.write({stdout!r})\nraise SystemExit({exit_code})\n",
        encoding="utf-8",
    )
    return path
