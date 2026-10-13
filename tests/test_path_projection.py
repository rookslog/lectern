"""No outward-facing command may emit a filesystem path.

This is a property of the *program*, not of any one command, and stating it that
way is the whole point of the module. An earlier version of this design asserted
the property over the response types of two new commands and called the boundary
closed; it was not, because the commands that already existed kept returning
absolute paths, and nothing was checking them.

The rule enforced here is about data rather than columns: a value is
path-bearing if it *can* contain a filesystem path, including free text that
interpolates one. That phrasing is what reaches `last_error`, which carries a
path only when an operation fails and is therefore invisible to every fixture
that succeeds.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest
from pytest import CaptureFixture

from lectern import cli
from lectern.automation import open_state
from lectern.ingest import ingest_local
from lectern.records import PATH_REDACTED, redact_paths
from lectern.sources import local

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
SYNTHETIC_TALK = FIXTURE_DIR / "synthetic_talk.wav"
SYNTHETIC_TRANSCRIPT = FIXTURE_DIR / "synthetic_talk.transcript.txt"

# Every command that returns data to a caller. A command absent from this tuple
# is a command nothing checks, which is the failure this module exists to
# prevent -- see `test_every_outward_command_is_covered`.
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
    "migrate",
)


def _watched_folder(root: Path) -> Path:
    """A source folder whose absolute path is distinctive enough to grep for."""

    folder = root / "Private Therapy" / "recordings"
    folder.mkdir(parents=True, exist_ok=True)
    media = folder / "synthetic_talk.wav"
    media.write_bytes(SYNTHETIC_TALK.read_bytes())
    media.with_suffix(".transcript.txt").write_text(
        SYNTHETIC_TRANSCRIPT.read_text(encoding="utf-8"), encoding="utf-8"
    )
    return folder


def _current_private_bundle(tmp_path: Path) -> tuple[Path, Path]:
    folder = _watched_folder(tmp_path)
    media = folder / "synthetic_talk.wav"
    bundle = ingest_local(media, folder.parent / "bundles").bundle_dir
    return folder, bundle


def _leaky_substrings(folder: Path) -> tuple[str, ...]:
    """Fragments whose appearance in output would constitute a leak.

    The absolute path is the obvious one. The parent directory names are
    included because a partial path is still the user's filesystem: on macOS the
    home directory carries the account name, and "Private Therapy" is the kind
    of thing a directory name discloses on its own.
    """

    return (str(folder), str(folder.parent), folder.parent.name)


@pytest.fixture
def registered(tmp_path: Path) -> Iterator[tuple[Path, Path, Path]]:
    """A state store with one registered, scanned source folder."""

    folder = _watched_folder(tmp_path)
    state = tmp_path / "state.sqlite"
    assert cli.main(["sources", "add-folder", "talks", str(folder), "--state", str(state)]) == 0
    assert cli.main(["sources", "scan", "talks", "--state", str(state)]) == 0
    yield tmp_path, folder, state


def _assert_no_path(captured: str, folder: Path, command: str) -> None:
    for fragment in _leaky_substrings(folder):
        assert fragment not in captured, f"`{command}` leaked {fragment!r}"


@pytest.mark.parametrize(
    "path",
    [
        pytest.param(r"\Users\alice\Private\session.wav", id="root-relative"),
        pytest.param(r"C:\Users\alice\Private\session.wav", id="drive-backslash"),
        pytest.param("C:/Users/alice/Private/session.wav", id="drive-forward-slash"),
        pytest.param(r"\\server\share\alice\Private\session.wav", id="unc"),
    ],
)
def test_windows_rooted_paths_are_completely_redacted(path: str) -> None:
    assert redact_paths(f"failed at {path}") == f"failed at {PATH_REDACTED}"


@pytest.mark.parametrize("json_output", [False, True], ids=["plain", "json"])
def test_sources_list_emits_no_filesystem_path(
    registered: tuple[Path, Path, Path], capsys: CaptureFixture[str], json_output: bool
) -> None:
    # Both renderings are checked because both print `root_path` today: the
    # plain branch formats it into a tab-separated line, so a JSON-only
    # assertion would pass while the default output still leaked.
    _, folder, state = registered
    args = ["sources", "list", "--state", str(state)]
    if json_output:
        args.append("--json")
    assert cli.main(args) == 0
    _assert_no_path(capsys.readouterr().out, folder, "sources list")


@pytest.mark.parametrize("json_output", [False, True], ids=["plain", "json"])
def test_sources_scan_emits_no_filesystem_path(
    registered: tuple[Path, Path, Path], capsys: CaptureFixture[str], json_output: bool
) -> None:
    _, folder, state = registered
    args = ["sources", "scan", "talks", "--state", str(state)]
    if json_output:
        args.append("--json")
    assert cli.main(args) == 0
    _assert_no_path(capsys.readouterr().out, folder, "sources scan")


def test_queue_list_emits_no_filesystem_path(
    registered: tuple[Path, Path, Path], capsys: CaptureFixture[str]
) -> None:
    _, folder, state = registered
    assert cli.main(["queue", "list", "--state", str(state), "--json"]) == 0
    _assert_no_path(capsys.readouterr().out, folder, "queue list")


def test_failed_queue_item_error_is_projected(
    registered: tuple[Path, Path, Path], capsys: CaptureFixture[str]
) -> None:
    """The leak that only exists when something goes wrong.

    A successful ingest never populates `last_error`, so a suite built entirely
    from passing fixtures certifies the happy path and reads as though it
    certified the program. The media is deleted between approval and ingest so
    the `OSError` message -- which interpolates the absolute filename -- is what
    gets persisted and served back.
    """

    tmp_path, folder, state = registered
    assert cli.main(["queue", "list", "--state", str(state), "--json"]) == 0
    queue = json.loads(capsys.readouterr().out)["queue"]
    assert queue, "fixture produced no queue item to fail"
    item_id = queue[0]["id"]

    assert cli.main(["queue", "approve", item_id, "--state", str(state)]) == 0
    (folder / "synthetic_talk.wav").unlink()
    cli.main(["queue", "ingest", item_id, "--state", str(state), "--output", str(tmp_path / "out")])
    capsys.readouterr()

    assert cli.main(["queue", "show", item_id, "--state", str(state), "--json"]) == 0
    output = capsys.readouterr().out
    assert json.loads(output)["last_error"], "fixture did not record a failure"
    _assert_no_path(output, folder, "queue show")


def _ingest_one(tmp_path: Path, state: Path, capsys: CaptureFixture[str]) -> str:
    assert cli.main(["queue", "list", "--state", str(state), "--json"]) == 0
    item_id = json.loads(capsys.readouterr().out)["queue"][0]["id"]
    assert cli.main(["queue", "approve", item_id, "--state", str(state)]) == 0
    assert (
        cli.main(
            ["queue", "ingest", item_id, "--state", str(state), "--output", str(tmp_path / "out")]
        )
        == 0
    )
    capsys.readouterr()
    assert cli.main(["library", "list", "--state", str(state), "--json"]) == 0
    return capsys.readouterr().out


def test_library_list_emits_no_filesystem_path(
    registered: tuple[Path, Path, Path], capsys: CaptureFixture[str]
) -> None:
    tmp_path, folder, state = registered
    _assert_no_path(_ingest_one(tmp_path, state, capsys), folder, "library list")


def test_library_show_emits_no_filesystem_path(
    registered: tuple[Path, Path, Path], capsys: CaptureFixture[str]
) -> None:
    """Green once phase 2 lands: the embedded manifest no longer carries a path.

    Held as xfail(strict=True) through phase 1 because the leak was inside an
    artifact this command reads rather than a field it serializes, so the
    projection layer could not reach it. The strict marker is what forced its
    own removal the moment content identity landed.
    """

    tmp_path, folder, state = registered
    listing = _ingest_one(tmp_path, state, capsys)
    bundle_id = json.loads(listing)["bundles"][0]["bundle_id"]
    assert cli.main(["library", "show", bundle_id, "--state", str(state), "--json"]) == 0
    _assert_no_path(capsys.readouterr().out, folder, "library show")


def test_library_show_error_emits_no_filesystem_path(
    registered: tuple[Path, Path, Path], capsys: CaptureFixture[str]
) -> None:
    tmp_path, folder, state_path = registered
    listing = _ingest_one(tmp_path, state_path, capsys)
    bundle_id = json.loads(listing)["bundles"][0]["bundle_id"]
    with open_state(state_path) as state:
        bundle = state.get_library_bundle(bundle_id)
    (Path(bundle.bundle_path) / "manifest.json").unlink()
    assert cli.main(["library", "show", bundle_id, "--state", str(state_path), "--json"]) == 1
    _assert_no_path(capsys.readouterr().err, folder, "library show error")


def test_info_metadata_paths_are_projected_in_library_output(
    tmp_path: Path, capsys: CaptureFixture[str]
) -> None:
    folder = _watched_folder(tmp_path)
    media = folder / "synthetic_talk.wav"
    private_path = str(folder / "session.wav")
    media.with_suffix(".info.json").write_text(
        json.dumps(
            {
                "title": f"Talk at {private_path}",
                "channel": private_path,
                "extractor_key": private_path,
                "id": private_path,
                "webpage_url": private_path,
                "playlist_id": private_path,
                "playlist_title": private_path,
                "language": private_path,
            }
        )
    )
    state_path = tmp_path / "state.sqlite"
    with open_state(state_path) as state:
        result = state.ingest_one_shot(media, tmp_path / "bundles")
    before = (result.bundle_dir / "manifest.json").read_bytes()
    bundle_id = result.manifest.bundle_id
    commands = (
        ["list"],
        ["list", "--playlist"],
        ["show", bundle_id],
        ["search", "inspectable knowledge"],
    )
    for command in commands:
        for json_output in (False, True):
            args = ["library", *command, "--state", str(state_path)]
            if json_output:
                args.append("--json")
            assert cli.main(args) == 0
            output = capsys.readouterr().out
            _assert_no_path(output, folder, f"library {command[0]}")
            if json_output:
                payload = json.loads(output)
                if command[0] == "list":
                    source = payload["bundles"][0]
                elif command[0] == "show":
                    source = payload["manifest"]["source"]
                    assert payload["channel"] == payload["url"] == PATH_REDACTED
                else:
                    source = payload["results"][0]
                assert source["title"] == f"Talk at {PATH_REDACTED}"
                if command[0] != "search":
                    assert all(
                        value == PATH_REDACTED
                        for value in source["external"].values()
                        if value is not None
                    )
    assert (result.bundle_dir / "manifest.json").read_bytes() == before
    assert result.manifest.source.title == f"Talk at {private_path}"


def test_library_show_contains_a_recursively_nested_manifest(
    registered: tuple[Path, Path, Path], capsys: CaptureFixture[str]
) -> None:
    tmp_path, _, state_path = registered
    listing = _ingest_one(tmp_path, state_path, capsys)
    bundle_id = json.loads(listing)["bundles"][0]["bundle_id"]
    with open_state(state_path) as state:
        bundle = state.get_library_bundle(bundle_id)
    manifest_path = Path(bundle.bundle_path) / "manifest.json"
    manifest_path.write_text("[" * 100_000 + "0" + "]" * 100_000, encoding="utf-8")

    assert cli.main(["library", "show", bundle_id, "--state", str(state_path), "--json"]) == 3
    error = capsys.readouterr().err
    assert error.startswith("library: ")
    assert "Traceback" not in error


def test_library_show_redacts_stage_error_paths(
    registered: tuple[Path, Path, Path], capsys: CaptureFixture[str]
) -> None:
    tmp_path, folder, state_path = registered
    listing = _ingest_one(tmp_path, state_path, capsys)
    bundle_id = json.loads(listing)["bundles"][0]["bundle_id"]
    with open_state(state_path) as state:
        bundle = state.get_library_bundle(bundle_id)
    manifest_path = Path(bundle.bundle_path) / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["stages"]["transcribe"]["error"] = f"failed {folder}/session.wav"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    assert cli.main(["library", "show", bundle_id, "--state", str(state_path), "--json"]) == 0
    _assert_no_path(capsys.readouterr().out, folder, "library show stage error")


def test_library_show_redacts_windows_root_relative_stage_error_path(
    registered: tuple[Path, Path, Path], capsys: CaptureFixture[str]
) -> None:
    tmp_path, _, state_path = registered
    listing = _ingest_one(tmp_path, state_path, capsys)
    bundle_id = json.loads(listing)["bundles"][0]["bundle_id"]
    with open_state(state_path) as state:
        bundle = state.get_library_bundle(bundle_id)
    manifest_path = Path(bundle.bundle_path) / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    private_path = r"\Users\alice\Private\session.wav"
    payload["stages"]["transcribe"]["error"] = f"failed at {private_path}"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    assert cli.main(["library", "show", bundle_id, "--state", str(state_path), "--json"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["manifest"]["stages"]["transcribe"]["error"] == f"failed at {PATH_REDACTED}"


def test_every_outward_command_is_covered() -> None:
    """The test that keeps this module from decaying into an allowlist.

    Without it, `OUTWARD_COMMANDS` records the commands someone happened to
    think of, and a command added later inherits no assertion at all. Comparing
    against the CLI's own usage text means a new outward command fails here
    until it is either covered or deliberately classified as not outward-facing.
    """

    covered = set(OUTWARD_COMMANDS)
    declared = set(cli.OUTWARD_COMMANDS)
    assert declared == covered, (
        "outward-facing commands and their path-projection coverage disagree; "
        f"uncovered={sorted(declared - covered)} stale={sorted(covered - declared)}"
    )


def test_migrate_success_emits_no_filesystem_path(
    tmp_path: Path, capsys: CaptureFixture[str]
) -> None:
    folder, bundle = _current_private_bundle(tmp_path)
    assert cli.main(["migrate", str(bundle)]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["outcome"] == "already_current"
    _assert_no_path(captured.out + captured.err, folder, "migrate")


def test_migrate_failure_emits_no_filesystem_path(
    tmp_path: Path, capsys: CaptureFixture[str]
) -> None:
    folder, bundle = _current_private_bundle(tmp_path)
    (bundle / "source.json").write_text("{}\n", encoding="utf-8")
    assert cli.main(["migrate", str(bundle)]) == 3
    captured = capsys.readouterr()
    assert "declared artifact integrity" in captured.err
    _assert_no_path(captured.out + captured.err, folder, "migrate")


@pytest.mark.parametrize("json_output", [False, True])
@pytest.mark.parametrize("failure", ["missing-root", "hash-error"])
def test_source_scan_errors_project_stored_paths(
    registered: tuple[Path, Path, Path],
    capsys: CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    json_output: bool,
    failure: str,
) -> None:
    _, folder, state_path = registered
    if failure == "missing-root":
        shutil.rmtree(folder)
        expected_exit = 3
    else:

        def fail_hash(path: Path) -> tuple[str, int]:
            raise OSError(13, "synthetic hashing refusal", str(path))

        monkeypatch.setattr(local, "digest_and_size", fail_hash)
        expected_exit = 1
    capsys.readouterr()
    args = ["sources", "scan", "talks", "--state", str(state_path)]
    if json_output:
        args.append("--json")
    assert cli.main(args) == expected_exit
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("sources: ")
    _assert_no_path(captured.err, folder, "sources scan error")
    assert PATH_REDACTED in captured.err
