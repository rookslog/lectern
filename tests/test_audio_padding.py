"""Bounded synthetic preflight tests: never render an oversized offset."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, BinaryIO, cast

import pytest

from lectern import ingest as ingest_module
from lectern.ingest import IngestError, ingest_local

FIXTURE = Path(__file__).parent / "fixtures/synthetic_talk.wav"


def _probe(pts: object = "0", start: object = "0") -> dict[str, object]:
    return {
        "streams": [{"index": 0, "start_time": "0"}],
        "frames": [{"stream_index": 0, "pts_time": pts}],
        "format": {"start_time": start},
    }


def _run_ingest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    payload: object,
    *,
    rejected: bool,
    failure: str | None = None,
) -> None:
    source = tmp_path / "synthetic.bin"
    source.write_bytes(b"synthetic noncanonical source")
    source.with_suffix(".transcript.txt").write_text("Synthetic transcript.")
    calls: list[str] = []

    def run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        binary = Path(args[0]).name
        calls.append(binary)
        if binary == "ffprobe":
            assert kwargs["timeout"] == 10
            assert args[args.index("-read_intervals") + 1] == "%+#64"
            assert "frame=stream_index,pts_time:stream=index:format=start_time" in args
            if failure == "timeout":
                raise subprocess.TimeoutExpired(args, 10)
            if failure == "oserror":
                raise OSError("synthetic probe failure")
            stdout = cast(BinaryIO, kwargs["stdout"])
            stdout.write(payload if isinstance(payload, bytes) else json.dumps(payload).encode())
            return subprocess.CompletedProcess(args, 1 if failure == "exit" else 0, "", "")
        assert binary == "ffmpeg"
        assert not rejected, "ffmpeg must not run for refused padding"
        shutil.copyfile(FIXTURE, args[-1])
        return subprocess.CompletedProcess(args, 0, "", "")

    def binary_path(name: str) -> str:
        return f"/synthetic/{name}"

    monkeypatch.setattr(ingest_module.shutil, "which", binary_path)
    monkeypatch.setattr(ingest_module.subprocess, "run", run)
    bundles = tmp_path / "bundles"
    if rejected:
        with pytest.raises(IngestError, match="initial audio|ffprobe"):
            ingest_local(source, bundles)
        assert calls == ["ffprobe"]
        assert list(bundles.iterdir()) == []
        assert not list(tmp_path.rglob("*.tmp.wav"))
    else:
        result = ingest_local(source, bundles)
        assert (result.bundle_dir / "media/audio.wav").read_bytes() == FIXTURE.read_bytes()
        assert calls == ["ffprobe", "ffmpeg"]


@pytest.mark.parametrize(
    ("pts", "start"),
    [
        ("0", "0"),
        ("2.4", "0"),
        ("599.999", "0"),
        ("600", "0"),
        ("610", "10"),
        ("590", "-10"),
        ("-2", "0"),
        ("-8", "-10"),
    ],
)
def test_padding_accepts_finite_offsets_through_600_seconds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pts: str, start: str
) -> None:
    _run_ingest(tmp_path, monkeypatch, _probe(pts, start), rejected=False)


@pytest.mark.parametrize(
    "payload",
    [
        _probe("600.001"),
        _probe("601", "0"),
        _probe("591", "-10"),
        _probe("611", "10"),
        _probe("NaN"),
        _probe("inf"),
        _probe("-inf"),
        _probe(None),
        _probe("invalid"),
        _probe("0", "NaN"),
        _probe("0", "inf"),
        _probe("0", None),
        _probe("0", "invalid"),
        {},
        [],
        {"streams": [{"index": 0}], "frames": [], "format": {"start_time": "0"}},
        {"streams": [{"index": 0}], "frames": [{"stream_index": 0}], "format": {"start_time": "0"}},
        {
            "streams": [{"index": 0}],
            "frames": [{"stream_index": 0, "pts_time": "0"}],
            "no_format": {},
        },
        {
            "streams": [{"index": 0}, {"index": 1}],
            "frames": [{"stream_index": 0, "pts_time": "0"}],
            "format": {"start_time": "0"},
        },
        {
            "streams": [{"index": 0}, {"index": 1}],
            "frames": [
                {"stream_index": 0, "pts_time": "0"},
                {"stream_index": 1, "pts_time": "601"},
            ],
            "format": {"start_time": "0"},
        },
        b"invalid JSON",
        b" " * 65537,
    ],
    ids=lambda payload: (
        "oversized-json" if isinstance(payload, bytes) and len(payload) > 65536 else None
    ),
)
def test_padding_refuses_unknown_or_oversized_offset_before_ffmpeg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, payload: object
) -> None:
    _run_ingest(tmp_path, monkeypatch, payload, rejected=True)


@pytest.mark.parametrize("failure", ["exit", "timeout", "oserror"])
def test_padding_probe_failures_refuse_before_ffmpeg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    _run_ingest(tmp_path, monkeypatch, _probe(), rejected=True, failure=failure)


def test_canonical_wav_copy_needs_no_probe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def missing_binary(name: str) -> None:
        return None

    monkeypatch.setattr(ingest_module.shutil, "which", missing_binary)
    result = ingest_local(FIXTURE, tmp_path / "bundles")
    assert (result.bundle_dir / "media/audio.wav").read_bytes() == FIXTURE.read_bytes()


def test_noncanonical_audio_requires_ffprobe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "synthetic.bin"
    source.write_bytes(b"synthetic noncanonical source")

    def only_ffmpeg(name: str) -> str | None:
        return "/synthetic/ffmpeg" if name == "ffmpeg" else None

    monkeypatch.setattr(ingest_module.shutil, "which", only_ffmpeg)
    with pytest.raises(IngestError, match="ffprobe is required"):
        ingest_module._normalize_to_canonical_wav(source, tmp_path / "audio.wav")  # pyright: ignore[reportPrivateUsage]


def test_absent_container_start_uses_ffmpeg_zero_origin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _probe()
    payload["format"] = {}
    _run_ingest(tmp_path, monkeypatch, payload, rejected=False)


def test_all_audio_streams_checked_without_changing_stream_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = {
        "streams": [{"index": 0}, {"index": 2}],
        "frames": [{"stream_index": 2, "pts_time": "3"}, {"stream_index": 0, "pts_time": "2"}],
        "format": {"start_time": "0"},
    }
    _run_ingest(tmp_path, monkeypatch, payload, rejected=False)
