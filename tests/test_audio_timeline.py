"""Synthetic regressions for normalization onto the source container timeline."""

from __future__ import annotations

import json
import math
import shlex
import shutil
import struct
import subprocess
import sys
import wave
from pathlib import Path

import pytest

from lectern import cli
from lectern.ingest import _normalize_to_canonical_wav  # pyright: ignore[reportPrivateUsage]

FIXTURE = Path(__file__).parent / "fixtures/synthetic_talk.wav"


@pytest.fixture
def ffmpeg() -> str:
    binary = shutil.which("ffmpeg")
    if binary is None:
        pytest.skip("ffmpeg is required")
    return binary


def _video(ffmpeg: str, tmp_path: Path, offset: float) -> Path:
    media = tmp_path / "talk.mkv"
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=160x90:r=5:d=12",
            "-itsoffset",
            str(offset),
            "-i",
            str(FIXTURE),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "mpeg4",
            "-threads",
            "2",
            "-c:a",
            "aac",
            str(media),
        ],
        capture_output=True,
        check=True,
    )
    return media


def _wav_times(path: Path) -> tuple[float, float]:
    with wave.open(str(path), "rb") as audio:
        assert (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) == (16000, 1, 2)
        count = audio.getnframes()
        samples = struct.unpack(f"<{count}h", audio.readframes(count))
        first_sound = next(index for index, sample in enumerate(samples) if abs(sample) > 100)
        return first_sound / audio.getframerate(), count / audio.getframerate()


def test_normalization_preserves_delayed_audio_on_container_timeline(
    ffmpeg: str, tmp_path: Path
) -> None:
    media = _video(ffmpeg, tmp_path, 2.4)
    output = tmp_path / "normalized.wav"
    _normalize_to_canonical_wav(media, output)

    first_sound, duration = _wav_times(output)
    _, fixture_duration = _wav_times(FIXTURE)
    assert math.isclose(first_sound, 2.4, abs_tol=0.05)
    assert math.isclose(duration, fixture_duration + 2.4, abs_tol=0.1)


def test_normalization_keeps_zero_offset_audio_timing(ffmpeg: str, tmp_path: Path) -> None:
    media = _video(ffmpeg, tmp_path, 0.0)
    output = tmp_path / "normalized.wav"
    _normalize_to_canonical_wav(media, output)

    first_sound, duration = _wav_times(output)
    # Base measurement with AAC in MKV: first sound 0.00025 s, duration 3.008 s.
    assert math.isclose(first_sound, 0.00025, abs_tol=0.05)
    assert math.isclose(duration, 3.008, abs_tol=0.1)


def test_normalization_keeps_audio_only_flac_timing(ffmpeg: str, tmp_path: Path) -> None:
    media = tmp_path / "talk.flac"
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(FIXTURE),
            "-c:a",
            "flac",
            "-threads",
            "2",
            str(media),
        ],
        capture_output=True,
        check=True,
    )
    output = tmp_path / "normalized.wav"
    _normalize_to_canonical_wav(media, output)

    first_sound, duration = _wav_times(output)
    # Base measurement: FLAC has no encoder priming; 0.00025 s onset, 2.9676875 s duration.
    assert math.isclose(first_sound, 0.00025, abs_tol=1 / 16000)
    assert math.isclose(duration, 2.9676875, abs_tol=1 / 16000)


def test_delayed_audio_transcript_uses_container_timeline(
    ffmpeg: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    media = _video(ffmpeg, tmp_path, 2.4)
    transcriber = tmp_path / "transcriber.py"
    transcriber.write_text(
        "import json, struct, sys, wave\n"
        "with wave.open(sys.argv[1], 'rb') as audio:\n"
        "    count = audio.getnframes()\n"
        "    samples = struct.unpack(f'<{count}h', audio.readframes(count))\n"
        "    first = next(i for i, sample in enumerate(samples) if abs(sample) > 100)\n"
        "    start = first / audio.getframerate()\n"
        "    end = count / audio.getframerate()\n"
        "print(json.dumps({'segments': [{'start_s': start, 'end_s': end,\n"
        "                              'text': 'Synthetic first sound.'}]}))\n",
        encoding="utf-8",
    )
    assert (
        cli.main(
            [
                "ingest",
                str(media),
                "--output",
                str(tmp_path / "bundles"),
                "--state",
                str(tmp_path / "state.db"),
                "--transcriber-command",
                shlex.join([sys.executable, str(transcriber)]),
                "--json",
            ]
        )
        == 0
    )
    bundle = Path(json.loads(capsys.readouterr().out)["bundle_dir"])
    segments = json.loads((bundle / "transcript/segments.json").read_text(encoding="utf-8"))
    assert segments[0]["source"] == "local_command"
    assert math.isclose(segments[0]["start_s"], 2.4, abs_tol=0.05)
