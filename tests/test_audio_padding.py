"""Bounded frame-stream tests: large timelines never produce PCM."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest

from lectern import ingest as ingest_module
from lectern.ingest import IngestError, ingest_local

FIXTURE = Path(__file__).parent / "fixtures/synthetic_talk.wav"


def _frame(
    n: int, pts: int, samples: int = 16000, rate: int = 16000, *, pointer: str = "0x123"
) -> str:
    return (
        f"[ashowinfo@TAG @ {pointer}] n:{n} pts:{pts} pts_time:{pts / rate:g} "
        f"fmt:s16 channels:1 chlayout:mono rate:{rate} nb_samples:{samples} "
        "checksum:1234ABCD plane_checksums: [ 1234ABCD ]\n"
    )


def _auxiliary(body: str = "downmix: preferred downmix type - unknown") -> str:
    return f"[ashowinfo@TAG @ 0x123]   side data - {body}\n"


@pytest.fixture
def analyzer(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[list[str], list[subprocess.Popen[bytes]]]]:
    """Fake FFmpeg at the process boundary; exercise real pipes/deadlines/reaping."""
    lines: list[str] = []
    children: list[subprocess.Popen[bytes]] = []
    real_popen = subprocess.Popen

    def launch(args: list[str], **kwargs: Any) -> subprocess.Popen[bytes]:
        assert args[args.index("-af") + 1].startswith("asettb=expr=1/sr,ashowinfo@")
        assert not any("aresample" in arg for arg in args)
        assert args[-3:] == ["-f", "null", "-"]
        assert "-map" not in args and "-copyts" not in args
        assert all(flag in args for flag in ("-vn", "-sn", "-dn", "-nostdin"))
        tag = args[args.index("-af") + 1].split("ashowinfo@", 1)[1]
        script = (
            "import json, os, sys, time\n"
            "for line in json.loads(sys.argv[1]):\n"
            " if line == 'SLEEP': time.sleep(30)\n"
            " elif line == 'LONG_JITTER':\n"
            "  template = sys.argv[3].replace('TAG', sys.argv[2])\n"
            "  for i in range(72001):\n"
            "   pts = i*16000 + (320 if i%2 else 0)\n"
            "   text = template.replace('n:0', f'n:{i}').replace('pts:0', f'pts:{pts}')\n"
            "   os.write(2, text.encode())\n"
            " elif line == 'EXIT': sys.exit(1)\n"
            " elif line == 'CLOSE': os.close(2); time.sleep(30)\n"
            " else: os.write(2, line.replace('TAG', sys.argv[2]).encode())\n"
        )
        child = cast(
            "subprocess.Popen[bytes]",
            real_popen(
                [sys.executable, "-u", "-c", script, json.dumps(lines), tag, _frame(0, 0)], **kwargs
            ),
        )
        children.append(child)
        return child

    monkeypatch.setattr(subprocess, "Popen", launch)
    yield lines, children
    assert children, "the full frame analysis must run"
    assert all(child.poll() is not None for child in children), "analysis child was not reaped"
    assert all(child.stderr is not None and child.stderr.closed for child in children)


def _run_ingest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, rejected: bool) -> None:
    source = tmp_path / "synthetic.bin"
    source.write_bytes(b"synthetic noncanonical source")
    source.with_suffix(".transcript.txt").write_text("Synthetic transcript.")
    normalizations: list[str] = []

    def run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        normalizations.append(args[-1])
        assert not rejected, "over-budget/indeterminate audio reached PCM normalization"
        shutil.copyfile(FIXTURE, args[-1])
        return subprocess.CompletedProcess(args, 0, "", "")

    def binary_path(name: str) -> str:
        assert name == "ffmpeg", "normalization must not require ffprobe"
        return "/synthetic/ffmpeg"

    monkeypatch.setattr(ingest_module.shutil, "which", binary_path)
    monkeypatch.setattr(subprocess, "run", run)
    bundles = tmp_path / "bundles"
    if rejected:
        with pytest.raises(IngestError, match="audio (padding|timing)") as error:
            ingest_local(source, bundles)
        assert "private" not in str(error.value)
        assert "ashowinfo" not in str(error.value)
        assert "lectern_" not in str(error.value)
        assert normalizations == []
        assert list(bundles.iterdir()) == []
        assert not list(tmp_path.rglob("*.tmp.wav"))
    else:
        result = ingest_local(source, bundles)
        assert (result.bundle_dir / "media/audio.wav").read_bytes() == FIXTURE.read_bytes()
        assert len(normalizations) == 1


@pytest.mark.parametrize(
    "records",
    [
        [_frame(0, 0)],
        [_frame(0, 9600000)],  # exactly 600 seconds initially
        [_frame(0, 0), _frame(1, 9616000)],  # exactly 600 seconds inside
        [_frame(0, 0), _frame(1, 4816000), _frame(2, 9632000)],  # 300 + 300
        [_frame(0, -1024, 1024), _frame(1, 0)],  # negative codec priming
        [_frame(0, 0), _frame(1, 15999), _frame(2, 32000)],  # tiny overlap
        [_frame(0, 0), _frame(1, 48000)],  # legitimate two-second gap
        [_frame(0, 0, 44100, 44100), _frame(1, 88200, 44100, 44100)],
    ],
)
def test_accepts_budget_boundary_and_accountable_timing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    analyzer: tuple[list[str], list[subprocess.Popen[bytes]]],
    records: list[str],
) -> None:
    analyzer[0].extend(records)
    _run_ingest(tmp_path, monkeypatch, rejected=False)


def test_accepts_old_ffmpeg_packet_position_field(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    analyzer: tuple[list[str], list[subprocess.Popen[bytes]]],
) -> None:
    # FFmpeg 5.1 emitted pkt_pos between pts_time and fmt; 6.1 removed it.
    analyzer[0].extend(
        [
            _frame(0, 0).replace("fmt:", "pos:-1 fmt:"),
            _frame(1, 16000).replace("fmt:", "pos:4096 fmt:"),
        ]
    )
    _run_ingest(tmp_path, monkeypatch, rejected=False)


@pytest.mark.parametrize("position", ["", "pos:-1 "])
@pytest.mark.parametrize(
    "body",
    [
        "downmix: preferred downmix type - unknown",
        "opaque auxiliary payload: n:999 pts:NOPTS side data",
    ],
)
def test_auxiliary_diagnostics_between_frames_do_not_advance_clock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    analyzer: tuple[list[str], list[subprocess.Popen[bytes]]],
    position: str,
    body: str,
) -> None:
    analyzer[0].extend(
        [
            _frame(0, 0).replace("fmt:", position + "fmt:"),
            _auxiliary(body),
            _auxiliary(),
            _frame(1, 16000).replace("fmt:", position + "fmt:"),
        ]
    )
    _run_ingest(tmp_path, monkeypatch, rejected=False)


@pytest.mark.parametrize(
    "records",
    [
        [_auxiliary(), _frame(0, 0)],  # no observed filter pointer yet
        [_frame(0, 0), _auxiliary().replace("0x123", "0x456")],
        [_frame(0, 0), _auxiliary().replace("0x123", "not-hex")],
        [_frame(0, 0), _auxiliary("")],
        [_frame(0, 0), _auxiliary().replace("side data - ", "side data -")],
        [_frame(0, 0), _auxiliary().replace("]   side", "]  side")],
        [_frame(0, 0), _auxiliary("downmix\rhidden suffix")],
        [_frame(0, 0), _auxiliary("x" * 4096)],
        [_frame(0, 0), _auxiliary(), _frame(2, 32000)],  # missing frame remains unknown
        [_frame(0, 0), _auxiliary(), _frame(1, 9632000)],  # later gap still over budget
        [_frame(0, 0), _frame(1, 16000).replace("n:1", "n:side data")],
        [_frame(0, 0), _frame(1, 16000).replace("pts:16000", "pts:side data")],
        [_frame(0, 0), _frame(1, 16000).rstrip("\n") + " side data - downmix\n"],
        [_frame(0, 0), _auxiliary(), "[ashowinfo@TAG @ 0x123] unknown suffix\n"],
        [_auxiliary().replace("TAG", "static_forgery")],
        [
            _frame(0, 0),
            _frame(1, 16000).replace("TAG", "static_forgery"),
            _auxiliary().replace("TAG", "static_forgery"),
            _frame(2, 32000),
        ],
        [_frame(0, 0), _auxiliary(), "SLEEP"],
    ],
)
def test_auxiliary_diagnostics_cannot_admit_unknown_or_overbudget_timing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    analyzer: tuple[list[str], list[subprocess.Popen[bytes]]],
    records: list[str],
) -> None:
    from lectern import audio_timing

    monkeypatch.setattr(audio_timing, "ANALYSIS_TIMEOUT_S", 0.2)
    analyzer[0].extend(records)
    _run_ingest(tmp_path, monkeypatch, rejected=True)


@pytest.mark.parametrize(
    "records",
    [
        [_frame(0, 9600001)],  # one native sample above 600 seconds
        [_frame(0, 0), _frame(1, 9616001)],
        [_frame(0, 0), _frame(1, 4832000), _frame(2, 9664000)],  # 301 + 301
        [*[_frame(i, i * 16000) for i in range(70)], _frame(70, 10736000)],
        [_frame(0, 0), _frame(1, 0)],  # one second overlap
        [_frame(0, 0), _frame(1, -1)],  # backward PTS
        [_frame(0, 0), _frame(1, 16000, 48000, 48000)],  # changed rate
        [_frame(0, 0), _frame(0, 16000)],  # filter reinit/reset
        [_frame(0, 0), _frame(1, 16000, pointer="0x456")],
        [_frame(0, 0), _frame(2, 16000)],  # dropped tagged record
        [_frame(0, 0).replace("fmt:s16", "fmt:fltp"), _frame(1, 16000)],
    ],
)
def test_refuses_aggregate_or_indeterminate_timing_before_pcm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    analyzer: tuple[list[str], list[subprocess.Popen[bytes]]],
    records: list[str],
) -> None:
    analyzer[0].extend(records)
    _run_ingest(tmp_path, monkeypatch, rejected=True)


@pytest.mark.parametrize(
    "records",
    [
        [],
        ["private diagnostics must not escape\n"],
        [_frame(0, 0).replace("TAG", "forged_static_tag")],
        [_frame(0, 0).replace("ashowinfo@TAG", "Parsed_ashowinfo_0")],
        [_frame(0, 0).replace("pts:0", "pts:NOPTS")],
        [_frame(0, 0).replace("rate:16000", "rate:0")],
        [_frame(0, 0, 0)],
        [_frame(0, 0, -1)],
        [_frame(0, 0).replace("pts:0", "pts:" + "9" * 100)],
        ["[ashowinfo@TAG @ 0x123] n:0 malformed private record\n"],
        ["x" * 4097],
        ["x" * 4097 + "\n"],
        [_frame(0, 0), "EXIT"],
        [_frame(0, 0), "SLEEP"],
        [_frame(0, 0), "CLOSE"],
    ],
)
def test_rejects_malformed_forged_missing_failed_or_stalled_analysis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    analyzer: tuple[list[str], list[subprocess.Popen[bytes]]],
    records: list[str],
) -> None:
    from lectern import audio_timing

    monkeypatch.setattr(audio_timing, "ANALYSIS_TIMEOUT_S", 0.2)
    analyzer[0].extend(records)
    _run_ingest(tmp_path, monkeypatch, rejected=True)


def test_long_continuous_jitter_does_not_accumulate_per_frame_padding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    analyzer: tuple[list[str], list[subprocess.Popen[bytes]]],
) -> None:
    # Twenty hours; naive positive adjacent-delta sums would charge 720 seconds.
    analyzer[0].append("LONG_JITTER")
    _run_ingest(tmp_path, monkeypatch, rejected=False)


def test_untagged_forgery_cannot_mask_later_real_gap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    analyzer: tuple[list[str], list[subprocess.Popen[bytes]]],
) -> None:
    analyzer[0].extend(
        [
            _frame(0, 0),
            _frame(1, 16000).replace("TAG", "generic_forgery"),
            _frame(1, 9632000),
            "SLEEP",
        ]
    )
    _run_ingest(tmp_path, monkeypatch, rejected=True)


def test_canonical_wav_copy_needs_no_analysis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def missing_binary(name: str) -> None:
        return None

    monkeypatch.setattr(ingest_module.shutil, "which", missing_binary)
    result = ingest_local(FIXTURE, tmp_path / "bundles")
    assert (result.bundle_dir / "media/audio.wav").read_bytes() == FIXTURE.read_bytes()


def test_analysis_spawn_failure_prevents_pcm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unavailable(*args: object, **kwargs: object) -> None:
        raise OSError("private process diagnostic")

    monkeypatch.setattr(subprocess, "Popen", unavailable)
    _run_ingest(tmp_path, monkeypatch, rejected=True)


def test_read_error_reaps_and_closes_analysis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    analyzer: tuple[list[str], list[subprocess.Popen[bytes]]],
) -> None:
    from lectern import audio_timing

    real_read = audio_timing.os.read
    analyzer[0].extend([_frame(0, 0), "SLEEP"])

    def unreadable(fd: int, count: int) -> bytes:
        pipe = analyzer[1][-1].stderr if analyzer[1] else None
        if pipe is not None and fd == pipe.fileno():
            raise OSError("private diagnostic")
        return real_read(fd, count)

    monkeypatch.setattr(audio_timing.os, "read", unreadable)
    _run_ingest(tmp_path, monkeypatch, rejected=True)


def test_maximum_sized_diagnostic_and_unterminated_final_frame(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    analyzer: tuple[list[str], list[subprocess.Popen[bytes]]],
) -> None:
    analyzer[0].extend(["x" * 4096 + "\n", _frame(0, 0).rstrip("\n")])
    _run_ingest(tmp_path, monkeypatch, rejected=False)
