"""Bound timestamp-induced silence before normalization creates a PCM file.

The null pass decodes the automatically selected audio, without compensation.
It deliberately does not establish a duration limit or sandbox the decoder.
"""

from __future__ import annotations

import os
import re
import secrets
import selectors
import subprocess
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

MAX_PADDING_S = 600
ANALYSIS_TIMEOUT_S = 60
MAX_RECORD_BYTES = 4096


class AudioTimingError(RuntimeError):
    """Timing could not be admitted within the padding/resource contract."""


@dataclass
class _SampleClock:
    frames: int = 0
    rate: int = 0
    signature: tuple[bytes, ...] = ()
    previous_pts: int = 0
    expected: int = 0
    padding: int = 0

    def observe(self, record: re.Match[bytes]) -> None:
        n, pts, rate, samples = (int(record[name]) for name in ("n", "pts", "rate", "samples"))
        signature = tuple(record[name] for name in ("pointer", "fmt", "channels", "layout"))
        if (
            n != self.frames
            or not 0 < rate <= 2**31 - 1
            or not 0 < samples <= 2**31 - 1
            or not 0 < int(record["channels"]) <= 2**31 - 1
            or not -(2**63) < pts < 2**63
        ):
            raise AudioTimingError("audio timing has invalid or discontinuous frame records")
        if self.frames == 0:
            self.rate, self.signature = rate, signature
            # first_pts=0 trims negative priming, rather than inserting silence.
            # Keep its sample origin so that the trimmed prefix is not charged.
            self.expected = min(pts, 0)
        elif rate != self.rate or signature != self.signature:
            raise AudioTimingError(
                "audio timing rate/format or filter reinitialization is unsupported"
            )
        elif pts < self.previous_pts:
            raise AudioTimingError("audio timing moves backward")

        delta = pts - self.expected
        # Refuse clock debt >= 50 ms, before the unchanged resampler's 100 ms
        # hard-trim threshold. Smaller quantization/overlap never resets the
        # clock downward. Without later hard trims, the positive high-water
        # displacement bounds inserted samples, including rounding to samples.
        if delta * 20 <= -rate:
            raise AudioTimingError("audio timing overlap/drift is unsupported (50 ms limit)")
        if delta > 0:
            self.padding += delta
            self.expected = pts
        if self.padding > MAX_PADDING_S * rate:
            raise AudioTimingError("audio padding exceeds the aggregate 600-second maximum")
        self.expected += samples
        self.previous_pts = pts
        self.frames += 1


def check_audio_padding(ffmpeg: str, source: Path) -> None:
    """Read exact, per-invocation tagged frame records to EOF, or refuse."""
    tag = "lectern_" + secrets.token_hex(16)
    prefix = ("[ashowinfo@" + tag + " @ ").encode("ascii")
    record = re.compile(
        re.escape(prefix) + rb"(?P<pointer>0x[0-9a-fA-F]+)\] n:(?P<n>[0-9]{1,19}) "
        rb"pts:(?P<pts>-?[0-9]{1,19}) "
        rb"pts_time:[-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][-+]?[0-9]+)? "
        # FFmpeg 5.1 included packet position; 6.1 and later omit it.
        rb"(?:pos:-?[0-9]{1,19} )?"
        rb"fmt:(?P<fmt>[a-z0-9]+) channels:(?P<channels>[0-9]{1,10}) "
        rb"chlayout:(?P<layout>[^\r\n]+?) rate:(?P<rate>[0-9]{1,10}) "
        rb"nb_samples:(?P<samples>[0-9]{1,10}) checksum:[0-9A-Fa-f]+ "
        rb"plane_checksums: \[[ 0-9A-Fa-f]+\]"
    )
    auxiliary = re.compile(
        re.escape(prefix) + rb"(?P<pointer>0x[0-9a-fA-F]+)\]   side data - [^\r\n]+"
    )
    clock = _SampleClock()

    def accept(line: bytes) -> None:
        if len(line) > MAX_RECORD_BYTES:
            raise AudioTimingError("audio timing analysis record exceeds 4096 bytes")
        if line.startswith(prefix):
            match = record.fullmatch(line)
            if match is None:
                side_data = auxiliary.fullmatch(line)
                if (
                    side_data is not None
                    and clock.frames > 0
                    and side_data["pointer"] == clock.signature[0]
                ):
                    # Authentic auxiliary diagnostics carry no timing. Only
                    # the already observed filter may emit them; do not skip
                    # malformed n/pts records merely mentioning side data.
                    return
                raise AudioTimingError("audio timing analysis has a malformed frame record")
            clock.observe(match)
        # All other diagnostics, including static forged ashowinfo messages,
        # are discarded, never returned to the caller or persisted.

    deadline = time.monotonic() + ANALYSIS_TIMEOUT_S

    def remaining() -> float:
        seconds = deadline - time.monotonic()
        if seconds <= 0:
            raise AudioTimingError("audio timing analysis exceeded its 60-second deadline")
        return seconds

    try:
        process = subprocess.Popen(
            [
                ffmpeg,
                "-hide_banner",
                "-nostdin",
                "-nostats",
                "-loglevel",
                "info",
                "-threads",
                "1",
                "-i",
                str(source),
                "-vn",
                "-sn",
                "-dn",
                "-filter_threads",
                "1",
                "-af",
                # sr is FFmpeg's actual frame sample rate, not stream metadata.
                # Same sample timebase used by aresample for input timestamps.
                "asettb=expr=1/sr,ashowinfo@" + tag,
                "-f",
                "null",
                "-",
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
    except OSError as exc:
        raise AudioTimingError("audio timing analysis could not start") from exc

    try:
        assert process.stderr is not None
        with selectors.DefaultSelector() as selector:
            selector.register(process.stderr, selectors.EVENT_READ)
            pending = b""
            while True:
                if not selector.select(remaining()):
                    remaining()  # raises at the whole-pass deadline
                    continue
                chunk = os.read(process.stderr.fileno(), MAX_RECORD_BYTES)
                if not chunk:
                    if pending:
                        accept(pending)
                    break
                pending += chunk  # at most two record-sized chunks
                while b"\n" in pending:
                    line, pending = pending.split(b"\n", 1)
                    accept(line)
                if len(pending) > MAX_RECORD_BYTES:
                    raise AudioTimingError("audio timing analysis record exceeds 4096 bytes")
            result = process.wait(timeout=remaining())
        if result != 0 or clock.frames == 0:
            raise AudioTimingError(
                "audio timing analysis failed or produced no tagged audio frames"
            )
    except subprocess.TimeoutExpired as exc:
        raise AudioTimingError("audio timing analysis exceeded its 60-second deadline") from exc
    except OSError as exc:
        raise AudioTimingError("audio timing analysis could not read frame records") from exc
    finally:
        try:
            if process.poll() is None:
                # It may exit between poll() and kill(); still reap it.
                with suppress(ProcessLookupError):
                    process.kill()
        finally:
            try:
                process.wait()
            finally:
                if process.stderr is not None:
                    process.stderr.close()
