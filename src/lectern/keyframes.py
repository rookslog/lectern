"""Local scene/interval sampling and dependency-free perceptual deduplication."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from lectern.bundle import FrameRecord, FramesDocument

FRAME_METHOD = "scene+interval+gray64x36-mad"
FRAME_EVIDENCE_LIMIT = "frames are samples; content between frames is not represented"
DEFAULT_FRAME_INTERVAL = 30.0
DEFAULT_MAX_FRAMES = 300


class KeyframeError(RuntimeError):
    """A requested local visual extraction could not complete."""


def validate_frame_options(interval: float, maximum: int) -> None:
    if not math.isfinite(interval) or interval <= 0:
        raise KeyframeError("frame interval must be finite and greater than zero")
    if maximum <= 0:
        raise KeyframeError("max frames must be greater than zero")


# Deduplicate by mean absolute luma difference on a 64x36 thumbnail against the
# last kept frame. Calibrated on one real screen-shared lecture (2026-10-13): a
# held document with a webcam inset differs by 0.3-1.2 levels between samples,
# while scrolled or changed text differs by 2.4-10. A 9x8 difference hash merged
# distinct pages of scrolled text and kept 14 frames where 21 were distinct.
_THUMB_W, _THUMB_H = 64, 36
_THUMB_BYTES = _THUMB_W * _THUMB_H
_MIN_MEAN_ABS_DIFF = 2.0


def _mean_abs_diff(a: bytes, b: bytes) -> float:
    return sum(abs(x - y) for x, y in zip(a, b, strict=True)) / _THUMB_BYTES


def extract_frames(
    snapshot: Path, bundle: Path, *, interval: float, maximum: int
) -> FramesDocument | None:
    """Read only the private snapshot; None means it contains no video stream.

    Merge scene changes and the first frame of each interval bucket. Deduplicate
    candidates against the last retained frame by mean absolute luma difference
    on a 64x36 thumbnail.
    Candidate files are temporary and are never declared as bundle evidence.
    """

    validate_frame_options(interval, maximum)
    probe = shutil.which("ffprobe")
    ffmpeg = shutil.which("ffmpeg")
    if probe is None:
        raise KeyframeError("ffprobe is required for frame extraction")
    result = subprocess.run(
        [
            probe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=index",
            "-of",
            "json",
            str(snapshot),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise KeyframeError("could not probe captured source for video")
    if not json.loads(result.stdout).get("streams"):
        return None
    if ffmpeg is None:
        raise KeyframeError("ffmpeg is required for frame extraction")

    with tempfile.TemporaryDirectory(prefix=".frame-candidates-", dir=bundle) as temporary:
        candidates = Path(temporary)
        # Scene is evaluated first on every input frame. prev_selected_t is also
        # used to detect entry into a new bucket, so scenes do not reset sampling.
        selection = (
            f"gt(scene,0.30)+isnan(prev_selected_t)"
            f"+gt(floor(t/{interval}),floor(prev_selected_t/{interval}))"
        )
        filters = (
            f"[0:v:0]setpts=PTS-STARTPTS,select='{selection}',metadata=mode=print,"
            "scale=w='min(1280,iw)':h='min(1280,ih)':force_original_aspect_ratio=decrease,"
            "showinfo,split=2[jpeg][small];[small]scale=64:36,format=gray[gray]"
        )
        result = subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "info",
                "-threads",
                "2",
                "-i",
                str(snapshot),
                "-filter_complex_threads",
                "2",
                "-filter_complex",
                filters,
                "-map",
                "[jpeg]",
                "-fps_mode",
                "vfr",
                "-c:v",
                "mjpeg",
                "-q:v",
                "5",
                "-threads",
                "2",
                str(candidates / "%08d.jpg"),
                "-map",
                "[gray]",
                "-fps_mode",
                "vfr",
                "-c:v",
                "rawvideo",
                "-threads",
                "2",
                "-f",
                "rawvideo",
                str(candidates / "hashes.gray"),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            # ffmpeg diagnostics contain private paths; do not persist them.
            raise KeyframeError("ffmpeg frame extraction failed")
        facts = re.findall(r"\bpts_time:([\d.eE+-]+).*?\bs:(\d+)x(\d+)", result.stderr)
        scores = re.findall(r"lavfi.scene_score=([\d.eE+-]+)", result.stderr)
        images = sorted(candidates.glob("*.jpg"))
        gray_size = (candidates / "hashes.gray").stat().st_size
        expected_gray = _THUMB_BYTES * len(images)
        if len(facts) != len(images) or len(scores) != len(images) or gray_size != expected_gray:
            raise KeyframeError("ffmpeg candidate timestamps and images do not match")
        frames_dir = bundle / "visual/frames"
        frames_dir.mkdir(parents=True)
        frames: list[FrameRecord] = []
        previous: bytes | None = None
        with (candidates / "hashes.gray").open("rb") as thumbnails:
            for image, (timestamp, width, height), score in zip(images, facts, scores, strict=True):
                gray = thumbnails.read(_THUMB_BYTES)
                if previous is not None and _mean_abs_diff(gray, previous) < _MIN_MEAN_ABS_DIFF:
                    continue
                index = len(frames)
                relative = f"visual/frames/{index:04d}.jpg"
                destination = bundle / relative
                image.replace(destination)
                frames.append(
                    FrameRecord(
                        index=index,
                        t_s=float(timestamp),
                        file=relative,
                        sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                        reason="scene" if float(score) > 0.30 else "interval",
                        width=int(width),
                        height=int(height),
                    )
                )
                previous = gray
                if len(frames) >= maximum:
                    break
        document = FramesDocument(frames)
        (bundle / "visual/frames.json").write_text(document.model_dump_json(indent=2) + "\n")
        return document
