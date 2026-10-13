from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from lectern import cli
from lectern.automation import open_state
from lectern.bundle import Manifest, StageName, StageState

FIXTURE = Path(__file__).parent / "fixtures/synthetic_talk.wav"


@pytest.fixture
def video(tmp_path: Path) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg and ffprobe are required")
    encoders = subprocess.run([ffmpeg, "-encoders"], capture_output=True, text=True, check=True)
    if " mpeg4 " not in encoders.stdout or " pcm_s16le " not in encoders.stdout:
        pytest.skip("ffmpeg needs mpeg4 and pcm_s16le encoders")
    path = tmp_path / "slides.mkv"
    command = [ffmpeg, "-hide_banner", "-loglevel", "error"]
    for color in ("black", "white", "red"):
        command.extend(["-f", "lavfi", "-i", f"color=c={color}:s=160x90:r=5:d=4"])
    command.extend(
        [
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=12",
            "-filter_complex",
            "[0:v][1:v][2:v]concat=n=3:v=1:a=0[v]",
            "-map",
            "[v]",
            "-map",
            "3:a",
            "-c:v",
            "mpeg4",
            "-threads",
            "2",
            "-c:a",
            "pcm_s16le",
            str(path),
        ]
    )
    subprocess.run(command, capture_output=True, check=True)
    path.with_suffix(".transcript.txt").write_text("Synthetic slides only.", encoding="utf-8")
    return path


def ingest(
    media: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], *options: str
) -> tuple[Path, str]:
    assert (
        cli.main(
            [
                "ingest",
                str(media),
                "--output",
                str(tmp_path / "bundles"),
                "--state",
                str(tmp_path / "state.db"),
                "--json",
                *options,
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    return Path(result["bundle_dir"]), result["bundle_id"]


def export(bundle_id: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> dict[str, Any]:
    assert cli.main(["export", bundle_id, "--state", str(tmp_path / "state.db")]) == 0
    return json.loads(capsys.readouterr().out)


def test_video_frames_are_deduplicated_declared_and_citeable(
    video: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bundle, bundle_id = ingest(video, tmp_path, capsys, "--frames")
    frames = json.loads((bundle / "visual/frames.json").read_text())
    assert len(frames) == 3
    assert all(abs(row["t_s"] - time) <= 1 for row, time in zip(frames, [0, 4, 8], strict=True))
    assert [row["reason"] for row in frames] == ["interval", "scene", "scene"]
    visual = Manifest.load(bundle).stages[StageName.VISUAL]
    assert visual.state is StageState.DONE
    assert {ref.path for ref in visual.outputs} == {
        "visual/frames.json",
        "visual/frames/0000.jpg",
        "visual/frames/0001.jpg",
        "visual/frames/0002.jpg",
    }
    for row in frames:
        assert row["width"] == 160 and row["height"] == 90
        assert hashlib.sha256((bundle / row["file"]).read_bytes()).hexdigest() == row["sha256"]
    packet = export(bundle_id, tmp_path, capsys)
    assert packet["layers"]["visual"] == {
        "present": True,
        "frame_count": 3,
        "method": "scene+interval+gray64x36-mad",
        "evidence_limit": "frames are samples; content between frames is not represented",
    }
    assert [row["file"] for row in packet["frames"]] == [row["file"] for row in frames]
    review = tmp_path / "review.md"
    review.write_text("\n\n".join(row["cite"] for row in packet["frames"]), encoding="utf-8")
    check = ["check-citations", str(review), "--state", str(tmp_path / "state.db"), "--json"]
    assert cli.main(check) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] == 3 and report["problems"] == 0
    short = bundle_id.rsplit("-", 1)[1]
    review.write_text(f"[t=00:05 {short}#f1]\n\n[t=00:00 {short}#f99]", encoding="utf-8")
    assert cli.main(check) == 1
    assert [row["status"] for row in json.loads(capsys.readouterr().out)["citations"]] == [
        "time_mismatch",
        "unknown_frame",
    ]
    assert (
        cli.main(["export", bundle_id, "--format", "md", "--state", str(tmp_path / "state.db")])
        == 0
    )
    markdown = capsys.readouterr().out
    assert "## Frames" in markdown and "#f<index>" in markdown
    assert f"[t=00:04 {short}#f1] frames/0001.jpg (scene)" in markdown
    frame_rows = [
        line for line in markdown.split("## Frames", 1)[1].splitlines() if line.startswith("[t=")
    ]
    review.write_text("\n\n".join(frame_rows), encoding="utf-8")
    assert cli.main(check) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] == 3 and report["problems"] == 0
    assert cli.main(["library", "show", bundle_id, "--state", str(tmp_path / "state.db")]) == 0
    assert "Visual: yes (3 frames)" in capsys.readouterr().out
    # Queue options must not replay the differently processed one-shot bundle.
    with open_state(tmp_path / "state.db") as state:
        queue_id = state.get_library_bundle(bundle_id).queue_item_id
        state.approve_queue_item(queue_id)
    assert (
        cli.main(
            [
                "queue",
                "ingest",
                queue_id,
                "--frames",
                "--frame-interval",
                "1",
                "--max-frames",
                "2",
                "--output",
                str(tmp_path / "bundles"),
                "--state",
                str(tmp_path / "state.db"),
                "--json",
            ]
        )
        == 0
    )
    limited = json.loads(capsys.readouterr().out)
    assert limited["bundle_id"] != bundle_id
    rows = json.loads((Path(limited["bundle_dir"]) / "visual/frames.json").read_text())
    assert len(rows) == 2
    assert all(abs(row["t_s"] - time) <= 1 for row, time in zip(rows, [0, 4], strict=True))
    # Strict artifact model round-trips the actual index.
    from lectern.bundle import FramesDocument

    document = FramesDocument.model_validate_json((bundle / "visual/frames.json").read_bytes())
    assert FramesDocument.model_validate_json(document.model_dump_json()) == document


def test_max_frames_one_leaves_only_one_bundle_jpeg(
    video: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bundle, _ = ingest(video, tmp_path, capsys, "--frames", "--max-frames", "1")
    assert list(bundle.rglob("*.jpg")) == [bundle / "visual/frames/0000.jpg"]


def test_visual_done_without_declared_frames_exports(
    video: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bundle, bundle_id = ingest(video, tmp_path, capsys)
    manifest = Manifest.load(bundle)
    manifest.stages[StageName.VISUAL].state = StageState.DONE
    manifest.save(bundle)
    packet = export(bundle_id, tmp_path, capsys)
    assert packet["layers"]["visual"] == {"present": True}
    assert packet["frames"] == []
    assert (
        cli.main(["export", bundle_id, "--format", "md", "--state", str(tmp_path / "state.db")])
        == 0
    )
    assert "- visual: present\n" in capsys.readouterr().out


def test_audio_with_frames_reports_no_video(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    if shutil.which("ffprobe") is None:
        pytest.skip("ffprobe is required")
    audio = tmp_path / "audio.wav"
    audio.write_bytes(FIXTURE.read_bytes())
    audio.with_suffix(".transcript.txt").write_text("Synthetic audio only.", encoding="utf-8")
    bundle, bundle_id = ingest(audio, tmp_path, capsys, "--frames")
    assert Manifest.load(bundle).stages[StageName.VISUAL].state is StageState.SKIPPED
    assert not (bundle / "visual").exists()
    assert export(bundle_id, tmp_path, capsys)["layers"]["visual"] == {
        "present": False,
        "reason": "no video stream",
    }


def test_video_without_frames_is_not_requested(
    video: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bundle, bundle_id = ingest(video, tmp_path, capsys)
    assert not (bundle / "visual").exists()
    assert export(bundle_id, tmp_path, capsys)["layers"]["visual"] == {
        "present": False,
        "reason": "not requested",
    }
    _, framed_id = ingest(video, tmp_path, capsys, "--frames")
    assert framed_id != bundle_id
