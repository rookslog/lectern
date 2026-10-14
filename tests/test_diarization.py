"""Local speaker-layer acceptance checks with synthetic audio and invented speech."""

from __future__ import annotations

import hashlib
import importlib
import json
import shutil
import wave
from pathlib import Path
from typing import Any

import pytest

from lectern import cli
from lectern.automation import open_state
from lectern.bundle import ARTIFACT_MODELS, Manifest, StageName, StageState
from lectern.ingest import plan_local_bundle_id

FIXTURE = Path(__file__).parent / "fixtures/synthetic_talk.wav"


def _input(tmp_path: Path) -> Path:
    media = tmp_path / "panel.wav"
    shutil.copyfile(FIXTURE, media)
    media.with_suffix(".vtt").write_text(
        "WEBVTT\n\n00:00.000 --> 00:01.000\nA proposed map.\n\n"
        "00:01.000 --> 00:02.000\nA second perspective.\n\n"
        "00:02.000 --> 00:03.000\nA closing thought.\n",
        encoding="utf-8",
    )
    return media


def _ingest(media: Path, tmp_path: Path, *options: str) -> int:
    return cli.main(
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


def _export(bundle_id: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> dict[str, Any]:
    assert cli.main(["export", bundle_id, "--state", str(tmp_path / "state.db")]) == 0
    return json.loads(capsys.readouterr().out)


def test_diarized_ingest_maps_overlap_and_keeps_citations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from lectern.diarization import SpeakerTurn

    class FakeDiarizer:
        def diarize(self, audio: Path) -> tuple[SpeakerTurn, ...]:
            with wave.open(str(audio)) as wav:
                assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (16000, 1, 2)
            # Raw ids are deliberately not first-appearance labels. Segment 0
            # overlaps both speakers (S1 longest); segment 2 overlaps neither.
            return (SpeakerTurn(0.0, 0.8, 7), SpeakerTurn(0.8, 2.0, 3))

    def fake_factory(
        models_dir: Path, *, speakers: int | None, allow_model_download: bool
    ) -> FakeDiarizer:
        assert speakers == 2 and not allow_model_download
        return FakeDiarizer()

    monkeypatch.setattr("lectern.diarization.create_diarizer", fake_factory)
    media = _input(tmp_path)
    assert _ingest(media, tmp_path, "--diarize", "--speakers", "2") == 0
    result = json.loads(capsys.readouterr().out)
    bundle, bundle_id = Path(result["bundle_dir"]), result["bundle_id"]
    manifest = Manifest.load(bundle)
    stage = manifest.stages[StageName.DIARIZE]
    assert stage.state is StageState.DONE
    payload = (bundle / "transcript/diarization.json").read_bytes()
    document = ARTIFACT_MODELS["diarization"].model_validate_json(payload, strict=True)
    artifact = stage.outputs[0]
    assert (artifact.path, artifact.bytes, artifact.sha256) == (
        "transcript/diarization.json",
        len(payload),
        hashlib.sha256(payload).hexdigest(),
    )
    assert document.model_dump()["segment_speakers"] == [
        {"segment_id": 0, "speaker": "S1"},
        {"segment_id": 1, "speaker": "S2"},
        {"segment_id": 2, "speaker": None},
    ]
    packet = _export(bundle_id, tmp_path, capsys)
    assert [s["speaker"] for s in packet["segments"]] == ["S1", "S2", None]
    assert packet["layers"]["speakers"]["speaker_count"] == 2
    assert packet["layers"]["speakers"]["count_source"] == "given"
    assert (
        cli.main(["export", bundle_id, "--format", "md", "--state", str(tmp_path / "state.db")])
        == 0
    )
    markdown = capsys.readouterr().out
    row = next(line for line in markdown.splitlines() if line.startswith("[t=00:00"))
    assert row == f"{packet['segments'][0]['anchor']['cite']} S1: A proposed map."
    assert "speakers: present; count: 2 (given)" in markdown
    review = tmp_path / "review.md"
    review.write_text(row, encoding="utf-8")
    assert (
        cli.main(["check-citations", str(review), "--state", str(tmp_path / "state.db"), "--json"])
        == 0
    )
    assert json.loads(capsys.readouterr().out)["citations"][0]["status"] == "ok"
    assert cli.main(["library", "show", bundle_id, "--state", str(tmp_path / "state.db")]) == 0
    assert "Speakers: yes (2, given)" in capsys.readouterr().out
    assert cli.main(["transcript", bundle_id, "--state", str(tmp_path / "state.db")]) == 0
    assert "S1: A proposed map." in capsys.readouterr().out
    assert bundle_id != plan_local_bundle_id(media)
    # Exercise queue parsing and deterministic identity without another backend.
    with open_state(tmp_path / "state.db") as state:
        queue_id = state.get_library_bundle(bundle_id).queue_item_id
        state.approve_queue_item(queue_id)
    assert (
        cli.main(
            [
                "queue",
                "ingest",
                queue_id,
                "--diarize",
                "--speakers",
                "2",
                "--diarize-models",
                str(tmp_path / "models"),
                "--output",
                str(tmp_path / "bundles"),
                "--state",
                str(tmp_path / "state.db"),
                "--json",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["bundle_id"] == bundle_id


def test_unrequested_diarization_preserves_unlabelled_ingest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    media = _input(tmp_path)
    assert _ingest(media, tmp_path) == 0
    result = json.loads(capsys.readouterr().out)
    bundle = Path(result["bundle_dir"])
    assert result["bundle_id"] == plan_local_bundle_id(media)
    assert not (bundle / "transcript/diarization.json").exists()
    stage = Manifest.load(bundle).stages[StageName.DIARIZE]
    assert stage.state is StageState.SKIPPED and stage.warnings == ["not requested"]
    packet = _export(result["bundle_id"], tmp_path, capsys)
    assert packet["layers"]["speakers"] == {"present": False, "reason": "not requested"}
    assert all("speaker" not in s for s in packet["segments"])


def test_missing_local_models_fail_cleanly_without_paths(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        _ingest(
            _input(tmp_path), tmp_path, "--diarize", "--diarize-models", str(tmp_path / "missing")
        )
        == 1
    )
    error = capsys.readouterr().err
    assert "missing diarization model: silero_vad.onnx" in error
    assert "--allow-model-download" in error and "--diarize-models" in error
    assert str(tmp_path) not in error and "Traceback" not in error
    assert list((tmp_path / "bundles").iterdir()) == []


@pytest.mark.parametrize("num_speakers", [None, 2])
def test_vendored_pipeline_separates_alternating_synthetic_speakers(
    num_speakers: int | None,
) -> None:
    pytest.importorskip("sklearn")
    np = pytest.importorskip("numpy")
    pipeline = importlib.import_module("lectern._vendor.diarize")
    types = importlib.import_module("lectern._vendor.diarize.utils")
    # Two Gaussian blobs in 256 dimensions, six windows per alternating turn.
    # Sustained turns also exercise upstream temporal smoothing within one VAD segment.
    rng = np.random.default_rng(42)
    means = np.zeros((2, 256), dtype=np.float32)
    means[0, 0] = 8.0
    means[1, 1] = 8.0
    labels = (np.arange(60) // 6) % 2
    embeddings = means[labels] + rng.normal(0.0, 0.1, (60, 256))
    subsegments = [
        types.SubSegment(start=i * 0.6, end=i * 0.6 + 1.2, parent_idx=0) for i in range(60)
    ]
    duration = subsegments[-1].end
    turns = pipeline.diarize_embeddings(
        [types.SpeechSegment(start=0.0, end=duration)],
        embeddings,
        subsegments,
        duration,
        num_speakers,
    )
    assert len({turn.speaker for turn in turns}) == 2
    assert len(turns) == 10
    assert [turn.speaker for turn in turns] == [turns[i % 2].speaker for i in range(10)]
    boundaries = [0.0, *(i * 3.6 + 0.3 for i in range(1, 10)), duration]
    np.testing.assert_allclose([turn.start for turn in turns], boundaries[:-1])
    np.testing.assert_allclose([turn.end for turn in turns], boundaries[1:])
