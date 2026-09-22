"""One local reader carries the evidence it found into a later citation."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pytest

from lectern import evidence as evidence_module
from lectern.automation import open_state
from lectern.records import AutomationError, LibraryStatus
from lectern.search import AnchorResolution

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _archive(tmp_path: Path) -> tuple[Path, Path]:
    media = tmp_path / "synthetic_talk.wav"
    media.write_bytes((FIXTURES / "synthetic_talk.wav").read_bytes())
    media.with_suffix(".transcript.txt").write_bytes(
        (FIXTURES / "synthetic_talk.transcript.txt").read_bytes()
    )
    state_path = tmp_path / "state.sqlite"
    with open_state(state_path) as state:
        bundle = state.ingest_one_shot(media, tmp_path / "bundles").bundle_dir
    return state_path, bundle


def _replace_selected_text(bundle: Path, text: str) -> None:
    selected = bundle / "transcript" / "segments.json"
    segments: list[dict[str, Any]] = json.loads(selected.read_text(encoding="utf-8"))
    segments[0]["text"] = text
    _write_selected_segments(bundle, segments)


def _write_selected_segments(bundle: Path, segments: list[dict[str, Any]]) -> None:
    selected = bundle / "transcript" / "segments.json"
    selected.write_text(json.dumps(segments), encoding="utf-8")
    payload = selected.read_bytes()
    manifest_path = bundle / "manifest.json"
    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    for stage in manifest["stages"].values():
        for output in stage["outputs"]:
            if output["path"] == "transcript/segments.json":
                output["sha256"] = hashlib.sha256(payload).hexdigest()
                output["bytes"] = len(payload)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_observed_hit_cites_the_same_selected_evidence_or_refuses_drift(tmp_path: Path) -> None:
    """A saved A result must never become a new exact citation to valid B."""

    state_path, bundle = _archive(tmp_path)
    with open_state(state_path) as state:
        first = state.search_observed("knowledge bundles")
        assert len(first.hits) == 1
        saved = first.hits[0]
        assert saved.hit.snippet == (
            "Lectern turns recorded talks into local inspectable knowledge bundles. "
            "This synthetic fixture contains no private or copyrighted media."
        )
        _, initial = state.cite_observed(saved)
        assert initial.outcome is AnchorResolution.EXACT

        _replace_selected_text(bundle, "new synthetic passage about questions")
        with pytest.raises(AutomationError, match="evidence changed"):
            state.cite_observed(saved)

        assert state.search_observed("knowledge bundles").hits == []
        replacement = state.search_observed("new synthetic passage").hits[0]
        _, current = state.cite_observed(replacement)
        assert current.outcome is AnchorResolution.EXACT
        assert current.current_text == "new synthetic passage about questions"

    with open_state(state_path) as reopened:
        with pytest.raises(AutomationError, match="evidence changed"):
            reopened.cite_observed(saved)
        assert reopened.search_observed("new synthetic passage").hits


@pytest.mark.parametrize("damage", ["missing", "malformed"])
def test_observed_search_reports_nonready_status_separately_from_no_match(
    tmp_path: Path, damage: str
) -> None:
    """A missing or broken transcript is not evidence that a phrase is absent."""

    state_path, bundle = _archive(tmp_path)
    with open_state(state_path) as state:
        saved = state.search_observed("knowledge bundles").hits[0]
        selected = bundle / "transcript" / "segments.json"
        if damage == "missing":
            selected.unlink()
        else:
            selected.write_text("{invalid json", encoding="utf-8")

        result = state.search_observed("knowledge bundles")
        assert result.hits == []
        assert [(item.bundle_id, item.status) for item in result.nonready] == [
            (bundle.name, LibraryStatus.NEEDS_REPROCESSING)
        ]
        with pytest.raises(AutomationError, match="no readable transcript"):
            state.cite_observed(saved)

    with open_state(state_path) as reopened:
        result = reopened.search_observed("knowledge bundles")
        assert result.hits == []
        assert [(item.bundle_id, item.status) for item in result.nonready] == [
            (bundle.name, LibraryStatus.NEEDS_REPROCESSING)
        ]


def test_observed_search_refuses_indexed_a_if_evidence_becomes_valid_b_after_refresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A new digest read after an old FTS hit must not bless that hit."""

    state_path, bundle = _archive(tmp_path)
    with open_state(state_path) as state:
        refresh = state.refresh_changed_bundles

        def change_after_refresh() -> list[str]:
            result = refresh()
            _replace_selected_text(bundle, "new synthetic passage about questions")
            return result

        monkeypatch.setattr(state, "refresh_changed_bundles", change_after_refresh)
        with pytest.raises(AutomationError, match="evidence changed during read"):
            state.search_observed("knowledge bundles")


def test_observed_search_refuses_false_operator_hit_from_changed_index_body(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A displayed A segment with a forged FTS body must not become a valid hit."""

    state_path, bundle = _archive(tmp_path)
    with open_state(state_path) as state:
        refresh = state.refresh_changed_bundles

        def corrupt_after_refresh() -> list[str]:
            result = refresh()
            # Same-connection commit leaves PRAGMA data_version unchanged, so
            # only validation of the query-bearing body can reject this hit.
            connection = state._connection  # pyright: ignore[reportPrivateUsage]
            connection.execute(
                "UPDATE segment_index SET body = 'forgedterm' WHERE bundle_id = ?",
                (bundle.name,),
            )
            connection.commit()
            return result

        monkeypatch.setattr(state, "refresh_changed_bundles", corrupt_after_refresh)
        with pytest.raises(AutomationError, match="index changed during read"):
            state.search_observed("forgedterm", literal=False)


def test_observed_search_checks_selected_bytes_when_indexed_words_stay_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retiming after refresh leaves FTS columns equal but changes the observation."""

    state_path, bundle = _archive(tmp_path)
    with open_state(state_path) as state:
        refresh = state.refresh_changed_bundles

        def retime_after_refresh() -> list[str]:
            result = refresh()
            selected = bundle / "transcript" / "segments.json"
            segments: list[dict[str, Any]] = json.loads(selected.read_text(encoding="utf-8"))
            segments[0]["start_s"] = 0.25
            _write_selected_segments(bundle, segments)
            return result

        monkeypatch.setattr(state, "refresh_changed_bundles", retime_after_refresh)
        with pytest.raises(AutomationError, match="evidence changed during read"):
            state.search_observed("knowledge bundles")


def test_observed_search_refuses_no_match_after_another_writer_drops_index_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A concurrent index deletion must not become a definitive empty search."""

    state_path, bundle = _archive(tmp_path)
    with open_state(state_path) as state:
        refresh = state.refresh_changed_bundles

        def delete_after_refresh() -> list[str]:
            result = refresh()
            with sqlite3.connect(state_path) as connection:
                connection.execute("DELETE FROM segment_index WHERE bundle_id = ?", (bundle.name,))
            return result

        monkeypatch.setattr(state, "refresh_changed_bundles", delete_after_refresh)
        with pytest.raises(AutomationError, match="index changed during read"):
            state.search_observed("knowledge bundles")


@pytest.mark.parametrize(
    ("damage", "status"),
    [
        ("missing-source", LibraryStatus.NEEDS_REPROCESSING),
        ("malformed-source", LibraryStatus.NEEDS_REPROCESSING),
        ("unsupported-manifest", LibraryStatus.INCOMPLETE),
    ],
)
def test_observed_read_refuses_damaged_source_or_unsupported_manifest(
    tmp_path: Path, damage: str, status: LibraryStatus
) -> None:
    state_path, bundle = _archive(tmp_path)
    with open_state(state_path) as state:
        saved = state.search_observed("knowledge bundles").hits[0]
        if damage == "missing-source":
            (bundle / "source.json").unlink()
        elif damage == "malformed-source":
            (bundle / "source.json").write_text("{invalid json", encoding="utf-8")
        else:
            manifest_path = bundle / "manifest.json"
            manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["schema_version"] = "99.0"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        result = state.search_observed("knowledge bundles")
        assert result.hits == []
        assert [(item.bundle_id, item.status) for item in result.nonready] == [
            (bundle.name, status)
        ]
        with pytest.raises(AutomationError, match="no readable transcript"):
            state.cite_observed(saved)

    with open_state(state_path) as reopened:
        assert reopened.search_observed("knowledge bundles").hits == []


def test_observation_refuses_same_words_with_changed_timing(tmp_path: Path) -> None:
    """A text-only token would silently keep a citation after retiming."""

    state_path, bundle = _archive(tmp_path)
    with open_state(state_path) as state:
        saved = state.search_observed("knowledge bundles").hits[0]
        selected = bundle / "transcript" / "segments.json"
        segments: list[dict[str, Any]] = json.loads(selected.read_text(encoding="utf-8"))
        segments[0]["start_s"] = 0.25
        _write_selected_segments(bundle, segments)

        with pytest.raises(AutomationError, match="evidence changed"):
            state.cite_observed(saved)
        current = state.search_observed("knowledge bundles").hits[0]
        assert current.observation != saved.observation
        anchor, resolved = state.cite_observed(current)
        assert anchor.start_s == 0.25
        assert resolved.outcome is AnchorResolution.EXACT


def test_same_bytes_with_a_new_selected_pointer_change_the_observation(tmp_path: Path) -> None:
    """The text candidate can stay indexed while the validated selection changes."""

    state_path, bundle = _archive(tmp_path)
    with open_state(state_path) as state:
        saved = state.search_observed("knowledge bundles").hits[0]
        original = bundle / "transcript" / "segments.json"
        alternate = bundle / "transcript" / "alternate.json"
        alternate.write_bytes(original.read_bytes())
        source_path = bundle / "source.json"
        source: dict[str, Any] = json.loads(source_path.read_text(encoding="utf-8"))
        source["transcript"]["segments"] = "transcript/alternate.json"
        source_path.write_text(json.dumps(source), encoding="utf-8")
        source_payload = source_path.read_bytes()
        manifest_path = bundle / "manifest.json"
        manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
        for stage in manifest["stages"].values():
            for output in stage["outputs"]:
                if output["path"] == "source.json":
                    output["sha256"] = hashlib.sha256(source_payload).hexdigest()
                    output["bytes"] = len(source_payload)
                elif output["path"] == "transcript/segments.json":
                    output["path"] = "transcript/alternate.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        with pytest.raises(AutomationError, match="evidence changed"):
            state.cite_observed(saved)
        current = state.search_observed("knowledge bundles").hits[0]
        assert current.observation != saved.observation
        _, resolved = state.cite_observed(current)
        assert resolved.outcome is AnchorResolution.EXACT


def test_observed_read_preserves_authored_path_text_without_emitting_media_location(
    tmp_path: Path,
) -> None:
    media = tmp_path / "synthetic_talk.wav"
    media.write_bytes((FIXTURES / "synthetic_talk.wav").read_bytes())
    media.with_suffix(".transcript.txt").write_text(
        "The speaker wrote /authored/example.wav on the board.", encoding="utf-8"
    )
    state_path = tmp_path / "state.sqlite"
    with open_state(state_path) as state:
        state.ingest_one_shot(media, tmp_path / "bundles")
        result = state.search_observed("authored")
        assert "/authored/example.wav" in result.hits[0].hit.snippet
        _, citation = state.cite_observed(result.hits[0])
        assert "/authored/example.wav" in str(citation.current_text)
        assert str(tmp_path) not in json.dumps(asdict(result))
        assert str(tmp_path) not in str(citation.current_text)


def test_observation_cannot_be_reused_for_another_segment_in_the_same_document(
    tmp_path: Path,
) -> None:
    """A bundle-level token alone permits a selected hit to cite a different row."""

    state_path, bundle = _archive(tmp_path)
    selected = bundle / "transcript" / "segments.json"
    segments: list[dict[str, Any]] = json.loads(selected.read_text(encoding="utf-8"))
    segments.append(
        {
            "id": 1,
            "start_s": 1.0,
            "end_s": 2.0,
            "text": "second independent evidence",
            "source": "fixture",
        }
    )
    _write_selected_segments(bundle, segments)
    with open_state(state_path) as state:
        saved = state.search_observed("knowledge bundles").hits[0]
        changed_selection = replace(saved, hit=replace(saved.hit, segment_id=1))
        with pytest.raises(AutomationError, match="evidence changed"):
            state.cite_observed(changed_selection)


def test_citation_mints_from_the_bytes_it_compared_even_if_files_change_after_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second file load after expectation comparison would substitute B."""

    state_path, bundle = _archive(tmp_path)
    with open_state(state_path) as state:
        saved = state.search_observed("knowledge bundles").hits[0]
        read_payload = evidence_module._contained_payload  # pyright: ignore[reportPrivateUsage]
        changed = False

        def replace_after_selected_payload(root: Path, relative: str) -> bytes:
            nonlocal changed
            payload = read_payload(root, relative)
            if relative == "transcript/segments.json" and not changed:
                changed = True
                _replace_selected_text(bundle, "new synthetic passage about questions")
            return payload

        monkeypatch.setattr(evidence_module, "_contained_payload", replace_after_selected_payload)
        anchor, resolved = state.cite_observed(saved)
        assert changed
        assert resolved.outcome is AnchorResolution.EXACT
        assert resolved.current_text == saved.hit.snippet
        assert anchor.text_sha256 != state.cite_segment(bundle.name, 0)[0].text_sha256


def test_observed_search_refuses_a_poisoned_index_id_without_echoing_its_location(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A retained-store row changed after refresh cannot leak a path via int()."""

    state_path, bundle = _archive(tmp_path)
    poisoned = "/private/synthetic-media.wav"
    with open_state(state_path) as state:
        refresh = state.refresh_changed_bundles

        def poison_after_refresh() -> list[str]:
            result = refresh()
            with sqlite3.connect(state_path) as connection:
                connection.execute(
                    "UPDATE segment_index SET segment_id = ? WHERE bundle_id = ?",
                    (poisoned, bundle.name),
                )
            return result

        monkeypatch.setattr(state, "refresh_changed_bundles", poison_after_refresh)
        with pytest.raises(AutomationError, match="index changed during read") as caught:
            state.search_observed("knowledge bundles")
        assert poisoned not in str(caught.value)


def test_nonready_status_can_coexist_with_a_strictly_readable_hit(
    tmp_path: Path,
) -> None:
    """An unrelated declared artifact can damage readiness while citation stays valid."""

    state_path, bundle = _archive(tmp_path)
    metadata = bundle / "transcript" / "metadata.json"
    metadata.write_text(metadata.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with open_state(state_path) as state:
        result = state.search_observed("knowledge bundles")
        assert len(result.hits) == 1
        assert [(item.bundle_id, item.status) for item in result.nonready] == [
            (bundle.name, LibraryStatus.NEEDS_REPROCESSING)
        ]
        _, citation = state.cite_observed(result.hits[0])
        assert citation.outcome is AnchorResolution.EXACT


def test_observed_read_errors_do_not_echo_a_poisoned_bundle_location(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_path, bundle = _archive(tmp_path)
    poisoned = "/private/synthetic-media.wav"
    with open_state(state_path) as state:
        refresh = state.refresh_changed_bundles

        def poison_after_refresh() -> list[str]:
            result = refresh()
            with sqlite3.connect(state_path) as connection:
                connection.execute(
                    "UPDATE segment_index SET bundle_id = ? WHERE bundle_id = ?",
                    (poisoned, bundle.name),
                )
            return result

        monkeypatch.setattr(state, "refresh_changed_bundles", poison_after_refresh)
        with pytest.raises(AutomationError, match="evidence changed during read") as caught:
            state.search_observed("knowledge bundles")
        assert poisoned not in str(caught.value)
