"""Full-segment recovery through the accepted internal seam, not a new backend.

The mutation rows catch stale-hit substitution, text-only identity, conflating
unavailable evidence with no match, or treating library status as admission.
The qualifier control catches using a bounded snippet as the cited full text.
Consumer/transport requirements below are explicitly skipped, not simulated.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pytest

from lectern.automation import open_state
from lectern.bundle import TranscriptSegmentsDocument
from lectern.records import AutomationError, LibraryStatus

FIXTURES = Path(__file__).resolve().parent / "fixtures"
MATRIX_DIR = FIXTURES / "recovery_matrix"
MATRIX: dict[str, Any] = json.loads((MATRIX_DIR / "matrix.json").read_text(encoding="utf-8"))
QUERY: str = MATRIX["query"]
LITERAL: bool = MATRIX["literal"]
LIMIT: int = MATRIX["limit"]
ABSENT_QUERY: str = MATRIX["absent_query"]
QUALIFIER: str = MATRIX["qualifier"]


@dataclass(frozen=True)
class RecoveryRow:
    name: str
    mutation: str
    current_fixture: str | None
    held_error: str | None
    status: str
    diagnostic: str
    start_s: float | None


ROWS = [RecoveryRow(**row) for row in MATRIX["rows"]]


def _passage(fixture: str) -> str:
    # Expectations come from authored fixtures, never from a response or index.
    document = TranscriptSegmentsDocument.model_validate_json(
        (MATRIX_DIR / fixture).read_bytes(), strict=True
    )
    return next(segment.text for segment in document.root if segment.id == 1)


def _write_selected(bundle: Path, payload: bytes, *, declare: bool = True) -> None:
    """Controlled test mutation; this is not a correction or revision workflow."""

    (bundle / "transcript" / "segments.json").write_bytes(payload)
    if not declare:
        return
    manifest_path = bundle / "manifest.json"
    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    for stage in manifest["stages"].values():
        for output in stage["outputs"]:
            if output["path"] == "transcript/segments.json":
                output["sha256"] = hashlib.sha256(payload).hexdigest()
                output["bytes"] = len(payload)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def _archive(tmp_path: Path) -> tuple[Path, Path]:
    media = tmp_path / "synthetic_talk.wav"
    media.write_bytes((FIXTURES / "synthetic_talk.wav").read_bytes())
    media.with_suffix(".transcript.txt").write_bytes(
        (FIXTURES / "synthetic_talk.transcript.txt").read_bytes()
    )
    state_path = tmp_path / "state.sqlite"
    with open_state(state_path) as state:
        bundle = state.ingest_one_shot(media, tmp_path / "bundles").bundle_dir
    _write_selected(bundle, (MATRIX_DIR / "original_segments.json").read_bytes())
    return state_path, bundle


def _mutate(bundle: Path, row: RecoveryRow) -> None:
    selected = bundle / "transcript" / "segments.json"
    if row.mutation == "none":
        return
    if row.mutation == "replace":
        assert row.current_fixture is not None
        _write_selected(bundle, (MATRIX_DIR / row.current_fixture).read_bytes())
    elif row.mutation == "remove":
        selected.unlink()
    elif row.mutation == "malform":
        # Declare the malformed bytes so this isolates parsing, not a hash error.
        _write_selected(bundle, b"{invalid synthetic json")
    elif row.mutation == "undeclared":
        _write_selected(bundle, (MATRIX_DIR / "drifting_segments.json").read_bytes(), declare=False)
    elif row.mutation == "damage-metadata":
        metadata = bundle / "transcript" / "metadata.json"
        metadata.write_bytes(metadata.read_bytes() + b" ")
    else:
        raise AssertionError(f"unknown fixture mutation: {row.mutation}")


@pytest.mark.parametrize("row", ROWS, ids=lambda row: row.name)
@pytest.mark.parametrize("mode", ["retained", "reopened"])
def test_full_passage_recovery_matrix(tmp_path: Path, row: RecoveryRow, mode: str) -> None:
    """A held selection either cites its full A passage or explicitly refuses."""

    state_path, bundle = _archive(tmp_path)
    original_text = _passage("original_segments.json")
    with open_state(state_path) as retained:
        before = retained.search_observed(QUERY, literal=LITERAL, limit=LIMIT)
        assert len(before.hits) == 1
        assert before.nonready == []
        held = before.hits[0]
        assert held.hit.segment_id == 1
        held_projection = asdict(held)
        original_anchor, original = retained.cite_observed(held)
        assert original.current_text == original_text
        assert original.start_s == original_anchor.start_s == 0.4
        assert original.outcome.value == "exact"
        empty = retained.search_observed(ABSENT_QUERY)
        assert empty.hits == [] and empty.nonready == []

        _mutate(bundle, row)
        # The reopened control receives the very same held A selection. The
        # retained mode must reconcile without requiring a new store instance.
        if mode == "reopened":
            retained.close()
        context = open_state(state_path) if mode == "reopened" else nullcontext(retained)
        with context as reader:
            if row.held_error is None:
                anchor, resolved = reader.cite_observed(held)
                assert anchor == original_anchor
                assert resolved.current_text == original_text
                assert resolved.outcome.value == "exact"
            else:
                with pytest.raises(AutomationError, match=row.held_error) as refused:
                    reader.cite_observed(held)
                assert str(tmp_path) not in str(refused.value)

            diagnostic = reader.resolve_anchor(original_anchor)
            assert diagnostic.outcome.value == row.diagnostic
            assert diagnostic.start_s == row.start_s
            # Diagnostic readability does not grant strict admission.
            if row.mutation == "undeclared":
                assert diagnostic.current_text == _passage("drifting_segments.json")
            elif row.current_fixture is None:
                assert diagnostic.current_text is None
            else:
                assert diagnostic.current_text == _passage(row.current_fixture)

            result = reader.search_observed(QUERY, literal=LITERAL, limit=LIMIT)
            expected_nonready = (
                [] if row.status == "ready" else [(bundle.name, LibraryStatus.NEEDS_REPROCESSING)]
            )
            assert [(item.bundle_id, item.status) for item in result.nonready] == expected_nonready
            assert reader.list_library()[0].status.value == row.status
            no_match = reader.search_observed(ABSENT_QUERY)
            assert no_match.hits == []
            assert no_match.nonready == result.nonready
            if row.current_fixture is None:
                assert result.hits == []
            else:
                assert len(result.hits) == 1
                candidate = result.hits[0]
                assert candidate.hit.bundle_id == held.hit.bundle_id
                assert candidate.hit.segment_id == held.hit.segment_id
                # Both the retiming and the wording edit are invisible in this
                # excerpt. An identical snippet must not stand in for identity.
                assert candidate.hit.snippet == held.hit.snippet
                assert (candidate.observation == held.observation) is (row.held_error is None)
                anchor, full = reader.cite_observed(candidate)
                assert full.outcome.value == "exact"
                assert full.current_text == _passage(row.current_fixture)
                assert full.start_s == anchor.start_s == row.start_s
                assert full.current_text != candidate.hit.snippet
                if row.name == "retimed":
                    assert full.current_text == original_text
                    assert diagnostic.outcome.value == "exact"
                    assert anchor != original_anchor
                if row.name == "drifting":
                    assert full.current_text != original_text
                    assert anchor.text_sha256 != original_anchor.text_sha256
                assert str(tmp_path) not in json.dumps(asdict(full))

            # Neither refusal nor candidate inspection overwrites the held hit.
            assert asdict(held) == held_projection
            assert str(tmp_path) not in json.dumps(asdict(result))


@pytest.mark.parametrize("mode", ["retained", "reopened"])
def test_out_of_snippet_qualifier_is_in_the_full_cited_passage(tmp_path: Path, mode: str) -> None:
    """Exact citation is compatible with a misleading snippet-level claim."""

    state_path, _ = _archive(tmp_path)
    with open_state(state_path) as retained:
        held = retained.search_observed(QUERY, literal=LITERAL, limit=LIMIT).hits[0]
        # Actual omission/presence is the oracle, not merely passage length.
        assert "intervention reduced errors" in held.hit.snippet
        assert QUALIFIER not in held.hit.snippet
        assert "field trial" not in held.hit.snippet
        if mode == "reopened":
            retained.close()
        context = open_state(state_path) if mode == "reopened" else nullcontext(retained)
        with context as reader:
            anchor, full = reader.cite_observed(held)
            assert full.outcome.value == "exact"
            assert full.current_text is not None
            assert full.current_text == _passage("original_segments.json")
            assert QUALIFIER in full.current_text
            assert anchor.segment_id == 1
            # Full passage here means the selected segment, not adjacent text
            # or all recording/visual context.
            assert "The next section" not in full.current_text
            assert "This invented talk introduces" not in full.current_text


@pytest.mark.skip(
    reason="M5b MCP/API adapter and recovery consumer do not exist; no transport contract adopted"
)
@pytest.mark.parametrize("row", ROWS, ids=lambda row: row.name)
def test_read_surface_recovery_preserves_question_and_explicit_reassociation(
    row: RecoveryRow,
) -> None:
    """Reserved workflow acceptance, NOT evidence from a test-only controller.

    Run each matrix row through the eventual read surface. Keep the original
    question, query/mode/limit and optional claim when a candidate is inspected
    or declined. Carry the held observation unchanged; do not fall back to
    unobserved citation. Expose the candidate's full passage before an explicit
    reader decision, and permit leaving the support relationship unresolved.
    The fixture question/claim are illustrative, not adopted response fields.
    """

    raise NotImplementedError(f"wire {row.name} to the adopted read consumer")


@pytest.mark.skip(
    reason="No full-passage review/reassociation UI exists; citation success is not endorsement"
)
def test_qualifier_must_be_exposed_before_support_confirmation() -> None:
    """Reserved: confirmation cannot rely only on the misleading search excerpt.

    The consumer must expose the full qualifier-bearing passage, require a
    deliberate review action, keep the tentative field-errors claim visible,
    and allow qualification, rejection or no assessment. A click would test
    an affordance, not establish reader comprehension or a product preference.
    """

    raise NotImplementedError("wire the qualifier scenario to the adopted review affordance")
