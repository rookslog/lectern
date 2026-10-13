"""`library search`: what a person gets back when they ask their archive a question.

The literal-by-default decision is the one worth defending here. FTS5 has a real
query grammar, and passing user text to it raw means a remembered phrase like
`C++ discussion` raises `fts5: syntax error near "+"` instead of returning the
talk it appears in. A search box that answers a question with a parser error has
failed at retrieval, not taught the user about syntax.
"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path

import pytest
from pytest import CaptureFixture

from lectern import cli, search
from lectern.automation import open_state

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
SYNTHETIC_TALK = FIXTURE_DIR / "synthetic_talk.wav"
SYNTHETIC_TRANSCRIPT = FIXTURE_DIR / "synthetic_talk.transcript.txt"
MULTISCRIPT = json.loads((FIXTURE_DIR / "multiscript_segments.json").read_text(encoding="utf-8"))[
    "cases"
]


def _archive(tmp_path: Path) -> Path:
    media_dir = tmp_path / "media"
    media_dir.mkdir(parents=True, exist_ok=True)
    media = media_dir / "synthetic_talk.wav"
    media.write_bytes(SYNTHETIC_TALK.read_bytes())
    media.with_suffix(".transcript.txt").write_text(
        SYNTHETIC_TRANSCRIPT.read_text(encoding="utf-8"), encoding="utf-8"
    )
    state_path = tmp_path / "state.sqlite"
    assert (
        cli.main(
            [
                "ingest",
                str(media),
                "--output",
                str(tmp_path / "bundles"),
                "--state",
                str(state_path),
            ]
        )
        == 0
    )
    return state_path


def test_search_returns_matching_bundles(tmp_path: Path) -> None:
    state_path = _archive(tmp_path)
    with open_state(state_path) as state:
        hits = state.search_segments("knowledge bundles")
    assert hits
    assert all(hit.bundle_id.startswith("synthetic-talk-") for hit in hits)


def test_search_returns_nothing_for_absent_text(tmp_path: Path) -> None:
    # A search that matched everything would be as useless as one that matched
    # nothing, and much harder to notice.
    state_path = _archive(tmp_path)
    with open_state(state_path) as state:
        assert state.search_segments("kangaroo jurisprudence") == []


@pytest.mark.parametrize(
    "query",
    ["C++ discussion", 'unmatched " quote', "alpha OR beta", "NEAR(a b)", "wild*", "a-b"],
)
def test_search_treats_operators_as_literal_text(tmp_path: Path, query: str) -> None:
    """Grammar characters must be data, not syntax.

    Asserted as "does not raise" rather than "returns a hit": the fixture does
    not contain these strings, so the property under test is that the query is
    answered at all.
    """

    state_path = _archive(tmp_path)
    with open_state(state_path) as state:
        assert state.search_segments(query) == []


def test_search_operator_mode_is_reachable(tmp_path: Path) -> None:
    # Advanced syntax stays available; it is simply not what an unqualified
    # query means.
    state_path = _archive(tmp_path)
    with open_state(state_path) as state:
        assert state.search_segments("knowledge OR kangaroo", literal=False)


def test_literal_search_supports_casefold_expansion_without_breaking_operator_mode(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "state.sqlite"
    with open_state(state_path) as state:
        state.index_synthetic_segment("casefold-expansion", 0, "Die Straße bleibt")
        literal_hits = state.search_segments("STRASSE")
        operator_hits = state.search_segments("Straße", literal=False)
        false_phrase_hits = state.search_segments('"bleibt die"', literal=False)
        literal_column_hits = state.search_segments('literal:"bleibt die"', literal=False)

    assert [hit.bundle_id for hit in literal_hits] == ["casefold-expansion"]
    assert [hit.bundle_id for hit in operator_hits] == ["casefold-expansion"]
    assert false_phrase_hits == []
    assert literal_column_hits == []


def test_search_operator_mode_surfaces_bad_syntax(tmp_path: Path) -> None:
    """Opting into the grammar means opting into its errors, reported clearly."""

    state_path = _archive(tmp_path)
    with open_state(state_path) as state, pytest.raises(ValueError, match="search query"):
        state.search_segments("C++", literal=False)


def test_search_operator_mode_rejects_unsegmented_scripts_explicitly(tmp_path: Path) -> None:
    state_path = _archive(tmp_path)
    with (
        open_state(state_path) as state,
        pytest.raises(ValueError, match="operator-mode search does not support"),
    ):
        state.search_segments("现象 OR 哲学", literal=False)


def test_literal_filter_fills_the_requested_limit_after_false_candidates(tmp_path: Path) -> None:
    state_path = _archive(tmp_path)
    with open_state(state_path) as state:
        for index in range(51):
            state.index_synthetic_segment(f"false-{index:02d}", 0, "C discussion")
        state.index_synthetic_segment("exact", 0, "C++ discussion")
        hits = state.search_segments("C++ discussion", limit=1)
    assert [hit.bundle_id for hit in hits] == ["exact"]


def test_search_snippet_is_bounded_around_the_literal_match(tmp_path: Path) -> None:
    state_path = tmp_path / "state.sqlite"
    display = f"{'before ' * 100}needle phrase{' after' * 100}"
    with open_state(state_path) as state:
        state.index_synthetic_segment("long-segment", 0, display)
        hit = state.search_segments("needle phrase")[0]
    assert "needle phrase" in hit.snippet
    assert len(hit.snippet) <= 243
    assert hit.snippet != display


def test_literal_snippet_centers_a_canonically_equivalent_match(tmp_path: Path) -> None:
    state_path = tmp_path / "state.sqlite"
    display = f"{'before ' * 100}café   phrase{' after' * 100}"
    with open_state(state_path) as state:
        state.index_synthetic_segment("canonical-snippet", 0, display)
        hit = state.search_segments("cafe\u0301 phrase")[0]
    assert "café" in hit.snippet


def test_literal_snippet_offset_survives_a_long_decomposed_prefix(tmp_path: Path) -> None:
    state_path = tmp_path / "state.sqlite"
    display = ("e\u0301" * 200) + ".needle" + (" after" * 50)
    with open_state(state_path) as state:
        state.index_synthetic_segment("decomposed-prefix", 0, display)
        hit = state.search_segments("needle")[0]
    assert "needle" in hit.snippet


def test_literal_snippet_offset_survives_decomposed_hangul(tmp_path: Path) -> None:
    state_path = tmp_path / "state.sqlite"
    decomposed = unicodedata.normalize("NFD", "현상학")
    display = ("before " * 100) + decomposed + (" after" * 50)
    with open_state(state_path) as state:
        state.index_synthetic_segment("decomposed-hangul", 0, display)
        hit = state.search_segments("현상학")[0]
    assert decomposed in hit.snippet


def test_operator_search_snippet_contains_an_actual_matching_term(tmp_path: Path) -> None:
    state_path = tmp_path / "state.sqlite"
    display = f"{'before ' * 100}needle{' after' * 100}"
    with open_state(state_path) as state:
        state.index_synthetic_segment("long-operator-segment", 0, display)
        hit = state.search_segments("needle OR absent", literal=False)[0]
    assert "needle" in hit.snippet
    assert len(hit.snippet) <= 243


@pytest.mark.parametrize("selector", ["body", "{body}", "{body display}"])
def test_operator_column_selector_is_not_used_as_the_snippet_match(
    tmp_path: Path, selector: str
) -> None:
    state_path = tmp_path / "state.sqlite"
    display = "body " + ("filler " * 100) + "needle"
    with open_state(state_path) as state:
        state.index_synthetic_segment("column-selector", 0, display)
        hit = state.search_segments(f"{selector}:needle", literal=False)[0]
    assert "needle" in hit.snippet


@pytest.mark.parametrize("case", MULTISCRIPT, ids=[c["script"] for c in MULTISCRIPT])
def test_search_retrieves_every_supported_script(tmp_path: Path, case: dict[str, str]) -> None:
    """Including the two-character CJK queries no tokenizer handles unaided."""

    state_path = _archive(tmp_path)
    with open_state(state_path) as state:
        state.index_synthetic_segment("synthetic-script", 0, case["text"])
        hits = state.search_segments(case["query"])
    assert any(hit.bundle_id == "synthetic-script" for hit in hits), (
        f"{case['script']}: {case['query']!r} did not retrieve its own segment"
    )


def test_search_retrieves_supplementary_cjk_ideographs(tmp_path: Path) -> None:
    state_path = tmp_path / "state.sqlite"
    with open_state(state_path) as state:
        state.index_synthetic_segment("extension-b", 0, "𠀀𠀁𠀂")
        assert state.search_segments("𠀀𠀁")
        with pytest.raises(ValueError, match="operator-mode search does not support"):
            state.search_segments("𠀀 OR 𠀁", literal=False)

    with open_state(state_path) as state:
        state.index_synthetic_segment("extension-i", 0, "\U0002ebf0\U0002ebf1\U0002ebf2")
        assert state.search_segments("\U0002ebf0\U0002ebf1")


def test_search_retrieves_unicode_17_extension_j_ideographs(tmp_path: Path) -> None:
    state_path = tmp_path / "state.sqlite"
    with open_state(state_path) as state:
        state.index_synthetic_segment("extension-j", 0, "\U000323b0\U000323b1\U000323b2")
        assert state.search_segments("\U000323b0\U000323b1")


def test_search_retrieves_half_width_katakana(tmp_path: Path) -> None:
    state_path = tmp_path / "state.sqlite"
    with open_state(state_path) as state:
        state.index_synthetic_segment("half-width-katakana", 0, "ﾃｽﾄ")
        assert state.search_segments("ﾃｽ")
        with pytest.raises(ValueError, match="operator-mode search does not support"):
            state.search_segments("ﾃ OR ｽ", literal=False)


def test_search_hit_carries_an_anchorable_reference(tmp_path: Path) -> None:
    state_path = _archive(tmp_path)
    with open_state(state_path) as state:
        hit = state.search_segments("knowledge bundles")[0]
    assert hit.bundle_id
    assert hit.segment_id is not None


def test_cli_search_emits_no_filesystem_path(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    state_path = _archive(tmp_path)
    # `_archive` ingests through the CLI, which prints the bundle directory.
    # Dropping that output is what makes the assertions below about `search`
    # rather than about everything the fixture happened to emit.
    capsys.readouterr()
    assert cli.main(["library", "search", "knowledge", "--state", str(state_path), "--json"]) == 0
    output = capsys.readouterr().out
    assert json.loads(output)["results"]
    assert str(tmp_path) not in output


def test_cli_search_reports_no_matches_without_failing(
    tmp_path: Path, capsys: CaptureFixture[str]
) -> None:
    # An empty result is an answer, not an error: exiting non-zero here would
    # make "nothing found" indistinguishable from "search is broken".
    state_path = _archive(tmp_path)
    # `_archive` ingests through the CLI, which prints the bundle directory.
    # Dropping that output is what makes the assertions below about `search`
    # rather than about everything the fixture happened to emit.
    capsys.readouterr()
    assert cli.main(["library", "search", "kangaroo", "--state", str(state_path), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["results"] == []


def test_query_and_index_agree_on_segmentation() -> None:
    """The invariant object identity was standing in for, at the query boundary."""

    for case in MULTISCRIPT:
        assert search.segment_text(case["query"]) in search.segment_text(case["text"])


@pytest.mark.parametrize("text", ["Ꞵeta", "Ϳota", "Ԩame", "Straße", "CAFÉ"])
@pytest.mark.parametrize("fold_query", [False, True])
def test_literal_search_retrieves_exact_and_casefolded_text(
    tmp_path: Path, text: str, fold_query: bool
) -> None:
    with open_state(tmp_path / "state.sqlite") as state:
        state.index_synthetic_segment("changed-fold", 0, text)
        query = text.casefold() if fold_query else text
        assert search.text_contains_literal(text, query)
        assert [hit.bundle_id for hit in state.search_segments(query)] == ["changed-fold"]


@pytest.mark.parametrize("field", ["snippet", "bundle_id"])
def test_plain_search_escapes_row_controls_without_changing_evidence(
    tmp_path: Path, capsys: CaptureFixture[str], field: str
) -> None:
    controls = "".join(chr(code) for code in [*range(32), *range(127, 160), 0x2028, 0x2029])
    multilingual = "日本語 العربية עברית"
    raw = f"needle{controls}{multilingual}"
    bundle_id = raw if field == "bundle_id" else "plain-bundle"
    snippet = raw if field == "snippet" else f"needle {multilingual}"
    state_path = tmp_path / "state.sqlite"
    with open_state(state_path) as state:
        state.index_synthetic_segment(bundle_id, 0, snippet)
        hit = state.search_segments("needle")[0]
        assert hit.bundle_id == bundle_id
        assert hit.snippet == snippet

    args = ["library", "search", "needle", "--state", str(state_path)]
    assert cli.main(args) == 0
    plain = capsys.readouterr().out
    assert len(plain.splitlines()) == 1
    assert plain.count("\t") == 3
    rendered_fields = plain.removesuffix("\n").split("\t")
    rendered = rendered_fields[0 if field == "bundle_id" else 3]
    assert all(character not in rendered for character in controls)
    assert multilingual in rendered
    for escape in (
        r"\u0000",
        r"\u0009",
        r"\u000a",
        r"\u000d",
        r"\u001b",
        r"\u007f",
        r"\u0085",
        r"\u009b",
        r"\u2028",
        r"\u2029",
    ):
        assert escape in rendered

    assert cli.main([*args, "--json"]) == 0
    result = json.loads(capsys.readouterr().out)["results"][0]
    assert result["bundle_id"] == bundle_id
    assert result["snippet"] == snippet
    with open_state(state_path) as state:
        hit = state.search_segments("needle")[0]
        assert hit.bundle_id == bundle_id
        assert hit.snippet == snippet


def test_plain_registered_search_without_controls_retains_existing_output(
    tmp_path: Path, capsys: CaptureFixture[str]
) -> None:
    state_path = _archive(tmp_path)
    with open_state(state_path) as state:
        hits = state.search_segments("knowledge")
    assert hits
    expected = "".join(
        f"{hit.bundle_id}\t{hit.segment_id}\t[00:00]\t{hit.snippet}\n" for hit in hits
    )
    capsys.readouterr()
    assert cli.main(["library", "search", "knowledge", "--state", str(state_path)]) == 0
    assert capsys.readouterr().out == expected


@pytest.mark.parametrize("start", [0x3105, 0x31A0, 0x31F0, 0x1AFF0, 0x1B000, 0x1B100, 0x1B150])
@pytest.mark.parametrize("operator", [False, True], ids=["literal", "operator-refusal"])
def test_bopomofo_and_kana_block_literal_search(tmp_path: Path, start: int, operator: bool) -> None:
    text = "".join(chr(start + offset) for offset in range(3))
    with open_state(tmp_path / "state.sqlite") as state:
        state.index_synthetic_segment("script-range", 0, text)
        if operator:
            with pytest.raises(ValueError, match="operator-mode search does not support"):
                state.search_segments(f"{text[0]} OR {text[1]}", literal=False)
        else:
            assert [(hit.bundle_id, hit.segment_id) for hit in state.search_segments(text[:2])] == [
                ("script-range", 0)
            ]
