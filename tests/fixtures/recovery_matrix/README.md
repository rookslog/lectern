# Synthetic full-passage recovery matrix

These authored transcripts are invented, redistributable test material. The
intervention, groups, counts and field trial are fictional. Tests copy the
existing synthetic tone audio, ingest locally, then install these segments and
update their manifest declaration in temporary bundles. The timing is test
data, not a claim that the audio speaks these words. No generated bundle,
private media, provider, network connection or revision store is required.

The runnable oracle is [`test_m5b_recovery_matrix.py`](../../test_m5b_recovery_matrix.py).
Each internal-read row runs with both a retained store and a reopened store,
using the same held search observation across the mutation. The selected
passage is segment 1, with neighboring segments that must not be returned as
part of its full text.

| Row | Mutation and current behavior pinned |
| --- | --- |
| Unchanged | A held observation still produces an exact citation and the complete original segment. |
| Drifting | Manifest-valid wording changes outside the snippet. The excerpt stays identical, but the held A observation is refused. A separately searched B observation cites full B; the old anchor diagnostically reports modified. |
| Retimed | Identical words move from 0.4 to 0.65 seconds. The held observation is refused; a new observation cites the new time. Old text-anchor diagnosis remains exact, demonstrating that exact does not certify unchanged timing. |
| Unreadable — missing | Removing selected segments refuses citation and yields no hits with nonready status, unlike an ordinary healthy no-match response. |
| Unreadable — malformed | Invalid JSON with a matching manifest declaration isolates parsing failure; citation is refused, with the same nonready/no-hit distinction. |
| Unreadable — undeclared drift | Valid replacement text without updated integrity metadata is diagnostically visible as modified, but cannot produce a strict observed citation or search hit. |
| Nonready but readable | An unrelated metadata artifact has damaged integrity. The bundle is nonready, but its unchanged selected evidence remains strictly readable and citable. Status alone is not an admission decision. |
| Out-of-snippet qualifier | Actual search output omits the decisive field-trial qualifier, while successful observed citation returns it in full. Exact citation does not endorse the illustrative claim. |

`matrix.json` freezes the original query/mode/limit, illustrative question/optional claim,
qualifier, mutations and expected outcomes. The original and replacement
segment files are separate authored inputs; expected passage text is never
derived from the response under test. Malformed/missing/metadata damage is
constructed deterministically in the temporary bundle.

## Deferred consumer acceptance

Seven explicitly skipped matrix tests reserve the eventual MCP/API recovery
workflow. They require preserving the original question, query/mode/limit and
optional claim; keeping held and candidate observations distinct; permitting
candidate rejection without losing work; and avoiding unobserved-citation
fallback. One further skipped test reserves full-passage exposure and a
deliberate review action before explicit support confirmation, with rejection,
qualification and an unresolved relationship still possible.

These are placeholders, **not consumer coverage or adopted API/UI fields**.
Unskipping them requires an actual consumer and an adopted interaction
contract, not a test-only implementation. The passing tests establish only
current internal-read behavior. They do not test comprehension, preference,
session restart/export, correction overlays or multimedia understanding.

## Run offline

With the existing development environment installed:

```bash
UV_OFFLINE=1 UV_NO_SYNC=1 uv run pytest -q -rs tests/test_m5b_recovery_matrix.py
UV_OFFLINE=1 UV_NO_SYNC=1 make verify
UV_OFFLINE=1 UV_NO_SYNC=1 make public-check
```

The focused matrix contains 16 runnable cases and 8 explicit skips. The default
suite denies network sockets and deselects integration tests. `UV_NO_SYNC`
uses the existing environment; these commands do not install dependencies.

Validation on this matrix: the focused command passed 16 cases with 8 explicit
skips; `make verify` passed 977 tests with 8 skips and 1 integration test
deselected, including lint, formatting, strict type checks and public safety.
Two temporary fault controls also exercised the oracles: substituting
unobserved citation caused all four drift/retiming cases to fail on missing
refusal; returning only the snippet caused both qualifier cases to fail on
missing full text. No production files were edited for these controls.

## M5b boundary

This addition does not change the backend or complete M5b. Full M5b still needs
the thin MCP/API adapter, fixture and real-local-bundle acceptance, response
path-disclosure checks, adopted no-match/as-of and downstream disclosure/consent
semantics, the owner compatibility decision and its support wording, and
complete-request cost measurement/optimization before sustained archive use.

"Full passage" here means the full selected transcript segment, not a whole
recording or adjacent/visual context. Mutations finish between calls; no
whole-library atomicity or A-to-B-to-A history guarantee is tested. Keeping the
held immutable hit is not proof of preserved question/claim state or explicit
reassociation; those consumer checks remain skipped.
