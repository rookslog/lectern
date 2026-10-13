# Changelog

All notable public changes are recorded here. Lectern is pre-release software;
versioned release notes will become stricter once preview support stabilizes.

## Unreleased

- Added experimental `lectern export BUNDLE_ID` JSON/Markdown talk packets with
  source facts, layer limits, and existing content-hash anchors extended with
  time-range and quote selectors. Added `lectern check-citations FILE` for
  current-evidence citation resolution, whole-second time checks, and uncited
  paragraph counts. Both commands project filesystem paths out of their output.
- Added adjacent local `.info.json` metadata intake with an allowlisted optional
  external identity, channel, publication date, and talk title. Captured metadata
  bytes are bound to queue approval and bundle identity; malformed metadata is
  ignored with a recorded acquire-stage warning. No downloader is included.
- Added `library list --playlist [ID]` grouping/filtering and title/external JSON
  fields, plus provenance facts in `library show` and the overview card. Existing
  bundles remain readable without a manifest schema version change.
- Added timed VTT/SRT sidecar ingestion, including language-tagged filenames and
  rolling auto-caption de-duplication, with captured caption bytes bound to queue
  approval. Added optional local faster-whisper ASR via `lectern[asr]`, model and
  language flags, and additive language/backend transcript metadata.
- Replaced the extracted first-sentence summary with an honest overview card at
  `analysis/summary.md`, retaining its artifact path, stage, and manifest shape.
  Lectern does not summarize; agents write summaries citing Lectern's anchors.
- Added `lectern transcript BUNDLE_ID` with timestamped Markdown, text, and JSON
  views of ready bundles through strict registered evidence readers.
- Expanded `library show` into a readable overview and additive JSON fields;
  appended title to plain `library list` rows without changing JSON listings.
- Added segment time and talk title to search-hit JSON, and a timestamp field
  between segment ID and snippet in plain search output. Search matching and
  ranking are unchanged.

- Bound local planning and processing to a private captured input, checked
  against existing queue approval before processing. Preserved existing source
  identities, sidecar consent coverage, replay behavior, and original-media
  mutation refusal.

- Aligned migration, library readiness, indexing, and new citations on strict
  manifest-backed source and selected-transcript evidence. Missing or
  contradictory declarations are refused while existing-anchor diagnostics
  retain their changed-evidence outcomes.

- Established nonblank transcript segment validity in the pre-release `1.0.0`
  contract, with matching runtime and exported schema constraints. Migration
  rejects blank evidence without changing the original bundle; valid text is
  preserved exactly. Registered readers remove invalid documents from retrieval
  and refuse new citations.

- Changed local manifest `source.ref` from a filesystem path to a content
  identity, raised the manifest schema to `1.0.0`, and added the explicit
  `lectern migrate BUNDLE` procedure with a retained `0.1.0` backup.
- Added local command transcription support for media without a transcript
  sidecar.
- Added transcript method metadata and timestamped transcript artifacts.
- Added local automation spine commands for folder sources, queue operations,
  and minimal library inspection.
- Added public preview documentation for privacy, roadmap, support, and
  quickstart verification.

## 0.0.1

- Initial public package metadata, CLI scaffold, manifest schema, synthetic
  fixture ingest, public safety checks, and CI verification.

## Versioning Notes

The package version and bundle manifest schema version are separate.

- Package version: reports the installed Lectern software version.
- Manifest schema version: records the bundle manifest contract read by tools
  and agents.

The current package version is `0.0.1`. The current manifest schema version is
`1.0.0`.
