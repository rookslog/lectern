# Changelog

All notable public changes are recorded here. Lectern is pre-release software;
versioned release notes will become stricter once preview support stabilizes.

## Unreleased

- Added timed VTT/SRT sidecar ingestion, including language-tagged filenames and
  rolling auto-caption de-duplication, with captured caption bytes bound to queue
  approval. Added optional local faster-whisper ASR via `lectern[asr]`, model and
  language flags, and additive language/backend transcript metadata.

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
- Added transcript method metadata, timestamped transcript artifacts, and
  timestamp-derived summary anchors.
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
