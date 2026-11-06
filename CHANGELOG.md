# Changelog

All notable public changes are recorded here. Lectern is pre-release software;
versioned release notes will become stricter once preview support stabilizes.

## Unreleased

- Fixed timed-caption/local-ASR follow-ups (#89): preserve ordinary repeated
  VTT cues while de-duplicating rolling captions; bind derived language/method
  to language-tagged caption approvals; preserve Hub model IDs while redacting
  local paths; refuse cached-only local models missing `tokenizer.json`; and
  retain faster-whisper's effective language with a warning when it differs
  from `--language`. Existing approvals for language-bearing captions change
  once and require a rescan and approval before ingest; untagged-sidecar
  approval digests are unchanged. Command/sidecar language overrides still apply.
- Fixed packet citation round-trips for migrated IDs and negative times, and
  report oversized numeric tokens as malformed (#94). Citation checks now show
  uncited paragraph locations/excerpts and pasteable time corrections without
  changing citation validity, exit codes, or the plain summary line (#95).
- Preserved source-container audio offsets during normalization so transcript
  times align with video (#84, part: audio timeline).
- Bound aggregate leading/interior normalization silence to 600 seconds using an
  uncompensated, streaming FFmpeg decoded-frame analysis through EOF. It has a
  60-second whole-pass deadline, bounded records/memory, exact per-run tagged
  timing, and refuses indeterminate rate/reinit/backward/overlap configurations
  before PCM output. This supersedes the initial-only, 64-packet `ffprobe` guard;
  audio normalization no longer requires `ffprobe`. Canonical WAV copy, bounded
  timestamp jitter, and ordinary delayed/gapped audio are preserved. General
  decoded-duration expansion and hostile-decoder isolation remain outside this
  guard (#84).
- Refuse ingest replays of damaged owned bundles before provenance repair (#82).
  Queue and one-shot replay now share artifact, completed-stage, and strict
  selected-transcript admission checks; provenance-only recovery is checked again
  on disk before reporting success and preserves prior completion on refusal.
- Retain core-bound pending provenance bytes before declaring a repaired source
  (#103). Interrupted queue and one-shot repairs can recover across writer schema
  and source-name changes, including non-final provenance keys. Recognized stale
  receipts are cleaned up; invalid receipt paths/contents and edited cores are
  refused. Failed repairs restore receipt bytes or absence along with source and
  manifest. Pre-receipt recovery supports schema 3 and the current writer only,
  with unchanged source-name/policy values; arbitrary historical annotations are
  not inferred from a digest.
- Raised the writer's manifest schema to `1.1.0` for optional transcript
  language/ASR backend metadata, `source.external`, and stage warnings. Readers
  accept both `1.0.0` and `1.1.0` without migration and refuse other versions,
  including `1.2.0`. Migration of a `1.0.0` bundle remains a validated no-op;
  the legacy `0.1.0` to `1.0.0` migration target is unchanged.
- Projected metadata paths in JSON library listings, show output (including
  embedded manifest sources), and retained search-title projection. Non-UTF-8
  metadata strings are ignored with the existing acquire-stage warning and
  filename fallback.
- Added opt-in `--frames` for one-shot and queued ingest: local scene/interval
  JPEG samples, perceptual deduplication, timestamped `visual/frames.json`, and
  manifest-covered images. Interval/max options bind visual bundle identity.
  Agent packets expose frame limits and `#f<index>` citations; citation checks
  resolve them and `library show` reports the frame count. No OCR is included.
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
  `1.0.0` bundles remain readable under the additive `1.1.0` contract.
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
`1.1.0`.
