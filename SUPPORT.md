# Support And Compatibility

Lectern is pre-release software. Support currently targets the public `main`
branch and the current documented preview workflow.

## Supported Environment

- Python 3.12 or newer.
- `uv` for environment and command execution.
- `ffmpeg` for media normalization and `lectern doctor`.
- macOS and Linux are expected development targets. CI currently runs the
  verification matrix on `ubuntu-latest` and `macos-latest` across Python 3.12,
  3.13, and 3.14.

## Version Policy

The package version, the bundle manifest schema version, and the local
automation state schema version are separate.

- The package version describes the installed Lectern software.
- The manifest `schema_version` describes the bundle manifest contract.
- The automation state schema version describes the layout of the local SQLite
  automation store (sources, discovery queue, and library index).

The current package version is `0.0.1`. The current manifest schema version is
`1.1.0`. The current automation state schema version is `3`.

Before a stable release, CLI flags and command output may change. Bundle schema
changes are treated more carefully. Version increments classify changes:
additive bundle manifest changes should raise the manifest schema minor version,
and breaking changes require a major schema version plus migration notes. The
current preview reader accepts manifest versions `1.0.0` and `1.1.0` without
migration. It does not claim forward compatibility with later same-major
schemas: strict artifact validation rejects undeclared fields rather than
silently discarding data, including privacy-sensitive data. Reading a later
schema therefore requires an updated reader or an explicitly documented
migration path.

Manifest `1.1.0` adds optional transcript metadata `language` and ASR backend
model, library version, compute type, detected language, and language probability
fields, plus optional `source.external` identity/playlist metadata and stage
`warnings` arrays. New bundles declare `1.1.0`; `1.0.0` bundles remain readable
because they are a subset of this additive contract. All other versions,
including `1.2.0`, are refused by the reader; `0.1.0` still requires migration.

The current `0.1.0` to `1.0.0` bundle procedure is documented in
[docs/MIGRATIONS.md](docs/MIGRATIONS.md). Manifest `1.0.0` does not mean the
Lectern package has reached `1.0.0`; that legacy migration target stays unchanged.

The pre-release `1.0.0` contract requires nonblank transcript segment text:
at least one character must fall outside Python's Unicode whitespace set.
This requirement was tightened in place while establishing that contract.
Otherwise valid text is preserved exactly. Current readers reject documents
containing blank segments for indexing and citation creation; existing derived
index entries are removed when registered bundles are refreshed. Migration
refuses such legacy documents without rewriting their evidence.

### Automation State Schema

The automation state schema version is recorded in the store's SQLite
`user_version` and reported by `lectern` as `state_schema_version` in bundle
provenance. It is an internal storage contract, not part of the bundle
contract, and it moves independently of the package and manifest versions.

Current behavior when a store's version differs from the running code's:

- A store at version 1 is migrated forward to the current version on open. The
  pre-migration file is copied aside as `<state-file-name>.v1.bak` first, so the
  original bytes survive a failed or interrupted upgrade.
- A store at version 2 gains the local retrieval index on open, and every
  bundle already registered in the library is indexed as part of that
  upgrade. Search covers an existing archive, not only recordings added
  afterwards. A bundle whose files cannot be read is skipped rather than
  failing the upgrade, so one moved or deleted directory does not prevent
  the store from opening.
- The retrieval index records the text-normalization and segmentation rules
  it was built under. If a later version changes either, the index is
  rebuilt on open rather than queried under rules it was not written with.
- An empty or uninitialized store is created at the current version.
- A store written by a newer version than the running code supports is refused
  with an error naming both versions; Lectern does not open it and does not
  attempt to downgrade it.

This section describes what the current preview does. It is not a
forward-compatibility guarantee.

## Security Fixes

Until versioned releases exist, security fixes target the current public `main`
branch. Do not include private media, transcripts, credentials, generated
bundles, or local state databases in public reports.

## Unsupported Workflows

The current preview does not support:

- package-registry installation;
- external media acquisition;
- external source discovery beyond the documented metadata-only YouTube
  playlist workflow;
- MCP/API access;
- OCR or visual evidence extraction;
- remote model stages;
- a GUI or watch daemon;
- transcript faithfulness guarantees for arbitrary media.
