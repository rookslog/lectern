# ADR-0003: The bundle is the contract (typed, versioned, file-centric)

Status: accepted · 2026-06-10

Amended: 2026-11-06 · scoped preview transcript-metadata exception below.

## Context

Every consumer — CLI, MCP server, future GUI, arbitrary agents with file tools, Obsidian — needs a stable way to read pipeline output. Pipelines that communicate through in-memory objects or databases couple consumers to the implementation and defeat resumability.

## Decision

All pipeline state and output lives in a **bundle**: a self-describing directory with a `manifest.json` carrying `schema_version`, source provenance, per-stage status, and artifact content hashes. Stages communicate *only* through bundle artifacts. Schema is defined once in `src/lectern/bundle.py` (strict pydantic models, JSON Schema export committed to `schemas/`); migrations are explicit (`lectern migrate`), and `schema_version` bumps follow semver discipline: additive = minor, breaking = major + migration.

### Scoped preview amendment · 2026-11-06

The original decision above did not provide a local-artifact exception. This
amendment introduces one for the preview `transcript/metadata.json` artifact,
which explicitly records `schema_contract.manifest_schema_versioned: false`.
The manifest contract retains its schema-version semver discipline; this
exception does not change the manifest wire format or its current `1.1.0`
version. The earlier `1.1.0` changes and their rationale remain as documented in
[SUPPORT.md, Version Policy](../../SUPPORT.md#version-policy).

The optional backend `terms` and `terms_applied` fields are an in-place extension
of this local artifact under its unchanged `lectern.transcript.metadata.v0` tag.
They require updated strict artifact readers: older strict readers that do not
declare these fields reject hints-bearing metadata. With no terms, the new
fields remain omitted and transcript identity and backend provenance bytes
remain unchanged. Strict models and the committed artifact schema still define
the fields; this exception does not permit silently discarding unknown data.

The scope is this preview metadata extension, not a general forward-compatibility
or stable API promise. Retaining the versions avoids expanding an unchanged
manifest contract and preserves the no-term identity/provenance baseline. A
manifest bump, migration engine, and artifact-version negotiation are not part
of this amendment. It is a bounded technical execution decision, not human
ratification of a stable-contract policy change; explicit human review remains
required for breaking stable manifest changes.

## Consequences

- Agents need no SDK — file tools suffice; the MCP server is convenience, not gatekeeper.
- Idempotence, resume behavior, and crash safety fall out of hash-checked, file-centric stage state.
- Every analysis assertion carries a `[t=hh:mm:ss]` anchor; faithfulness is checkable (DESIGN §7).
- Accepted cost: schema discipline is ongoing work; treated as the project's most protected surface, with breaking changes requiring explicit human review.
