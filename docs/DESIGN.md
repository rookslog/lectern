# Lectern - Design

Status: v0.1 draft, 2026-06-10.

Lectern converts long-form talk media into a local, inspectable bundle: source
metadata, normalized audio, transcript segments, visual notes, references, and
analysis artifacts. The core design goal is to make every stage resumable,
auditable, and safe to run unattended on media the user is allowed to process.

## Problem

Recorded talks are high-value but hard to revisit. Useful claims, references,
questions, and visual context are often locked inside hours of linear audio or
video. Lectern turns those sources into structured artifacts that can be
searched, checked, and reused.

## Product Definition

Lectern operates as a pipeline of pure stages. Stages communicate through a
bundle on disk, not through hidden process state. Each stage reads declared
inputs, writes declared outputs, and records a small run log with tool versions,
parameters, timing, and artifact hashes.

The `Status` column separates the stages the current preview runs from the ones
this document plans but does not yet implement.

| Stage | Status | Input | Output | Notes |
| --- | --- | --- | --- | --- |
| acquire | implemented | source reference | `media/`, `source.json` | Source metadata and media capture. |
| normalize | implemented | media | `media/audio.wav` | Local ffmpeg normalization. |
| transcribe | implemented | audio | `transcript/segments.json`, `transcript/transcript.md`, `transcript/metadata.json` | Timestamped speech segments plus method/provenance metadata. |
| diarize | planned, not implemented | audio + transcript | `transcript/diarization.json` | Optional speaker structure. |
| visual | planned, not implemented | video | `visual/frames/`, `visual/slides.json` | Frame sampling, OCR, and visual descriptions. |
| enrich | planned, not implemented | transcript + visuals | `refs/references.json` | Extracted names, titles, URLs, and identifiers. |
| situate | planned, not implemented | transcript + references | contextual notes for synthesis | Positions the source in relation to resolved references and prior bundle context. |
| synthesize | partially implemented | bundle artifacts | `analysis/summary.md` (implemented as an overview card); `analysis/claims.md`, `analysis/questions.md` (planned) | The overview card records title, duration, transcript method, counts and time span. Lectern does not summarize; summaries are left to agents citing transcript anchors. |

The bundle layout below shows the full planned layout, including the paths the
unimplemented stages would write; the current preview produces only the
artifacts of the implemented stages.

## Bundle Layout

```text
<bundle-id>/
  manifest.json
  source.json
  media/
    audio.wav
  transcript/
    segments.json
    transcript.md
    metadata.json
    diarization.json
  visual/
    frames/
    slides.json
  refs/
    references.json
  analysis/
    summary.md
    claims.md
    questions.md
  log/
```

`src/lectern/bundle.py` is the spine of the system. Pipeline stages should use
the bundle API for all cross-stage data exchange so schema validation, hashing,
and resumability stay centralized.

## Architecture

- The portable core is Python-first: bundle management, schemas, CLI entry
  points, and stage orchestration live in the package.
- Local automation state is SQLite-backed. Source registry, discovery queue,
  policy state, and the minimal library index live in local state rather than in
  hidden process memory.
- Platform adapters are edges, not the core. YouTube, local file, and future UI
  integrations should feed the same bundle contract.
- The CLI is the first stable interface. GUI and daemon surfaces can be added
  after the bundle contract and stage behavior are stable.
- JSON schemas are committed generated artifacts and should be regenerated with
  the source changes that alter them.

## Privacy Posture

Lectern is local-first. Deterministic media stages, transcription, OCR, scene
detection, and deduplication should run locally by default. Remote APIs are
per-stage opt-in and must respect explicit budget and consent boundaries.
Artifacts derived from local-source media must not be sent to a remote service
without explicit per-item consent.

## Quality Evaluation

- Transcript quality should be checked against sampled segments or known-good
  captions when available.
- Analysis claims should carry timestamp anchors back to transcript or visual
  evidence.
- Visual extraction should be verified against frame samples, OCR output, and
  slide grouping behavior.
- Bundle integrity should be tested through hash validation, schema validation,
  and resume/idempotence behavior.

## Non-Goals

- Lectern does not redistribute copyrighted media.
- Lectern is not a real-time transcription product.
- Lectern's local command transcriber path records method metadata and timestamp
  anchors; it does not prove transcript faithfulness or prove that a
  user-supplied command is network-free.
- Lectern does not require a hosted service to run the local pipeline.
