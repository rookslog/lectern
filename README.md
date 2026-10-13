# lectern

[![CI](https://github.com/loganrooks/lectern/actions/workflows/ci.yml/badge.svg)](https://github.com/loganrooks/lectern/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](pyproject.toml)

**Turn talks into thought.** Lectern ingests recorded talks and produces local,
inspectable knowledge bundles for humans and agents.

Lectern is pre-release software. The current public preview surface is a
CLI-first local workflow: ingest local media, record provenance in a bundle, and
inspect the local source, queue, and library state. There is no package-registry
release yet.

## Quickstart

Prerequisites:

- Python 3.12 or newer
- `uv`
- `ffmpeg`

Check that Lectern can be installed from the public repository:

```bash
uv tool run --from git+https://github.com/loganrooks/lectern.git lectern --version
```

Run the local fixture workflow from a checkout:

```bash
set -e
LECTERN_TMP="$(mktemp -d)"
git clone https://github.com/loganrooks/lectern.git "$LECTERN_TMP/lectern"
cd "$LECTERN_TMP/lectern"
make sync
uv run lectern doctor
uv run lectern ingest tests/fixtures/synthetic_talk.wav --output "$LECTERN_TMP/bundles" --state "$LECTERN_TMP/state.sqlite"
uv run lectern library list --state "$LECTERN_TMP/state.sqlite"
```

The fixture uses synthetic audio and a committed transcript sidecar so the
workflow is redistributable and does not download an ASR model.

## Design Shape

- Pipeline of pure stages: acquire, normalize, transcribe, diarize, visual,
  enrich, situate, synthesize.
- Durable bundle artifacts on disk, with explicit manifests and stage records.
- Local-first media processing with per-stage opt-in for remote APIs.
- Synthetic fixtures only; no copyrighted media fixtures in the repository.

## Development

Prerequisites:

- Python 3.12 or newer
- `uv`
- `ffmpeg` for media work beyond the scaffold

```bash
make sync
make verify
```

`make verify` is the local and CI verification entrypoint. It runs linting,
format checks, type checks, tests, and the public repository safety check.

## Current Preview Support

The current `lectern ingest` path supports the synthetic fixture workflow:

```bash
uv run lectern ingest tests/fixtures/synthetic_talk.wav
```

That fixture uses a committed `.transcript.txt` sidecar so CI can check bundle
behavior and transcript passthrough without downloading ASR models or sending
media to remote services.

Lectern can also use an optional local JSON transcriber command for media
without a sidecar:

```bash
uv run lectern ingest local-talk.wav --transcriber-command "my-local-asr --json {input}"
```

The command is executed locally with the normalized audio path and must emit JSON
segments or text. Lectern records transcript method metadata and timestamp
anchors, but it does not bundle an ASR model, does not call remote transcription
providers, and does not claim transcript faithfulness. A user-supplied command
runs with the user's privileges; Lectern cannot prove that command never opens a
network connection.

### Caption sidecars and local ASR

With no `--transcriber-command`, ingest reads caption files beside the media:
`talk.vtt` or `talk.srt`, then language-tagged names such as `talk.fr.vtt` or
`talk.pt-BR.srt`, before falling back to `talk.transcript.txt`. Exact VTT takes
precedence over exact SRT; otherwise the alphabetically first language-tagged
filename is selected. The bundle records the selected caption filename and cue
times. VTT markup is removed and repeated rolling auto-caption lines are
de-duplicated.

Without a sidecar, the optional `lectern[asr]` extra provides local
faster-whisper transcription of normalized audio:

```bash
uv sync --extra asr
uv run lectern ingest local-talk.wav --asr-model small --allow-model-download
```

The default model is multilingual `small`, running on CPU with `int8` compute.
First run: pass `--allow-model-download` once to let
[faster-whisper download model weights](https://github.com/SYSTRAN/faster-whisper#model-conversion);
later runs use the cached model offline. Inference stays local. Both `lectern ingest`
and `lectern queue ingest` accept `--asr-model NAME`, `--allow-model-download`, and
optional `--language CODE`. Downloads are disabled by default, and the bundle records
whether model downloads were allowed.
Without a language override, ASR auto-detects the language; captions use the
filename's language tag when present. The bundle records language and ASR backend
details, including language probability. Captions (including third-party and
auto-generated captions) and ASR output make no transcript faithfulness claim.

Before planning or processing, Lectern captures a private temporary media copy
and the sidecar bytes, if present. Queued ingest checks those captured inputs
against the existing approval before normalization or transcription; planning,
replay selection, and processing use the same capture. Approval still covers a
present sidecar when an explicit command supplies the transcript instead.
This binds the input Lectern supplies, not the behavior or faithfulness of an
arbitrary user command. It does not authenticate previously generated bundles.

Lectern also has an early local automation spine for folder sources. It records
source and queue state in local SQLite, scans local folders without network
access, requires explicit queue approval before ingesting a discovered item, and
indexes completed bundles:

```bash
uv run lectern sources add-folder talks ~/Talks
uv run lectern sources scan talks
uv run lectern queue list
uv run lectern queue approve <queue-item-id>
uv run lectern queue ingest <queue-item-id>
uv run lectern library list
```

### Playlists and an inbox folder

Obtaining media is your own action and rights judgment, outside Lectern: use
media you have the right to access. Lectern includes no downloader. Place local
media in an inbox folder with optional adjacent files:

- `<stem>.info.json` in yt-dlp's `--write-info-json` format. Lectern reads only
  `id`, `title`, `channel` (fallback `uploader`), `upload_date` (YYYYMMDD),
  `duration`, `webpage_url`, `language`, `playlist_id`, `playlist_title` (fallback
  `playlist`), `playlist_index`, and `extractor_key`; all other keys are ignored
  and never copied into bundles. Missing fields are fine. Malformed metadata is
  ignored with an acquire-stage warning; its bytes still bind queue approval.
- `<stem>.<lang>.vtt` for timed captions, using the selection rules above.

```bash
lectern sources add-folder inbox ~/Lectern/inbox
lectern sources scan inbox
lectern queue list
lectern queue approve <queue-item-id>
lectern queue ingest <queue-item-id>
lectern library list --playlist
```

Repeat approval/ingest for each queued talk. `library list --playlist` groups by
playlist ID, orders talks by playlist index (missing indexes last), then lists
talks without a playlist. Plain groups have a `playlist title (playlist ID)`
header and `index<TAB>bundle ID<TAB>title` rows; `--json` keeps a flat, ordered
`bundles` array. `--playlist ID` filters to one playlist. Metadata language is
only a recorded hint: it does not override ASR detection or `--language`.

Plain `library list` rows contain four tab-separated fields: bundle ID, creation
time, status (`ready`, `incomplete`, `failed`, or `needs-reprocessing`), and title.
The first three fields retain their order; JSON rows also include `title` and
`external` (null when no external identity is recorded).
`library show <bundle-id>` prints a readable overview with title, status, creation
time, recorded source kind, duration, transcript method, segment and word counts,
layer availability from manifest stage states, and the redacted bundle directory.
Channel, published date, original URL, and playlist title/index appear when present.
Its `--json` output keeps `bundle` and `manifest` and adds these overview fields.
Transcript facts that cannot be read through strict evidence checks are shown as
unknown (null in JSON); inspection remains available for non-ready bundles.

`analysis/summary.md` is an overview card, not a summary: it records the title,
duration, transcript method, segment and word counts, and segment time span,
plus channel, published date, original URL, and playlist title/index when present.
Lectern does not summarize. Agents write summaries using Lectern's anchors.
Read a ready bundle's transcript with:

```bash
uv run lectern transcript <bundle-id> --state PATH
uv run lectern transcript <bundle-id> --format txt --state PATH
uv run lectern transcript <bundle-id> --format json --state PATH
```

The default Markdown output includes a title heading and timestamped paragraphs
such as `[03:12] (#14) text`; `txt` omits the heading. Times use `[h:mm:ss]`
past an hour. JSON includes `bundle_id`, `title`, and segments with `id`, `start_s`,
`end_s`, and `text`. The command refuses non-ready bundles.

Use `--json` on source, queue, and library commands for machine-readable output.

### For agents: packets and citation checks

`lectern export BUNDLE_ID` exports a ready talk's source facts, layer availability
and evidence limits, and transcript segments with durable anchors. The JSON
packet is **experimental** (`lectern.talk-packet`, version `0.1`); its shape may
change before a stability decision. Use `--format md` for a readable packet.
Agents write breakdowns, briefs, and reviews; Lectern supplies the evidence.

```bash
uv run lectern export <bundle-id> --format json --state PATH
uv run lectern export <bundle-id> --format md --state PATH
uv run lectern check-citations review.md --state PATH
uv run lectern check-citations review.md --state PATH --json
```

Cite moments as `[t=MM:SS bundle#seg]` (use `H:MM:SS` past an hour), for example
`[t=00:01 synthetic-talk#1]`. Each packet anchor preserves Lectern's content hash
and canonicalization version and adds time-range and exact-quote selectors.
Displayed text and quotes are path-redacted; hashes still refer to original
evidence, so a redacted quote need not hash to the anchor's digest.

The checker reports `ok`, `time_mismatch`, `unknown_bundle`, `unknown_segment`,
or `not_ready` for each token against current strict registered evidence. Times
must match the segment start's whole second. It exits 1 for problem tokens and
0 otherwise; uncited paragraphs are counted but do not fail the check. This
checks references, not whether a claim is supported or whether every claim is
cited. Short tokens do not carry a historical text hash: after legitimate
reprocessing, use stored packet anchors to compare historical content.

Visual and speaker presence follow manifest stage states. Translation is absent
because the current bundle schema has no translation stage or artifact.

A failed ingest rerun still reports its invocation error, but retains the prior
completion when that exact owned, registered bundle survives. Readiness remains
separately derived from current evidence. Queue state is not a complete history
of every invocation.
The state database is local run state under `.lectern/` by default and should not
be committed.

Lectern can also discover public YouTube playlist metadata through the YouTube
Data API. This is metadata-only discovery: it records playlist/video metadata in
the local source registry and review queue, but it does not download YouTube
media, captions, or transcripts and cannot yet ingest a YouTube queue item into
a bundle.

```bash
export YOUTUBE_API_KEY=...
uv run lectern sources preflight-youtube "PL..."
uv run lectern sources add-youtube-playlist lectures "PL..."
uv run lectern sources scan lectures
uv run lectern sources scan lectures --max-pages 2
uv run lectern queue list --json
```

The API key is read from the environment and is not stored in Lectern state or
bundle artifacts. Each playlist page request consumes an estimated 1 YouTube
Data API quota unit; scan JSON reports estimated units consumed for the scan.
`--max-pages N` caps how many playlist pages one scan requests. A capped scan is
a partial view of the playlist, so it records new and changed items but skips
removals entirely and reports `removals_skipped_due_to_truncation` in the scan
metadata. `--max-pages` and `--api-key-env` apply to YouTube playlist sources
only; passing either to a local-folder scan is rejected with exit code 2.

A playlist entry is identified by its video ID, not by its playlist-item ID, so
the same video listed twice in one playlist is recorded once, and reordering a
playlist or re-adding a video under a new playlist-item ID does not create a
second entry for it. Playlist-item IDs, playlist positions, and insertion
timestamps are recorded as metadata but do not count as content changes, so a
remove-and-re-add is not re-queued for review.
Private and deleted playlist placeholders are recorded as source items flagged
`video.placeholder` and are never added to the review queue.

A YouTube queue item cannot be ingested into a bundle. Attempting it records the
item in the terminal `unsupported` state with an explanatory error, and
`queue retry` / `queue approve` on such an item are refused rather than looping
through a failure that can never succeed. Use `queue list --queue-state
unsupported` to inspect them.

## Search and citations

`lectern library search QUERY --state PATH` searches transcript segments and is
literal by default; pass `--operators` only when you intentionally want SQLite
FTS operators. When literal query text contains option-like tokens such as
`--json`, put a standalone `--` before the query.
Plain search hits contain four tab-separated fields: bundle ID, segment ID,
`[mm:ss]` (or `[h:mm:ss]` past an hour), and snippet. JSON hits include `start_s`
and `title` alongside the existing bundle ID, segment ID, and snippet. These
added facts use strict registered transcript evidence; unavailable facts are
null in JSON and an unavailable time is `[unknown]` in plain output.
`lectern library cite BUNDLE_ID SEGMENT_ID --state PATH` returns
an anchor tied to the cited transcript content. An `exact` resolution means the
segment ID and content digest agree; it does not establish that the timestamp is
within the recording or playable. The separate anchor-correctness sampler checks
temporal bounds and ordering. Displayed timestamps truncate to whole seconds
and retain a negative sign when the recorded time is negative.

Search tokenization separates words on whitespace and punctuation for Latin,
Greek, Cyrillic, Arabic, and Hebrew, including right-to-left text. Literal search
uses bounded character segmentation for the listed Han, kana, Hangul and Bopomofo
ranges: CJK unified/compatibility ideographs and their supported extensions;
Hiragana, Katakana, half-width Katakana, Katakana Phonetic Extensions, Kana
Supplement, Kana Extended-A/B and Small Kana Extension; Hangul syllables and
Jamo ranges; and Bopomofo and Bopomofo Extended. This is not exhaustive support
for every language, orthography or Unicode script. Two-character terms can match
inside longer runs, with substring false positives and no word-boundary accuracy;
stemming is not provided. Operator-mode queries refuse these character ranges;
use literal search for them. Later search work may add ranking or semantic
retrieval; the current literal search and citation commands are available now.

The [recorded M5a latency datum](docs/benchmarks/m5a-search-latency.json)
captures one synthetic 1,000-bundle measurement and its conditions; it is not
a performance guarantee. The state store currently reconciles registered
transcript files on each command open to keep search results aligned with
changed or missing bundle content. The datum keeps separately labeled historical open/reconcile measurements
alongside the current SQLite query-only timing; the open-path measurements were
not rerun for the current segmenter version.

## Current Limits

- YouTube support is limited to public-playlist metadata discovery through an
  API key. OAuth/private playlist access is not implemented.
- Lectern does not download media, captions, or transcripts from external
  services.
- MCP/API access, ranked or semantic search, visual evidence, OCR, reference resolution, and
  citation-gated synthesis are later roadmap items.
- The local command transcriber path is an integration point, not a bundled ASR
  engine or transcript-quality guarantee.
- Local bundles can contain sensitive media-derived artifacts. Keep them out of
  commits and issue reports.
- Bundle manifest compatibility is tied to the manifest `schema_version`. The
  current manifest schema version is `1.0.0`; pre-release compatibility policy
  is described in [SUPPORT.md](SUPPORT.md).

Read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request. Security
and privacy reporting guidance is in [SECURITY.md](SECURITY.md).

## Documentation

- [Design](docs/DESIGN.md)
- [Grey Areas](docs/GREY_AREAS.md)
- [Bundle migrations](docs/MIGRATIONS.md)
- [Privacy](PRIVACY.md)
- [Roadmap](ROADMAP.md)
- [Support and compatibility](SUPPORT.md)
- [Changelog](CHANGELOG.md)
- [Architecture decisions](docs/adr/)
- [Automated contributor guidance](AGENTS.md)

## License

MIT. Lectern processes media you have the right to access, on your own machine,
and provides no redistribution features.
