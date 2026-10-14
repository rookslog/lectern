# Grey Areas & Positions

The honest list: things that are legally, ethically, or technically unsettled, made explicit with our position and reasoning. Revisit any of these if facts change; changes of position get an ADR.

## 1. YouTube ToS and copyright

**The grey zone.** Downloading via yt-dlp breaches YouTube's ToS (a contract matter between user and YouTube, not per se copyright infringement in most jurisdictions). Transcripts and summaries of others' talks are derivative works; fair-use/fair-dealing analysis favors private research use but has never been litigated for this exact shape.

**Position.** Lectern is a local tool acting on the user's behalf for private study — functionally a research note-taking aid. We: (a) never redistribute media, transcripts, or bundles; v1 ships no sharing/publishing features for third-party content; (b) prefer official caption tracks when available and store them; (c) commit no copyrighted fixtures to the repo (synthetic media only); (d) document that ToS compliance is the user's responsibility, like yt-dlp itself does. The public repo ships *code*, which is uncontroversial — yt-dlp's continued existence and ubiquity is the practical precedent.

**Why not "captions-API only"?** Official captions are frequently absent, auto-generated (poor on technical vocabulary), and never cover the visual channel. A captions-only tool fails the core use case.

## 2. Playlist polling authentication

**The grey zone.** The clean way to read a private "Lectern Inbox" playlist is the YouTube Data API (OAuth) — but shipping an OAuth app requires Google verification, and per-user API quota setup is real friction. The alternative (yt-dlp with browser cookies) works but deepens the ToS exposure and is brittle.

**Position.** Support both; default to Data API with user-supplied credentials (documented setup, ~10 min, acceptable for the v1 audience of technical users), with yt-dlp+cookies as the documented fallback. Revisit if/when there's a non-technical audience. A *public* playlist requires no OAuth at all — recommend that as the lowest-friction default for non-sensitive queues.

## 3. Visual aids: necessary but unbounded cost

**The grey zone.** Slide-heavy talks are unintelligible from transcript alone, but naive frame-by-frame VLM analysis is absurdly expensive and slow.

**Position.** Spend determinism first, intelligence second: scene detection and perceptual-hash dedup are free and typically reduce a 1-hour talk to 30–80 unique slides; OCR (local, free) runs on all of them; VLM description runs only on deduped slides, under a per-item budget cap, and only where OCR text is insufficient (figures, diagrams, plots). Degradation is graceful: a bundle with OCR-only slides is still vastly better than transcript-only.

## 4. Diarization licensing

**The grey zone.** pyannote (the local SOTA) requires gated HuggingFace weights with their own license terms; we can't vendor or assume it.

**Position.** Diarization is an optional extra (`lectern[diarize]`) with its own setup step; the pipeline degrades gracefully without it. No core feature may depend on speaker labels.

Ungated models also exist: the [pyannote segmentation-3.0 ONNX export](https://huggingface.co/csukuangfj/sherpa-onnx-pyannote-segmentation-3-0)
([MIT, CNRS](https://huggingface.co/pyannote/segmentation-3.0/blob/main/LICENSE))
and the [WeSpeaker ResNet34 VoxCeleb embedding model](https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet34-LM)
(CC BY 4.0) are used
by the optional sherpa-onnx backend. Users fetch them during setup; Lectern does
not vendor either model.

## 5. Live recordings of other people

**The grey zone.** Recording a lecture you attend involves consent norms (and sometimes law — one/two-party consent rules, institutional policy) that vary by jurisdiction and venue. The tool can't know whether a given recording was permitted, and speakers hold rights in their own presentations.

**Position.** Same stance as §1: local private-study tool, user's responsibility, documented plainly. Reinforced technically by the privacy hard rule (DESIGN §5): local-source media never reaches a remote API without explicit per-item consent — so even a recording made in good faith never leaks by default. We add a docs note on asking permission; we do not add consent-verification theater the tool can't actually perform.

## 6. Hallucinated context in the "situate" stage

**The grey zone.** The most seductive feature — "what discourse is this talk part of?" — is the one most prone to confident fabrication, precisely because the user invokes it when they *don't know the area* and can't catch errors.

**Position.** Situate output is citation-gated: every claim must carry a resolvable reference (arXiv ID, URL, or a `[t=...]` anchor), and claims are labeled with explicit evidence status such as `Verified in scope`, `Proxy support`, or `Underdetermined`. Unresolvable claims are dropped, not hedged. The stage is skippable and clearly marked as the least-trustworthy artifact in the bundle. This is a faithfulness budget, not a style guide — violations are bugs.

## 7. Evaluating quality without a grader

**The grey zone.** "Is this transcript/summary good?" has no free oracle; agent-built pipelines drift toward self-graded plausibility.

**Position.** Manufacture cheap, decorrelated checks (DESIGN §7): a small gold-set with reference transcripts; official captions as a divergence alarm; timestamp-anchored claims verified by a separate verifier agent against the transcript span. None of these prove quality; together they bound it — and they're all runnable in CI or a nightly loop.

Current pre-release code records transcript method metadata and timestamp
anchors, but it does not yet implement the full gold-set, caption-divergence, or
verifier-agent quality loop.

## 8. Storage growth

**The grey zone.** Video is huge; a working ingestion habit will eat a disk within months.

**Position.** Policy default: video deleted after the visual stage completes; audio (16 kHz mono, ~30 MB/hr) and keyframes kept; everything re-derivable from source URL is re-derivable. `keep_video: true` per-source override exists. Local recordings are never auto-deleted (they're the only copy).

## 9. Long inference jobs inside agent loops

**The grey zone.** Transcribing a 2-hour talk takes real wall-clock time; an LLM agent polling "is it done yet?" burns tokens for nothing and is the classic harness failure mode.

**Position.** Long-running pipeline stages use detached processes, file-centric state, and event logs; harnesses and future watch daemons wait on deterministic events. No LLM in any wait loop, ever.

## 10. Repo licensing

**Position.** MIT for maximal contribution friendliness; no model weights or third-party media are vendored, so the license covers only our code. Model and dataset licenses are the user's installation-time concern, surfaced by `lectern doctor`.
