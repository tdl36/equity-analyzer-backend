# Full-transcript comparison trial

Status: automatic companion generation approved September 9, 2026. Existing generation prompts,
original notes, and automatic iCloud ingestion remain unchanged during the comparison trial.

## Entry point

Summary → open a saved note → Original / Improved / Side by side.
New audio, text/document/YouTube, podcast full summaries and newly saved manual notes
queue a separate improved version after the original save commits. Original generation
and content remain independent. No historical backfill is automatically started.
Older notes have one Generate improved notes button. Progress is visible without
opening nested controls; every improved section appears in one scrollable reader.
The comparison source and original snapshot are immutable. Duplicate saves do not
replay the same source/version. At most two comparison model workers execute per
server process; PostgreSQL advisory locks exclude concurrent execution of the same job.
Keys are never persisted. Missing keys are recorded as a retryable failure.
Original saves remain successful if companion queueing fails; an alert points to retry.
Interrupted workers retain checkpoints; Retry is available after failure or stale progress.
This release does not add automatic restart recovery or change the editorial prompts.
No original is overwritten, emailed, approved or published by comparison generation.

## Scope of full-source processing

The NEW path has no total 20,000 / 50,000 / 200,000 character cutoff. Lossless partitions
bound individual model requests; every part is processed and its detailed management record
saved. Analytical sections consume all part records, with recursive consolidation when needed.
This is hierarchical synthesis, not a claim that a model has unlimited context or that generated
records cannot omit a detail. Exact raw source is frozen privately with a SHA-256 fingerprint.
Original generation also sends complete source text as of September 9, 2026, without
20K/50K/200K slices, while retaining its existing editorial prompts. It uses direct model
requests and remains subject to provider context limits; only the comparison path has
hierarchical processing. Previously saved baseline notes are not regenerated automatically,
so older comparisons may still reflect differences in source coverage. Input must already exist as saved source text; this cannot recover missing audio
or omitted text from an earlier transcription/extraction stage.

## Conservative editorial changes

Preserve management commentary, numbers, hedges, examples and clarifying Q&A; separate
interpretation/open issues. No guessed entity identities, invented psychology, credibility
scores, or unsupported novelty/model-change claims. A shared instruction set applies to all
new sections. Executive brief is optional reading depth; detailed part records remain available.
No prior thesis/model is supplied in v1. No independent external verification is claimed.

## Persistence and recovery

Separate summary_comparisons table; immutable source and original snapshot. Each completed
part, consolidation and section checkpoints independently. Provider streams update progress
at most every 15 seconds. Non-end_turn results fail visibly, never save as complete. UI polls
while open and shows partial records. Resume saved comparison reuses completed steps; a
PostgreSQL advisory lock and worker token exclude duplicate/stale saves. Restart recovery is
explicit Resume, not an automatic paid replay. Provider keys are never persisted.

## Validation

- Five safe unit tests: lossless Unicode/long source partitioning, late-answer propagation,
  interrupted resume, retained full part records through hierarchical synthesis, no replay.
- scripts/verify-summary-comparison.py: disposable LOCAL PostgreSQL schema, mocked model,
  real HTTP/SQL checkpoints, duplicate exclusion, resume, baseline preservation, scoped feedback.
- Existing frontend suite: 29 passing tests. Frontend build and Python compile passed.
- Browser component preview: synthetic 250,001-character completion, section navigation,
  all eleven fixture records visible. No paid generation or production mutation performed.
- Still to validate: real note quality, long-source latency/cost,
  actual provider interruptions and mobile viewport behavior.

Review preview: http://127.0.0.1:8777/ (temporary local service, synthetic data only).
Production activation requires backend deployment and matching frontend/worker/service-worker
release IDs. Do not restart production during active audio/research work.

## Improved-note exports

Improved and Side by side views offer Save to iCloud, Download Word, Copy, and
Email me for all sections and each section. Exports require a completed version.
Word/iCloud read the exact comparison ID scoped to its parent summary, never the
original fields. Filenames include Improved and the immutable comparison ID.
The iCloud button queues the existing Mac export worker into SUMMARIES/Word Exports;
queue acceptance is not confirmation of a local file write. Email uses the existing
Settings credentials and labels subject/body Improved Notes. No email is sent by
rendering the controls or automatically completing generation. Timeout messages
warn of unknown delivery rather than automatically retrying a potentially sent email.

Validation: three isolated export-handler tests inspect real generated DOCX contents,
section selection, iCloud queue payload, scope rejection, and incomplete-version rejection.
Browser fixture verified controls and Copy. No live email or iCloud write used for testing.
