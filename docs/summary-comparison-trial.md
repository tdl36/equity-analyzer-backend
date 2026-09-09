# Full-transcript comparison trial

Status: approved for opt-in deployment September 9, 2026. Existing generation prompts,
original notes, and automatic iCloud ingestion remain unchanged during the comparison trial.

## Entry point

Summary → open an existing saved note → Compare improved notes → Generate improved comparison.
The alternative uses the entire saved raw_notes text, not a new audio transcription. The existing
output is frozen as the baseline when a comparison is created. Different source text creates a
new comparison; same source/version coalesces. No original note is overwritten, published,
emailed, or made the default. The comparison can be reviewed alongside the original and
feedback saved over multiple days.

## Scope of full-source processing

The NEW path has no total 20,000 / 50,000 / 200,000 character cutoff. Lossless partitions
bound individual model requests; every part is processed and its detailed management record
saved. Analytical sections consume all part records, with recursive consolidation when needed.
This is hierarchical synthesis, not a claim that a model has unlimited context or that generated
records cannot omit a detail. Exact raw source is frozen privately with a SHA-256 fingerprint.
Existing baseline generation still has its historical limits during this controlled comparison.
Consequently a comparison measures both source-coverage and editorial changes, not prompt
changes alone. Input must already exist as saved source text; this cannot recover missing audio
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
- Still to validate after approved activation: real note quality, long-source latency/cost,
  actual provider interruptions and mobile viewport behavior.

Review preview: http://127.0.0.1:8777/ (temporary local service, synthetic data only).
Production activation requires backend deployment and matching frontend/worker/service-worker
release IDs. Do not restart production during active audio/research work.
