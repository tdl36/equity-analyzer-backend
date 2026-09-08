# Meeting workspace reliability — September 8, 2026

This increment addresses the six meeting-workspace priorities. It does not certify
full unattended AlphaSense coverage or investment accuracy.

## Delivered behavior

1. Both manual-source and managed assignments share company-specific question
   quality instructions: probing inconsistencies, decision relevance, private
   attribution, concrete evasive-answer follow-ups, and planned-versus-answered
   history. New manual jobs freeze server-owned documents and use the managed
   passage-ID generator. Text-only sources disclose the missing original. Existing
   in-flight legacy jobs retain their original generation path on retry.
2. Saved packs expose mechanical quality diagnostics and a repeatable read-only
   audit script. Diagnostics include source coverage, incomplete/duplicate questions,
   passage records, and numeric strings missing from attached passages. Numeric
   flags require human interpretation: requested targets, periods and calculations
   can legitimately differ from quoted text. These are not factual verdicts.
3. Manual jobs use PostgreSQL connection-scoped ownership and a durable 30-second
   recovery sweep, capped at two automatic recoveries. Active owners exclude
   duplicate execution. Per-document checkpoints survive interruption. Owner tokens
   prevent stale saves; an atomic saved-question-set receipt prevents duplicate
   versions after interrupted delivery. Explicit retry cannot replay a completed job.
   My work includes manual jobs. Polling displays elapsed/checkpoint time and the
   output destination. Recovery needs a configured server research key.
4. Completed packs accept a plain-language revision request using the selected
   documents, format and audience. The full prior pack and instruction are frozen.
   A revision creates a new saved version. The prior version remains available;
   optimistic version checks stop a revision from silently superseding intervening
   edits. This is asynchronous revision, not token-streaming chat or an approved
   thesis update.
5. Collection controls expose scheduled/verified policy counts and overdue or
   attention-paused requests. Presentation collection is configurable alongside
   transcripts, releases and broker reports. Existing schedules, event detection,
   source decisions and receipt-based handoffs remain in place. A connected Mac or
   an awake scheduler is not proof of browser execution. The current Codex heartbeat
   depends on the current task being available; background collection can wait while
   this task is busy. Complete unattended coverage still requires an operational soak.
6. Format and audience preferences persist on this browser across both entry
   points. Meeting mode shows must-ask questions, expandable evidence/follow-ups,
   and management-answer capture. Answers save into the existing meeting history.
   Versions can be inspected without overwriting the current pack. Preferences do
   not yet sync across devices; no offline answer-save guarantee is made.

## Verification

- Safe unit and frontend suites (never pytest).
- `scripts/verify-manual-meeting-recovery.py` creates and removes a disposable
  LOCAL PostgreSQL schema, using no production DB URL and making no model calls.
  Checks active-owner exclusion, retained partial checkpoints, ownership rotation,
  stale-write rejection and completed-job replay protection.
- `scripts/audit-meeting-pack.py --meeting 26`: ABT, 12 questions/6 topics,
  12 with passage records, seven of nine sources cited, no exact duplicates,
  unknown filenames or incomplete questions. Numeric flags still need review.
- `scripts/audit-meeting-pack.py --meeting 27`: MDT, 24 questions/9 topics,
  24 with passage records, six of eight sources cited, no exact duplicates,
  unknown filenames or incomplete questions. Source coverage is not guaranteed by counts.
- UNH meeting 28 was interrupted at 9/10 analyses by the previous backend restart.
  Explicit retry resumed the same job and reached synthesis and question generation.
  Final delivery and new-release live checks are recorded after deployment.

## Still requiring operational proof

Fresh collection across the full covered universe; authentication pause/resume
under actual expiry; long-running unattended throughput; factual/sector-expert
review across companies and formats; new manual grounded generation and revision
on real sources; mobile viewport and cross-device preference behavior. Existing
ABT/MDT delivery receipts are evidence for those runs only.
