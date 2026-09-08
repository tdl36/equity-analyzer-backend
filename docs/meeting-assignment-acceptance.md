# Real meeting assignment acceptance

Start one real, user-requested assignment from **Research desk → Command Charlie →
Prepare for a meeting**. Select a covered company, meeting date, source window and
focus. Keep its command, refresh and recap-job IDs from the collection receipts.

The acceptance checker is read-only and does not start model work:

```sh
.venv/bin/python scripts/verify-meeting-assignment.py \
  --command COMMAND_UUID --refresh REFRESH_UUID --recap-job RECAP_JOB_UUID
```

Exit 0 means all delivery checks passed; exit 2 means at least one remains
incomplete. The check requires matching assignment identity, completed collection,
all imported originals in the recap input record, the same source set in Meeting
Prep, a matching saved question-set receipt, and supported filenames, rationale,
priority and follow-up on every question. It reads the existing local ledger and
Charlie APIs using the configured agent authentication. It does not alter either.

After mechanical acceptance, open the saved pack through **Open meeting pack**.
Review questions against their original sources: figures, units, periods, company
guidance versus broker estimates, unresolved disagreements and historical thesis
context. Check that planned prior questions are not described as answered meetings.
A correct source filename alone does not prove that the question's premise is true.

Record source-collection time separately from recap and question-generation time.
A single-company run does not establish conference-batch throughput, unattended
collection across all coverage, recovery after every failure mode, or expert-level
investment quality. Those need separate acceptance runs.

Managed question generation uses schema-constrained JSON, exact source-filename
choices and a concise 12–15-question brief. Incomplete output stops before saving.
A failed question stage retains document analyses and synthesis for explicit retry.
The saved question-set model records the actual question generator; document
analysis and synthesis retain their existing routing. Provider schema reference:
https://platform.claude.com/docs/en/build-with-claude/structured-outputs

The Meeting Prep detail screen checks managed-job state independently of the tab
that started it. Active or unavailable status suppresses a duplicate-start button;
a completed managed job reloads the saved question set automatically.
