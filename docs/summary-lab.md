# Summary Lab — independent opt-in experiment

Open Library → Summary Lab (`#view=summarylab`). Local development uses
http://127.0.0.1:3000/?local=1#view=summarylab with the matching backend.

The lab reads a saved Summary's full raw_notes and freezes its original section outputs,
or accepts pasted source text. Each deliberate generation creates a separate experiment
in summary_lab_experiments with source SHA-256, prompt version, model, focus, stages,
results and feedback. No write touches meeting_summaries or summary_comparisons.
Existing automatic triggers, original prompts and companion generation are unchanged.

Pipeline: lossless source partitions → detailed source review with exact-substring
passage checks → all-record synthesis (hierarchical for large records) → five independent
sections → original-part model checks → targeted revision → cross-section reviewer note.
Completed stages checkpoint. Retry reuses records; a PostgreSQL session advisory lock
excludes competing workers. Two generation slots per backend process. API keys are
never persisted. Experiment creation is not idempotent: a user-requested rerun is a new
experiment. After an uncertain start response check history before retrying.

Copy section/all, JSON download including frozen original and source, side-by-side
comparison, source record, check findings and user evaluation are available. Results
are saved automatically in the lab. No automatic email or iCloud export is performed.

Limitations: text input only, no audio segment re-listening or page-image OCR verification;
no external fact-checking, live model/thesis baseline or verified citation for every claim.
Exact passages are checked, but model-authored records and judgments can be wrong.
All source parts receive initial draft checks; the revised result receives a cross-section
review against reviewed records, not a second full original-part audit. Reviewer notes
remain visible; “ready” means generation completed, not proven error-free. No automatic
restart replay: users explicitly resume stale jobs. No live quality benchmark completed yet.
The legacy HTML original is sanitized by the existing application renderer.

Validation (safe, no live database fixtures):
- PYTHONPATH=. .venv/bin/python tests/unit/test_summary_lab.py
- .venv/bin/python scripts/verify-summary-lab.py (disposable LOCAL PostgreSQL schema)
- npm run test:frontend
- npm run build
- Desktop 1440 and mobile 390 component fixtures: comparison, section navigation,
  feedback save, no browser exceptions or horizontal overflow.

Production release T74 includes the iCloud document picker. It uses the shared connected STOCKS/CATALYSTS manifest, fetches selected originals through the existing agent bridge, and extracts each file separately. An unreadable selection prevents the import from replacing the previous source. Audio input uses a previously saved Summary transcript. Backend must include summary_lab registration
and frontend must include the Summary Lab route together before production use.

T76: Lab sections render as escaped, formatted documents in 11pt Calibri (Carlito/Arial
fallback). Email all sections opens an editable preview and sends only after the user's
Send action, to their Settings recipient using existing Gmail credentials. Preview edits
are temporary and never overwrite saved results. Email excludes private reviewer/source
records; source labels and caveats in sections remain. Copy includes HTML and plain text.
The Lab-only email template uses inline typography and an allowlisted HTML vocabulary.
Prompt v2 reduces formulaic scaffolding and duplicated Q&A for new runs, retaining
management substance and uncertainty. Saved v1 experiments are not rewritten.
