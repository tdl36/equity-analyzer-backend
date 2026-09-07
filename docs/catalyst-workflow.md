# Catalyst source workflow — September 7, 2026

The MDT F1Q27 recap failed because the event directory contained a second directory with the same name and the old synthesis reader only inspected immediate children. The source files were subsequently moved to the outer directory. The fix supports both structures without moving originals.

Detection, the topic picker, and generation now share `catalyst_sources.py`. Sources are scanned recursively with relative names. Hidden files, symlinks, Processed directories and generated RECAP_/SYNTHESIS_ files are excluded. PDF validation and nonempty text checks happen before generation. A failed selected source blocks the complete request with actionable filenames, rather than silently omitting evidence. Presentations (PPTX text), Word tables, HTML and XLSM are supported. Legacy DOC/XLS/PPT and unextracted ZIP files require conversion/extraction. Cloud placeholders are surfaced; automatic iCloud hydration and OCR are not implemented. PPTX charts/images need a PDF export for visual analysis.

Automatic detection uses relative names, byte sizes and nanosecond modification times, plus a two-minute quiet period measured from observation (not just a downloaded file's old publication timestamp). Existing compatible fingerprints migrate without replaying the inventory. Generated recap saves do not cause another synthesis. Per-analyst earnings/takeaway auto settings remain respected; changes do not enable paid generation for every analyst retroactively.

Backend dispatch prevents concurrent runs of the same activity; regeneration preserves prior completed output history and refuses to clear a live run. Routing is serialized and deduplicates a source revision across statuses and days. Approve & Save requires a completed recap and no longer queues an older generic catalyst proposal. Late completion callbacks cannot overwrite a newer run or an approved activity. Approval/export remains the existing manual workflow.

Remaining audit findings: global catalyst auto mode and per-analyst auto mode are separate controls and may generate both generic and analyst-specific outputs; long model calls are process-bound rather than resumable; source provenance is model-generated, not an independent verification of each claim; iCloud-only placeholders need materialization; automatic web acquisition needs configured source URLs. These are not claims of an error-free or fully autonomous research system.

# AlphaSense

Added idempotent full-universe planning (12 tickers per batch), retaining fixed date windows and distinguishing queued searches from reviewed searches. The remaining 49 tickers for August 7–September 6 are now queued in five runs. The monitor shows cross-run coverage. This does not mean the 49 have been downloaded or that a recurring unattended browser daemon exists.

Event-specific collection can now target an existing event directory:

    .venv/bin/python charlie_collector.py create --tickers MDT --since 2026-08-07 --until 2026-09-06 --topic 'MDT F1Q27 Earnings'

The destination is stored on the run and survives restarts. Validated eligible originals go to CATALYSTS/MDT/MDT F1Q27 Earnings; restricted originals remain in staging. Manifest verification understands the catalyst folder label. Normal coverage runs continue to use STOCKS. Browser company/date/type confirmation and export remain supervised; no credentials or undocumented authenticated APIs are introduced.

Completed recaps now enter a private local outbox before upload. Failed deliveries retain the original output and retry from the heartbeat without another model call. Duplicate or superseded result callbacks do not re-notify or overwrite an approved draft. Claim conflicts are no longer bypassed through a progress-update fallback. PDF batching uses a page-based estimate instead of encoded PDF byte size; this is an estimate, not a guarantee for unusually dense documents.
