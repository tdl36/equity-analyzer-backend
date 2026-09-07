# Evidence and changes — first release

Research Desk → Evidence & changes reads the latest two saved investment reviews for a ticker. It preserves all saved theses, reviews, analyst edits, and iCloud files. A missing investment review is explicitly distinguished from a missing thesis.

New investment reviews persist a versioned evidence snapshot in their existing metadata JSONB. Source identities bind the captured document payload and extraction with SHA-256. Thesis statements, facts, changes, KPIs, estimates, and scenarios can reference these sources by ID and contiguous quotation. Whitespace-normalized matching requires at least 30 characters. Quotes with unknown IDs or fabricated text fail. Text matching establishes provenance, not entailment, accuracy, consensus, or publication date. No page numbers are invented. The independent reviewer sees the claimed excerpts and matching results alongside the state and arithmetic.

Readiness remains `needs_review` for missing evidence, absent/malformed independent QC, a revise verdict, medium/high findings, arithmetic inconsistencies, or omitted selected documents. `checks_passed` means automated checks passed; analyst review is still required. Generation completion is a separate processing state. An unreadable selection stops before any model call. Findings are deduplicated and included in report rendering. Older reviews receive an unchecked status when read; they are not rewritten or backfilled.

The authenticated, read-only `/api/research/evidence/<ticker>` endpoint uses the existing global auth gate, parameterized SQL, a two-record limit and no-store responses. It does not load document payloads or report PDFs. No new migration or mutation endpoint is required.

## Validation

Standalone unittest suites cover fabricated quotes, unknown sources, insufficient excerpts, source version changes, missing per-claim references, failed QC, dropped documents, legacy records, malformed collections, read-only route behavior, ticker validation, and the actual review orchestration with stubbed model/database/rendering boundaries. No paid model jobs or production research writes are used in QA. Browser QA uses a clearly labeled synthetic fixture to exercise selection and the source inspector; production read-only checks follow deployment.

## Boundaries and next steps

This release is the evidence foundation for investment reviews, not the full ten-priority roadmap. It compares stored review versions; it does not yet reconcile new iCloud events, apply thesis amendments, provide page-level original viewers, or share a canonical company state across all generators. The existing first-batch coverage limit remains explicit. Follow-on work: all-document extraction and reconciliation, claim entailment evaluation, deterministic financial-period/basis checks, shared fact/version IDs across thesis/notes/one-pagers, and a conflict-aware amendment review flow. Automated checks are not a guarantee of error-free investment analysis.

## Saved-research fallback

Evidence & changes now also reads the company's saved thesis and the uploaded-document inventory, plus a whitelisted snapshot of the local agent's iCloud inventory. Without an investment review it renders that research directly, stating that no investment-review comparison baseline exists. With a review, saved company research is available in an expandable section. Both source locations are listed separately (not counted as unique documents), local inventory timestamps are visible, and an absent manifest is described as unavailable rather than empty iCloud folders. Inventory presence never implies claim verification. This path is read-only and never generates a report.
