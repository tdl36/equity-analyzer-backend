# Catalyst synthesis comparison trial

Scope: only automatic CATALYSTS recaps and manual Catalyst synthesis. Summary generation, its prompts, controls and existing saved outputs are unchanged. Legacy catalyst prompts and `markdown` are retained for comparison.

## New flow

1. Produce the existing catalyst recap normally.
2. Extract the complete readable source text by page. Partition long pages/text into processing segments without a total character cutoff. Preserve original page numbers for split PDFs. Extraction limitations remain visible.
3. Extract material source records with speaker attribution, interpretation, uncertainty and follow-up. The model selects numbered source spans; Python retains the original passage rather than asking the model to retype quotations. Match each quotation against the source text. Independently review each whole record. Attempt one repair and review again; retain matched original source wording when a paraphrase still fails, label it as an uninterpreted excerpt, and disclose the failed check. Omit claims with no matched source passage.
4. Freeze the supplied thesis and source passages from up to three recent approved catalyst recaps whose improved version was explicitly saved for the same ticker, excluding the current activity. Compare records with every baseline partition. Require an exact baseline passage and a separate model support check before including a comparison. Without a baseline, explicitly say no thesis comparison is established. This does not retrieve historical market consensus or backfill older original documents. Previously accepted source passages become available as improved recaps accumulate.
5. Render all improved views from the same checked records. Store them alongside the original in `catalystComparison`. Completed extraction/review segments are checkpointed for reuse after interruption. An improved-run exception preserves the original and displays a failed trial state.

## User experience

The improved comparison panel appears in catalyst results, analyst recap review/archive and Earnings & evidence. It opens on Brief; Source record, Interpretation and Full note are separate buttons. Copy/email operate on the selected section. Save full improved note writes a separate Research → Catalyst Synthesis document with an `-improved` ID; it never overwrites the original or writes to Summary.

Old recaps are not automatically regenerated. Re-run a recap after the updated backend and local agent are deployed to produce both versions. This incurs extra model calls. No research or thesis is accepted automatically.

## Limits and validation

Text matching and model review are support checks, not proof that a management assertion is true. The extraction model can omit a material point; real-source coverage review remains necessary. The Brief selects seven material records by ID, without independently rewriting their facts; full coverage remains in the Source record and Full note. A numerical ambiguity stays unresolved; guessed corrections are prohibited. Novelty requires context and is never inferred from absence in a baseline partition.

Unit tests cover lossless long inputs, fabricated passage rejection, missing/duplicate review verdicts, repair/recheck, pagination, HTML escaping and failure isolation. Run with unittest, never pytest against this repository's live database fixtures.

Deployment must update the backend, local agent and frontend together. A built frontend alone does not activate generation on the running agent.

## Implementation verification (2026-09-15)

- Production frontend build passed.
- 16 catalyst comparison/route tests, 25 existing recap tests and 6 existing catalyst activity/delivery tests passed. These use isolated fixtures/fake cursors, not live database test fixtures.
- 38 existing frontend tests passed.
- Isolated browser checks at 390px and 1440px passed: layout without horizontal overflow, section selection, distinct improved-save request, email payload and readable copied text. Email/save requests were mocked; no email was sent by the test.
- Existing top-level prompt/rule constants in the backend and local agent were compared with the starting commit and remained identical. Summary modules have no diff.
- A real 3M conference-source trial exposed repeated repaired records and over-aggressive omission. The implementation now deduplicates exact repeated statements, selects original source spans and retains matched original wording when the paraphrase cannot be supported. Opening document context is supplied to preserve date/speaker interpretation across pages. This is an iterative trial, not a claim of error-free extraction.
- Changes remain local and uncommitted pending review of the trial outputs; the running production agent is not switched by a frontend build.

## Investor-note revision (local, September 16)

The initial evidence inventory was not a finished PM communication. A separate editorial
pass now writes two independently useful deliverables: PM takeaway and Detailed note.
Evidence & review is explicitly a private audit view, not another summary. Participants
appear once per note; filenames are listed once in a source key and short page references
remain beside supported paragraphs. Material watchpoints are consolidated, not repeated
beneath each fact. The detailed note is newly synthesized by theme, not concatenated tabs.

Editorial preparation reads every source record, consolidating batches for long inputs.
Final paragraphs carry validated record IDs; a separate model checks original quotations,
attribution and timing. One repair is allowed, followed by another check. Failure retains
the source audit and original recap, without enabling share/save of an unchecked note.
These are model support checks, not proof of completeness or external factual accuracy.
Saved legacy trial annotations remain visible as prior machine findings, explicitly not
current conclusions; later passages may resolve them.

Email/copy select only investor-facing views; save stores the standalone detailed note
instead of the full audit bundle. Summary and Summary Lab are unchanged by this revision.
No production deployment or automatic rerun is implied by this local implementation.

### MMM investor-note acceptance draft

September 16 local trial: the first fully model-written editorial revision failed its
final source check after one repair. Do not describe this as a successful end-to-end
automated generation. The exception now retains accepted sections, rejected candidate
and specific findings; repairs also receive original quotations, not just consolidated
source notes. Future review batches contain up to 12 paragraphs (each requires a verdict).

A separately authored editorial acceptance draft is saved as investor-catalyst-v2.json/html
in the existing Charlie/Reviews/2026-09-15-MMM-catalyst folder. Its PM takeaway is 470 words;
the detailed note is 1,019 words. A separate Opus source review returned supported verdicts
for all 23 prose blocks; exact results are in investor-catalyst-v2-review.json. This is a
model support check, not external verification or proof of exhaustive source coverage.
The draft demonstrates the intended content/UX standard; the revised automated writer
still requires a successful end-to-end real-source run before deployment.

25 safe unit/route tests pass. Desktop 1440px and mobile 390px checks passed with the
actual edited draft: three views, a single source key per note, hidden private-review
sharing controls, HTML export excluding audit material and no horizontal overflow.
Original source/trial files, Summary and Summary Lab are unchanged. No commit/deploy.
