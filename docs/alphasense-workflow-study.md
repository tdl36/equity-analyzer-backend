# AlphaSense study and Charlie implementation

Reviewed September 15, 2026 through the signed-in AlphaSense account, existing exported output, official product documentation, and Charlie's source code.

## The central finding

AlphaSense makes the deliverable the organizing principle: select a named task, provide a company and context, choose an output, and review the resulting artifact beside a follow-up conversation. Charlie has many of the underlying research and automation capabilities, but they are scattered across separate views. The best improvement is to connect those capabilities around a finished research deliverable while retaining Charlie's investment framework, source preferences, version history and explicit thesis acceptance.

Charlie should compete on how evidence changes the investor's decision, not on the number of agent names in a gallery.

## What was verified

| Area | Observed in the account | Implication for Charlie |
| --- | --- | --- |
| Main research entry | Report, Grid and Slides outputs; context, source and thinking controls | Keep the task, source scope and output format together |
| Agent library | Search, categories, output types, user agents and scheduled agents | Offer discoverable starters with visible prerequisites and outputs |
| User agents | Research template, guidance credibility audit, daily broker changes, weekly news roundup | Reusable instructions should support ticker/date variables and saved preferences |
| Custom agent setup | Title, description, company/watchlist mentions, variables, references, source filters, instruction improvement and output selection | Separate reusable recipe configuration from one run's ticker, date and sources |
| Scheduling | None, weekdays, daily, weekly and monthly; account displayed 0/10 scheduled | Reuse Charlie's durable scheduler and make next-run, login-blocked and recovery states visible |
| Source controls | Expert insights, broker research, company documents, news, regulatory and intellectual property; web search toggle | Show source provenance and availability; distinguish source categories from broker preferences |
| Earnings Recap Deck | Company selection and slide-template choice before generation | Provide a direct earnings-event-to-deck action |
| Existing CVS deck | Report canvas with follow-up chat; PowerPoint and PDF export | Let users iterate on an artifact while preserving previous versions |
| Exported PowerPoint | 12 slides, 4:3, predominantly native editable text and shapes, source appendix | Prefer native slide elements for research presentations |
| Readability | The inspected executive slide was dense; several slides had over 3,000 characters | Prefer readable pages and continuation slides to squeezed text |

The existing CVS output was inspected as a product example, not fact-checked as investment research. Its licensed content is not copied into repository fixtures. No AlphaSense agent was saved, scheduled or changed, and no new AlphaSense deck generation was initiated during this review.

Official documentation additionally describes financial data in slides, reusable uploaded style templates, sentence-level source links, and native PowerPoint/Excel tools. Template upload and those add-ins were not exercised end to end in this account. AlphaSense documents that style extraction does not yet mean filling an existing slide template or reproducing a specific slide with a different subject.

Sources:
- [Creating Slides in Generative Search](https://help.alpha-sense.com/hc/en-us/articles/52311020704915-Creating-Slides-in-Generative-Search)
- [PowerPoint and Excel capabilities](https://www.alpha-sense.com/resources/product-articles/alphasense-powerpoint-excel/)
- [Company Profiles](https://help.alpha-sense.com/hc/en-us/articles/42623871994131-Company-Profiles)
- [July 2026 product updates](https://help.alpha-sense.com/hc/en-us/articles/53942181071123-AlphaSense-Product-Updates-July-2026)

## Implemented in this change

### Earnings event → editable presentation

In Research desk → Earnings & evidence, open an event with a completed saved recap and use **Build earnings deck**.

- HTML and Markdown recaps are supported. For multi-format recaps, the comprehensive version is selected and disclosed, avoiding repeated short/PM/full versions.
- Full recap or section brief. Full mode retains recap text through continuation slides; brief selects the opening sentence of the first two paragraphs per section and discloses that selection. Complete original section text remains in notes on the section's first slide.
- Three slide styles: Editorial paper, Midnight boardroom and Sage research.
- A slide outline, readable preview, per-slide wording edits, copy-all, and native editable 16:9 PowerPoint export.
- Versioned server snapshots containing the recap hash, event identity and recorded source metadata. Regenerating a recap does not silently rewrite an existing presentation.
- Frozen source registers, disclosure of analyst edits, saved-version browsing and export of a specific revision.
- Existing numerical comparisons and selected-claim review counts become explicit review slides when present. Their saved status is carried through; deck formatting does not rerun verification.
- Idempotent deck creation and optimistic revision checks. Unsaved edits block export and deck switching; errors and uncertain responses remain visible.

This is a deterministic presentation workflow using existing research. It does not generate a fresh investment narrative, search sources, invent citations, or approve investment conclusions. Source filenames are an input register, not sentence-level citation proof. Long source names and edited text can require extra PowerPoint continuation slides; the web view is a reading/editor preview, not a pixel-identical PowerPoint rendering.

### Research starters

Command Charlie → Investigate & update research now includes earnings preview, earnings recap, broker reaction digest and guidance credibility review. A starter fills the existing task controls; users set ticker/date/window, apply source preferences and explicitly start the assignment. These are instruction templates over Charlie's existing collection and analyst pipeline, not separate new autonomous executors.

The instructions preserve distinctions between management statements, broker interpretation and analyst assessment; require matching periods/units/accounting basis; and forbid presenting one broker's estimate as consensus. Guidance credibility explicitly acknowledges missing history rather than inventing a multi-quarter record.

## Next implementation sequence

1. **Source-grounded slide composition.** Generate a structured storyline from the actual saved sources and thesis context, then validate claim-to-passage support before layout. Each claim needs original document identity, page/passage, as-of date and fact/interpretation classification. Unsupported numbers should remain absent or visibly unresolved.
2. **Decision-oriented earnings templates.** Executive change, results versus matched expectations, segment drivers, guidance bridge, broker disagreements, thesis-pillar implications, unresolved questions and next signposts. Differentiate operating outcomes from valuation implications and missing consensus from a genuine beat/miss.
3. **Native charts and tables.** Build from validated structured values with explicit period, currency, scale, basis and benchmark date. Reconcile totals before export. Never fabricate a chart from prose that lacks the required series.
4. **One research workspace.** Company/event, source tray, report/grid/deck tabs and artifact-specific discussion. Debate a claim or revise a slide without rerunning unrelated sections. Preserve a visible before/after diff and acceptance boundary.
5. **Reusable templates and schedules.** Support approved corporate slide masters, reusable output recipes, named source selections, and scheduled refreshes with clear prerequisites. Separate template style from investment methodology.
6. **Comparison grids.** Broker-by-broker estimate and rationale changes; guidance-to-outcome history; management Q&A themes across quarters. Missing values must stay missing and replicated commentary must not masquerade as independent corroboration.
7. **Source-to-artifact maintenance.** A new source should identify which report paragraphs, slide claims and thesis signposts may be stale, propose a targeted refresh, and preserve an audit trail. No automatic thesis acceptance.

## Acceptance gates for the next stage

Use several real earnings events with different data completeness, plus synthetic failure fixtures. Compare against the same-source AlphaSense output on factual support, quantitative reconciliation, broker attribution, thesis relevance, readability and editing effort. Record missing documents and exact source cutoffs. Test interrupted runs, duplicate triggers, stale saves and mobile review. A good-looking slide deck is not evidence that its reasoning is correct.

## Validation for this implementation

- Isolated Python tests for long recaps, brief disclosure, immutable sources, invalid edits, native PowerPoint output, duplicate creation, stale revisions, version history and export.
- Route-mocked browser checks across seven Charlie themes and four viewport widths, covering build, edit, save, export gating and conflict feedback. These use synthetic data and do not prove live model or collection behavior.
- Synthetic decks rendered through LibreOffice, reviewed visually, then adjusted for explicit left alignment, font fallback, shadows and footer legibility.
- Production frontend build and existing regression suites run separately; see task completion for final results and deployment state.

Live smoke test: the deployed application saved and exported a real CVS recap as a 58-slide full-coverage deck with native editable shapes and speaker notes. The original brief was too long (38 slides), prompting a change to opening-sentence extraction. This demonstrates formatting/persistence/export, not a real-source factual quality benchmark.

Final formatting checks preserve financial abbreviations such as “adj. EPS” and use glyph-width-aware continuation splits. The corrected CVS brief was 17 slides in the export-derived layout fixture, with complete valuation sentences. Final live verification is recorded in the task response.
