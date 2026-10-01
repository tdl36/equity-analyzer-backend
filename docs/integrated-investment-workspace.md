# Integrated Charlie investment workspace

Design and implementation specification — October 1, 2026.
Status: first foundation increment implemented (T115); Release 1 remains incomplete.
Implementation progress: saved-case Snapshot, historical wording comparisons,
current-wording excerpt links and versioned HTML investment-map export. Existing
evidence review remains the Update Thesis entry point. Durable full research
revisions, source inventories, issuer resolution and Deep Research remain next.
No paid validation has been performed.

## Scope and source material

The user requests that both shared designs become an integrated part of Charlie:

- Full research and visual workflow: https://chatgpt.com/share/6abe4956-d8d4-8329-ab20-07c2004d3440
- Investment committee and scenario extension: https://chatgpt.com/s/t_6abe48b29f388191a089271d8e2a5a3c

Reviewed the full 22-section master prompt, subsequent product discussion, generated
infographic reference, and downloaded stock-research-studio.zip. Inspected its
server.mjs, research-config.mjs, public/app.js, package.json and README.md without
executing it. The nine original screenshot attachments were not individually
transcribed; the synthesized master prompt is the requirements baseline.

The prototype is reference material, not proof of successful live model calls.
Its default model identifiers and provider parameters require validation against
Charlie-supported configurations before reuse. No new provider is required by
this design.

## Product outcome

A company has one persistent research workspace. A user enters a ticker, reviews
the resolved issuer/security, and chooses Snapshot, Deep research, or Update thesis.
Research, committee findings, scenarios, questions, monitoring and visual exports
refer to explicit versions of the same underlying evidence and assumptions.

Snapshot is a concise view of a saved research revision, displaying its date and
coverage gaps. A refresh is an explicit operation, not an invisible paid run on
every view. Deep research builds or refreshes the full research record. Update
thesis compares new evidence against a selected accepted case revision; it does
not overwrite that case. If no baseline exists, offer an initiation workflow.

The primary company navigation is Overview, Research, Investment case, Committee,
Scenarios, Investment map, and History. Evidence and sources are accessible from
every claim and through a shared drawer. Detailed research sections live within
Research rather than becoming 22 top-level navigation items. Views need stable
routes, browser back behavior, and readable mobile layouts.

## Requirements coverage

| Master prompt section | Canonical content / destination |
| --- | --- |
| 1 PM summary | Derived concise overview, core question, decisive drivers |
| 2 Business model | Segments, products, customers, geography, economic driver relationships |
| 3 Industry | Competitive relationships, durable versus temporary advantages |
| 4 Historical financials | Five fiscal years plus LTM where supported; sector-specific metrics |
| 5 Segments | Revenue/profit contributions, margins, KPIs and mix effects |
| 6 Latest earnings | Actuals versus prior guidance, dated expectations and comparable periods |
| 7 Estimates/revisions | Point-in-time estimate observations, definition and contributor scope |
| 8 Management | Dated promises versus outcomes and capital-allocation evidence |
| 9 Balance sheet | Liquidity, obligations, maturities and downside capacity |
| 10 Cash flow | Reconciled accounting-to-cash bridge and earnings-quality concerns |
| 11 Valuation | Appropriate methods with explicit inputs, denominators and dates |
| 12 Embedded expectations | Conditional reverse valuation; distinguish from observed consensus |
| 13 Peers | Economically relevant peers with selection rationale and comparable definitions |
| 14 Scenarios | Operating assumptions, calculated financial outcomes and valuation |
| 15 Catalysts | Confirmed/estimated timing, affected drivers and expectations |
| 16 Risks | Mechanism, exposure, horizon and observable early warnings |
| 17 Variant perception | Dated baseline, differentiated assumption, evidence and recognition path |
| 18 Blind spots | At most five material hypotheses, magnitude and disconfirmation tests |
| 19 Thesis monitor | Explicit metrics, thresholds, observations and evidence freshness |
| 20 Technical context | Optional, separately sourced price/volume context |
| 21 Gaps/questions | Missing evidence and management questions linked to decisions |
| 22 Final framework | Derived synthesis of the same accepted/reviewed structured state |

Unknown and inapplicable are different states. Do not fabricate completeness by
filling missing metrics. Sector templates select appropriate metrics and formulas;
a REIT, bank, managed-care company and semiconductor business cannot all be
represented by the same industrial EBITDA model.

## Existing Charlie integration points

- deepdive.py / deepdive_prompts.py: company research generation and structured
  reports. Adapt into the shared record rather than introducing a competing app.
- research_evidence.py: source hashes, exact-passage provenance and quality status.
  Extend typed references instead of reducing provenance to a report bibliography.
- investment_case.py: accepted working assumptions, immutable revisions, stale-edit
  protection and deterministic EPS/P-E calculations.
- case_signals.py: dated observations, falsification tests, variant baselines and
  user-authored position diagnostics.
- investment_review.py: structured review, computed outcomes and skeptical review.
- recap_coordination.py: bounded challenge/editor stages and retained decisions.
  Current challenger sees the draft, not independently retrieved source evidence.
- research_commands.py and collection_refresh.py: task identity, collection and
  frozen source-policy workflow. Retain original restrictions and source receipts.
- pipeline_recovery.py / recap_checkpoint.py: useful ownership/recovery patterns;
  audit suitability before extending to new cloud committee jobs.
- onepager.py / src/onepager.jsx: visual research normalization, stored versions and
  thesis comparison. Extend rather than introducing a second visual source of truth.
- Meeting Prep, Summary Lab and Catalyst synthesis: producers/consumers of linked
  evidence and research; preserve their originals and existing entry points.

The legacy TradingAgents wrapper emits decision/trade-oriented outputs. It should
not define the new committee contract. New outputs support investor decisions and
never execute trades or automatically accept research.

## Prototype findings to resolve during integration

1. Update mode changes a prompt flag and accepts manually supplied thesis text;
   it does not retrieve a baseline or compute a durable thesis comparison.
2. Deep/update adds a critique to report.review. No resolution stage applies or
   rejects corrections before delivery.
3. infographicPrompt uses report fields without review findings or a readiness
   gate, allowing disputed content into a polished image.
4. Financial values and scenario results are primarily strings. A strict JSON
   shape alone does not validate units, periods, arithmetic or source support.
5. Source registry entries are not claim-level passage links. The proposed
   verification badges are not implemented as an evidence verification ledger.
6. Reports are local JSON files. No accepted-thesis pointer, durable job queue,
   stage resumption, cancellation, spending ceiling or duplicate request fencing.
7. Missing API configuration selects synthetic demo data. Charlie must never
   substitute synthetic data for failed live research; demo fixtures stay isolated.
8. Some generated content is not presented in the UI, including the management
   promises/outcomes table. Verify full requirement-to-schema-to-screen coverage.
9. The master prompt's EV multiple example needs explicit grouping:
   equity value per share = (EBITDA * EV/EBITDA - net debt) / diluted shares,
   with other claims/nonoperating assets handled explicitly where relevant.
10. The supplied poster is a useful hierarchy reference but very dense and uses
    generic placeholders. Charlie visuals must use actual company-specific
    products, segments and debates, with readable text and inspectable sources.

## Shared data contract

Use PostgreSQL initially. Introduce stable identities and immutable versions for:

- Company/security identity: issuer identifier, listing, ticker, currency, fiscal
  calendar and sector-template choice. Resolve ambiguous tickers before research.
- Source version: original content hash, extracted-text hash, publisher, publication
  time, retrieval time, covered period, usage rights, document locator and extraction
  warnings. Revised filings retain prior versions.
- Observation: metric definition, value/range, unit, currency, fiscal/calendar period,
  GAAP/adjusted basis, source version, locator, availability time and restatement state.
- Claim: statement, classification, supporting/contrary evidence and review findings.
- Research revision: section payloads, source-set hash, coverage, unresolved issues,
  run identity, provider/model/prompt versions and creation date.
- Accepted case: reuse existing case revisions and stable assumption IDs. Generated
  research and draft committee conclusions do not become accepted cases implicitly.
- Model version: input observation/assumption IDs, explicit formulas, dependencies,
  validation results and numerical outputs.
- Committee run: frozen research/case/model references, role outputs, challenges,
  responses, dissent, resolution and proposed case changes.
- Visual version: source research/case/model revision IDs, content selection and
  rendering version. Existing exports remain reproducible historical artifacts.

Relationships such as supports, contradicts, depends_on and affects are typed.
Documented relationships and modeled causal assumptions remain distinct.
Synthetic agent statements cannot be promoted to observed facts by repetition.

Store evidence classification separately from verification status. An estimate can
be accurately sourced; a calculation can use disputed inputs. Prefer specific
statuses such as passage matched, arithmetic checked, source conflict and unavailable
over an unqualified green verified badge.

Historical review uses information available at the selected cutoff. Store both
publication/availability and economic measurement dates; future filings and later
restatements must not leak into a historical run.

## End-to-end workflow

1. Resolve company and select mode, horizon, baseline, source scope and budget.
2. Reuse eligible current originals; collect missing material through existing
   managed paths. Show gaps, restrictions and local Mac dependencies explicitly.
3. Freeze an evidence snapshot; extract and reconcile typed observations/claims.
4. Run bounded research modules. Save partial results and coverage per section.
5. Validate numbers and source references, review meaning, and reconcile findings.
   Unresolved problems remain visible and constrain dependent outputs.
6. Produce the research revision, concise overview and proposed assumption changes.
7. Run optional committee review against the frozen snapshot. Start with five roles:
   lead, upside, downside, accounting/evidence, valuation/expectations. Initial
   assessments are independent; later bounded rounds address named challenges.
8. Compute scenario consequences in tested code. Agents propose inputs and explain
   mechanisms; they do not calculate final financial tables in prose.
9. Investor reviews proposed changes and explicitly accepts a new case revision.
   Recheck baseline version before acceptance; preserve intervening history.
10. Render report, visual map and meeting questions from selected explicit versions.
    Draft visuals are clearly labeled. Unresolved quantitative claims are omitted
    or visibly qualified, never made authoritative by their graphic presentation.
11. New evidence identifies affected assumptions and modules. Show what changed;
    create a new review version without silently overwriting accepted views.

Snapshot and exports should not rerun the whole research pipeline. A price update
can invalidate valuation outputs without regenerating the business description.
An evidence change can reopen a specific challenge without erasing its history.

## Scenario and visual design

First numerical release: transparent operating-driver bear/base/bull models and
sensitivity tables, with sector-appropriate methods. Reject incompatible units,
missing required inputs, circular dependencies and unsupported model methods.
Reverse valuation must identify which assumptions are fixed; it rarely has a
unique solution. Subjective scenario weights are labeled and validated.

Later Monte Carlo uses explicit distributions, correlations, constraints, seed and
model versions. Present conditional outputs, not calibrated forecast probabilities.
Stakeholder simulations create alternative response assumptions with observable
tests; they are never evidence of actual customer or regulator behavior.

The investment map has an overview/business-flow layer and a thesis/debate layer.
Clicking a segment or KPI reveals its definition, sources, linked assumptions,
contrary evidence and scenario consequences. Provide a compact one-page export
and a more detailed readable version rather than shrinking all 22 sections.

Render factual text, charts and numbers deterministically with HTML/SVG or a
document renderer. Optional generated illustrations must not encode financial
facts. All note/export text uses black Calibri, with light colored fills and
company-specific illustrations where useful. Preserve source/date/version labels.

## Delivery sequence and acceptance gates

### Release 1: unified research record and complete first user journey

Company resolution, existing-source selection, bounded structured research,
Snapshot/Deep/Update modes, section coverage, source links, saved revisions,
comparison against an explicit case baseline, investor acceptance and an initial
deterministic investment-map export. No new external data subscription is required.
Unavailable consensus/metrics are shown rather than synthesized.

Acceptance: one authorized company source pack can travel from intake to saved
research, baseline comparison, explicit case revision and matching visual output.
All outputs identify the same versions. Existing Charlie research remains readable.

### Release 2: source-backed investment committee

Independent initial role assessments, bounded rounds, linked challenges, preserved
dissent, explicit resolutions, proposed changes and cost/recovery controls.
Compare against the existing single-reviewer baseline on identical evidence packs.

### Release 3: operating models and expectations

Typed input series, accounting/cash bridges, first sector template, comparable peer
metrics, scenario calculations and conditional reverse valuation. Add licensed
consensus adapter only after provider access and point-in-time rights are confirmed.

### Release 4: ongoing research and visual integration

Dependency-aware updates, promises/outcomes tracking, monitoring observations,
meeting-question follow-through, selective committee reopening and visual diffs.
Extend sector templates based on real investor review.

### Release 5: advanced simulation

Selective stakeholder responses and Monte Carlo after model/data quality has been
demonstrated. More agent roles are added for a concrete evidence need, not spectacle.

Every release needs safe unit/frontend tests, build, desktop/mobile verification,
explicit versioned migration behavior and the repository release procedure. Paid
live research is user-driven; synthetic tests are labeled and never presented as
real-source proof. Do not run pytest.

## Reliability and evaluation

Persist stage identity and completion before advancing. Freeze source/case/model
versions; enforce ownership leases and idempotency for submissions and deliveries.
Bound rounds, concurrent work, tokens/cost and recovery attempts. Record usage for
failed paid calls. Ambiguous provider outcomes require inspection, not blind retry.
Cancellation stops future stages; it does not promise to undo an in-flight call.

Show queued, running, waiting for sources, needs review, partial, failed, cancelled
and completed distinctly. A completed section is not a completed assignment.

Tests must cover wrong issuer, stale or conflicting sources, source restrictions,
period/unit mismatch, missing consensus, citation mismatch, numerical reconciliation,
baseline changes during a run, duplicate submission, interruption/recovery, partial
collection, critic failure, stale visuals, and proposal acceptance concurrency.

Evaluate useful material findings, coverage, citation correctness, numerical
accuracy, retained uncertainty, reviewer judgment, latency and actual cost. Agent
agreement, report length and attractive graphics are not quality measures.

## Open implementation decisions

- Which available licensed dataset, if any, can supply usable point-in-time consensus.
- First sector-specific operating model and investor-selected real validation pack.
- User-selected recurring cadence and budget for new committee/refresh workflows.
  Existing automation policies do not automatically authorize expanded paid runs.

These do not block the shared record, existing-source workflows, versioning,
review gates, deterministic rendering or synthetic verification work.
