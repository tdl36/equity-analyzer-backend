# Charlie AI engineering handoff

Updated: September 20, 2026

## Start here

Charlie production is currently **T97** at commit **`0cb42303dfc448af56e99e87f38e32ca4bdf4546`** on `main`.

- App: `https://charlie-deployment.tonydlee.workers.dev/?release=T97`
- Backend health: `https://equity-analyzer-backend.onrender.com/health`
- Repository: `/Users/tonydlee/Projects/equity-analyzer-backend`
- Branch: `main`
- Backend health was verified on September 20 and reported the T97 commit above.
- The Mac launch agent `com.charlie.local-agent` was restarted during T82 because `charlie_local_agent.py` changed. T83 is frontend-only and did not require another restart.

Read `AGENTS.md` before changing anything. Preserve unrelated dirty and untracked files. Do not clean the repository.

## Product intent

Charlie is intended to become the primary research and decision-support workspace for a professional equity investor. It should ingest evidence, preserve source provenance, synthesize information in the investor's voice, maintain investment theses longitudinally, prepare meetings, coordinate bounded research agents, and automate repetitive work without silently making investment decisions or accepting research changes.

The design standard is institutional: concise hierarchy, readable outputs, defensible claims, explicit uncertainty, durable recovery, and usable desktop/mobile interfaces.

## System map

| Layer | Current implementation |
| --- | --- |
| Frontend | React 18. Most legacy UI and routing remain in `src/app.jsx`; newer workspaces use separate `src/*.jsx` modules. |
| Frontend build | `build-frontend.sh` runs Babel with required block-scoping transformation, then esbuild. Generated assets are committed in `build/` and `dist/`. |
| Edge | `worker.js` hosts static assets on Cloudflare Workers and proxies `/api/*` to Render. `service-worker.js` manages client caching. |
| Backend | Flask/PostgreSQL. `app_v3.py` remains the large central application; newer workflows are isolated in modules. |
| Mac agent | `charlie_local_agent.py`, managed by launchd as `com.charlie.local-agent`. It scans iCloud, handles local sources/audio/catalysts, syncs manifests, and participates in managed collection/recovery. |
| Primary storage | Production PostgreSQL plus user originals under iCloud `STOCKS`, `CATALYSTS`, and `SUMMARIES`. |
| Research collection | Managed AlphaSense browser workflow requires the Mac, signed-in Chrome, and scheduled worker availability. Public SEC/FDA/ClinicalTrials supplements are bounded and provenance-checked. |
| Production deployment | Push to `main` triggers Render backend deployment. Cloudflare frontend deployment is explicit through Wrangler. |

## Latest production changes

### T97 — spoken rate-cycle dates, the corrections log, and tier agreement

Commit: `0cb4230` — `Read spoken rate-cycle dates, and stop the corrections log becoming a glossary`

Three defects found in the real CNC management-meeting note. None were caused by T94/T95;
all pre-dated them in the original Summary prompt.

**Spoken dates survived as nonsense quantities.** Managed-care speakers say effective dates
as bare digit runs, and transcription renders them without separators. The note reproduced
**"71 states"** — there are 50 — and left **"101 implementation"** uncorrected although the
questioner said "October 1st" aloud in the same exchange. The user supplied the domain
reading: `11` = 1/1, `71` = 7/1, `101` = 10/1, and `"71 states"` means states whose rate
cycle begins 7/1. `TRANSCRIPT_DATE_RULE` adds this as a named correction class in STEP 1,
including the year-suffixed form (`"11 27"` = 1/1/27) and a prohibition on rendering a bare
digit run as a count when the sentence is about timing.

**The Transcript Corrections Log became a glossary.** It listed eighteen entries shaped like
`"ICHRA" → ICHRA … Transcript rendered correctly`, including NDR and RADV, which do not
appear in the transcript at all. The section now logs only terms whose wording actually
changed, and names the identity entry and the acronym definition as the defects they are.

**The two tiers disagreed on source type.** The Brief classified the meeting
INVESTOR/PUBLIC while Key Takeaways classified it MGMT 1:1. Each tier classified
independently, and nothing reconciled them. The Brief is generated after Key Takeaways, so
`_classified_source_type()` now extracts the classification already made and the Brief is
told to reuse it verbatim.

**What did work:** the T95 doctrine produced exactly the intended assessment — *"Rating: 4 —
mostly supported, with minor unsupported assertions"* on the defined scale, with drivers
named and unsupported claims separated out, while staying candid ("notable squishiness",
"investors hoping for an early read got shut down"). No score out of ten, no interior-state
inference.

Validation: 629 backend tests, 46 frontend tests, production build, Render revision and
Cloudflare marker verified. **Unproven:** no summary generated under T97. The checks are
whether "71 states" becomes 7/1 states, whether the corrections log returns to real
corrections or "No corrections required", and whether both tiers agree on source type.

### T96 — the SUMMARIES fan-out was silently dead; job state no longer discarded

Commit: `2f80943` — `Stop transcription discarding the fields the caller set on a job`

A real folder-dropped file (`CNC CEO CFO Small Group - 091626.m4a`) transcribed and
summarised correctly but produced **no Summary Lab experiment**. The job reported
`summaryLabState: "not_requested"`.

T82 stored the upload's `origin` on the in-memory job dict. When transcription finishes,
`_run_transcription` **replaced that dict wholesale** with five keys of its own, so `origin`
was gone before the fan-out check ran — and the check correctly concluded that no fan-out
had been requested. The same replacement discarded `autoProcess`, which
`_mirror_transcription_state` reads when persisting the job.

The Mac agent was never at fault: it detected the file, sent `origin=summaries-folder`, and
its process was started after the T82 edit. The loss was entirely server-side.

- `origin` now travels as an argument through `_run_auto_process_audio_path` into
  `_run_auto_process_audio`, where no dict lifecycle can lose it.
- `_run_transcription` merges into the job dict instead of replacing it, on the success path
  and on all three error paths.
- A test walks the AST of `_run_transcription` and fails on any wholesale assignment to
  `_transcription_jobs[job_id]`. It caught a fourth site during this change.

**Lesson:** state set by a caller and read later by a different worker must travel as an
argument or in the database, not on a shared mutable dict that an intermediate stage owns.
The failure was silent because the fan-out check behaved correctly on the input it was given.

**Blast radius:** T82 shipped at 11:02 today and CNC was the first folder-dropped audio
after it, so one file is affected. Generating its Lab experiment now requires starting one
from the saved Summary in Summary Lab, which is a paid run and the user's call.

Validation: 617 backend tests, 46 frontend tests, production build, Render revision and
Cloudflare marker verified. **Unproven:** no folder-dropped audio has run under T96.

### T95 — the assessment stays candid; the thesis contradiction is fixed

Commit: `d64e4db` — `Keep the assessment candid, and fix the thesis contradiction it caused`

T94 over-corrected. Three of its rules removed capability the original prompt had
deliberately, and one of them broke a live feature. The user caught all three.

- **"Don't hedge" had a purpose.** It existed to stop mushy notes, not to license
  invention. The doctrine now requires committing to a view — if an answer was weak, say so
  — and forbids only inventing a fact to support it.
- **BS detection was deliberate.** Calling out non-answers, evasions, redirections,
  rehearsed talking points and self-contradiction is the job of an assessment. Restored and
  required, with the wording quoted as evidence. The line is drawn at **interior state**:
  feelings, anxiety, morale, private belief and motive stay out, because a joke or a hedge
  is not evidence of them. That still prevents the observed failure, which was reading "I
  lose lots of sleep for lots of reasons" as concern about the CVS renewal.
- **Rating credibility is legitimate.** An unanchored "8.5 out of 10" is not: it implies
  precision that does not exist and cannot be compared between notes. The rating is restored
  on a defined 1–5 scale with stated anchors, and must name what drives it.
- **The baseline rule was wrong, not merely strict.** When a ticker has a registered thesis,
  `thesis_addendum` injects it and asks for per-pillar CONFIRMED / WEAKENED / NO MENTION
  verdicts — and T94 placed "no prior thesis is supplied, do not claim thesis confirmation"
  immediately before that block in the same prompt. The rule is now conditional: compare
  against a baseline where one is supplied, invent one where it is not.

**Lesson:** a rule written to prevent an observed failure removed three capabilities that
were not causing it. Before constraining a prompt, check what each instruction was for — the
"UNFILTERED" framing, the disingenuousness question and the credibility rating were all
deliberate, and only the unanchored scale and the interior-state inference were defects.

**Where the user wants this to go.** The stated goal is that over time the model judges what
is new, what contradicts a previous statement or thesis, and what differs from consensus.
The thesis half already exists via `thesis_addendum`. The missing half is prior-meeting
context: `company_history_recall.py`, `company_memory.py` and `thesis_history.py` hold the
material, but no summary prompt is given the last N notes for the ticker. Supplying them
would let the assessment say "this contradicts June" without inventing the comparison. That
is the natural next step for this workstream.

Validation: 614 backend tests, 46 frontend tests, production build, Render revision and
Cloudflare marker verified. **Unproven:** no summary generated under the revised doctrine.

### T94 — convergence decided: improve the original, retire Improved

Commit: `26c8c37` — `Port the Improved pipeline's evidence discipline into the original Summary`

**This settles priority 3.** Direct comparison of the two prompts showed the original already
carries the stronger fidelity apparatus, and Improved carries almost none of it:

| Mechanism | Original | Improved |
| --- | --- | --- |
| Source-type auto-classification with distinct lenses | yes | no |
| Transcript-correction confidence taxonomy + corrections log | yes | no |
| No quantitative tightening ("teens" ≠ "low teens") | yes | no |
| Segment attribution discipline | yes | no |
| Clarifying follow-ups kept as separate Q&A entries | yes | no |
| Quarterly vs all-time scope discipline | yes | no |
| Quote-length rules, never paraphrase inside quotes | yes | no |
| Nine-point silent self-check | yes | no |
| Named-entity hallucination guard pass | yes | no |
| Known Unknowns with deflection phrase and next check | yes | no |

What Improved had was restraint in the **assessment** step. The original's assessment prompt
asked for a "CANDID, UNFILTERED" take, told the model not to hedge, asked whether anyone
"seemed disingenuous", and requested "Rate overall credibility of key claims" — which is
where the invented "8.5/10" and the psychology read of a deflecting joke came from. The
original's *summary* prompt was never the problem.

So the cheaper path to the user's goal was the reverse of the previous eight releases: port
Improved's four epistemic rules into the original rather than rebuild a dozen fidelity
mechanisms inside Improved.

- `RESEARCH_DOCTRINE`: separate statement from judgment and name the support and the limit;
  never infer psychology or motive; never assign a numerical credibility score; claim no
  novelty, consensus difference or thesis confirmation without a supplied baseline. It
  explicitly outranks the instruction to be candid.
- `ASSESSMENT_INSTRUCTION` replaces **five** near-identical assessment prompts that had
  drifted apart across the audio, document, podcast, meeting and manual paths. It stays
  candid and specific but evidences evasion by quoting wording rather than inferring intent.
- Both summary prompts carry the doctrine; their fidelity apparatus is untouched.

**The earlier instruction not to rewrite the original Summary's prompts no longer applies.**
It existed to protect the comparison baseline; the user has now decided the comparison's
outcome and directed this change.

**Improved is deliberately still running.** Retire it only after the user confirms the
improved original on a real source. Retirement means removing the `summary_comparison`
workflow, its UI and its automatic fan-out — not deleting saved notes.

Validation: 613 backend tests including a new doctrine suite, 46 frontend tests, production
build, Render revision and Cloudflare marker verified. **Unproven:** no summary has been
generated under the new doctrine.

### T93 — every Improved section is written from the source

Commit: `e4907ae` — `Write every Improved section from the source, not a summary of it`

T92 fixed the Q&A log by reading it from the source. The user pointed out the obvious
consequence: **every** section should be, because the goal is for Improved to replace the
original Summary. They were right, and the architecture was worse than it looked.

Improved split the source into 24,000-character parts, summarised each into a
topic-organised record, and wrote every section from those records. The source never reached
the sections at all. The original Summary, by contrast, passes the full transcript to each
section prompt — so Improved was structurally **less faithful than the thing it is meant to
replace**.

The map-reduce exists for sources too large for one prompt. Measured against the last 60
real sources, that case does not occur:

| | |
| --- | --- |
| CAH transcript | 37,453 characters |
| Median source | 39,618 |
| Largest of the last 60 | 213,720 (~53k tokens) |
| Sources above 300,000 characters | 0 |
| Model context | ~200k tokens (~800k characters) |

- Below `DIRECT_SOURCE_LIMIT` (400,000 characters) every section is written from the
  complete original source, marked authoritative, with the evidence records alongside as a
  navigation aid that is explicitly not a substitute.
- Above it the pipeline falls back to records and says so in both the prompt and the
  workspace, so a degraded basis is visible rather than silent.
- Record consolidation is skipped when the source is read directly; it only ever existed to
  shrink records for synthesis.
- `state.synthesisBasis` records which path ran and the method panel reports it.

Cost: each section prompt now carries the full source, roughly 9k extra input tokens per
section on a median transcript.

**Lesson:** the map-reduce was carried over from a context budget that no longer binds, and
it silently degraded every section. Before summarising a source for a model, check whether
the source fits — on this workload it always does.

Validation: 598 backend tests, 46 frontend tests, production build, Render revision and
Cloudflare marker verified. **Unproven:** no note generated under `source-first-v7`.

### T92 — the Q&A log is read from the source, not from a summary of it

Commit: `c0da9ef` — `Build the Q&A log from the source, not from a summary of it`

The user compared the Improved Q&A log against the Original's and supplied the full
transcript. The Original captures essentially every exchange; Improved captured 10, labelled
most "(Implied)", and stated the source was management commentary rather than a
question-and-answer transcript. Both claims are false — the questions are verbatim in the
transcript from Speakers 2 and 3.

**This was a data-flow bug, not a prompt problem.** Every section including the Q&A log was
drafted from the evidence records, which are organised by topic and discard exchanges.
Measured on the saved note: the source has roughly 17 question turns, while the 15,265
characters of records retained **4 question marks and zero `Q:` markers**. The section was
asked to reproduce exchanges it was never shown, and — correctly forbidden from inventing
them — reported them as implied.

- The Q&A log is now generated per raw source part, checkpointed like the evidence records
  and joined in order. Parts containing no exchange are omitted.
- The instruction forbids describing a present question as implied or claiming the source
  lacks a Q&A structure.
- `qa_findings()` flags both a collapsed log and one that disclaims questions in a
  question-rich source.

Cost: one extra model call per source part, only for the Q&A section.

**Lesson:** a section whose job is fidelity to the source cannot be built from a summary of
the source. `brief`, `takeaways`, `assessment` and `questions` are syntheses and are
correctly record-derived; `qa` is a record and is not. Any future section should be
classified that way before it is wired up.

Validation: 593 backend tests, 46 frontend tests, production build, Render revision and
Cloudflare marker verified. **Unproven:** no note generated under `qa-from-source-v6`. The
check is whether the Q&A log reaches roughly the Original's exchange count with no "implied"
labels.

### T91 — assessment stops transcribing, and dropped figures are caught

Commit: `7cec6ce` — `Stop the assessment restating the record, and catch dropped figures`

`readable-v4` was generated on the real CAH note. Measured against every predecessor:

| | conservative-v1 | readable-v2 | readable-v3 | readable-v4 |
| --- | --- | --- | --- | --- |
| Quotes per 1k chars | 4.5 | 3.7 | 0.6 | **1.3** |
| Total characters | 15,762 | 21,084 | 13,778 | **31,689** |
| Q&A exchanges | 0 | 9 | 3 | **14** |

v4 is the best note produced so far and resolves the original complaint: 71% fewer quotes
per 1k than the version first flagged, with 50% more content than v2 and the richest Q&A
log of any version. The checks behaved as designed — takeaways (4.4/1k, worst block 6),
assessment (3.5/1k) and questions (2.0/1k) were each repaired once and each *grew*, so the
shrink guard never had to discard a repair.

Two defects remained, both fixed in `readable-v5`:

- The assessment reached 11,233 characters against 6,767 of takeaways and opened with
  "MANAGEMENT STATEMENTS", restating the record instead of assessing it — a consequence of
  T90 removing the cross-section rule with nothing in its place. The instruction now says so
  explicitly and `assessment_findings()` flags a section disproportionate to the takeaways.
  The shrink guard is bypassed for that one repair, since a shorter assessment is its point.
- The 12–14% long-term algorithm was dropped from every section *and* from the management
  record, while 3.5% was asserted as the long-term target. It was caught only by grepping
  for that string. `figure_coverage()` now extracts distinctive figures from the evidence
  records, normalising spacing and dash style, and reports any the note does not carry.

`figure_coverage` **reports, it does not rewrite.** Another automatic edit risks the damage
v3 caused, and a missing figure is not always an error.

Validation: 587 backend tests, 46 frontend tests, production build, Render revision and
Cloudflare marker verified.

**`readable-v5` was then generated on the real CAH note and both defects are fixed:**

| | v1 | v2 | v3 | v4 | v5 |
| --- | --- | --- | --- | --- | --- |
| Quotes per 1k chars | 4.5 | 3.7 | 0.6 | 1.3 | 1.7 |
| Total characters | 15,762 | 21,084 | 13,778 | 31,689 | 26,459 |
| Q&A exchanges | 0 | 9 | 3 | 14 | 10 |
| Assessment ÷ takeaways | 0.67 | 0.53 | 0.39 | 1.66 | 0.68 |
| 12–14% figure present | yes | yes | yes | **no** | yes |

The assessment fell from 11,233 to 5,257 characters and its ratio from 1.66 to 0.68, in line
with v1 and v2. `assessment_findings()` did not fire — the instruction alone was sufficient,
and the check stands as an unused backstop. `figure_coverage` reported 12 distinct figures in
the records with none missing from the note. Against the note originally complained about:
62% fewer quotes per 1k, 68% more content, and a Q&A log where there was none.

Open judgment call for the user, not a defect: v5's Q&A log labels most questions "(Implied)"
and closes with a note that the source is management commentary with moderator prompts rather
than a formal Q&A transcript. That is the anti-fabrication rule working, and it is why v5 has
10 exchanges where v4 asserted 14. Whether the hedging helps or reads as noise is a product
preference.

**Stop single-note tuning here.** Five regenerations against one CAH transcript produced two
regressions caused by generalising from a sample of one (T89 and T90 both document a case).
Further prompt work should wait for priority 6's frozen real-source packs so a change can be
scored across several meetings. The machinery to do that now exists: `quote_findings`,
`qa_findings`, `assessment_findings` and `figure_coverage` are all deterministic and can be
run over any note without a model call.

### T90 — the quota must not be met by deleting evidence

Commit: `055df48` — `Stop the quota from being met by deleting evidence`

`readable-v3` was generated on the real CAH note and measured against its predecessors:

| | conservative-v1 | readable-v2 | readable-v3 |
| --- | --- | --- | --- |
| Quotes per 1k chars | 4.5 | 3.7 | **0.6** |
| Total characters | 15,762 | 21,084 | **13,778** |
| Q&A exchanges | — | 9 | **3** |

The quoting complaint was solved. The note also lost **35% of its content**, and picked up
four defects worse than the one it fixed:

- The Q&A log fell to three exchanges and asserted that no others were identifiable, from a
  source containing dozens of questions. One answer was openly reconstructed from elsewhere.
- Sections began deferring to each other. The executive brief — the first thing read —
  contained "As noted in the takeaways", "per the takeaways", "in the assessment section".
- The FY27 metric regressed from correctly flagged as unstated (v2) to asserted as a 3.5%
  long-term algorithm alongside a contradictory 12–14% figure in the same section, which is
  the Original's error.

**The repair pass was not the cause** — it fired on one section only. Both prompt changes
were. T90 (`readable-v4`) keeps the code enforcement and reverts them:

- The numeric quota is out of the prompt. Stating it made the model aim for zero and reach
  it by dropping evidence. `quote_findings()` still enforces the ceiling, and the rules now
  state that converting a quote never means dropping the fact, number, qualification or
  attribution it carried.
- Sections are no longer shown each other. Each must stand on its own.
- A repair returning a materially shorter section is discarded and the original kept.
- `qa_findings()` flags a Q&A log that collapsed against a question-rich source, and the Q&A
  instruction forbids reconstructing an answer.
- The method panel reports what each check found and which draft was kept.

**Lesson worth carrying:** on this pipeline, a prompt instruction that competes with the
fidelity rules is either ignored or over-complied with by deletion. Both failure modes were
observed on the same note. Constraints of this kind belong in code, with a guard against the
model satisfying them destructively.

Validation: 580 backend tests, 46 frontend tests, production build, Render revision and
Cloudflare marker verified. **Unproven:** no note has been generated under `readable-v4`.
The open question is whether it restores v2's content at v3's quote discipline.

### T89 — the quoting quota is enforced, not requested

Commit: `1b7ad81` — `Enforce the quoting quota instead of asking for it`

The user regenerated the CAH note under `readable-v2` and still saw heavy quotation.
Measured on the two saved notes rather than judged by impression:

| Section | conservative-v1 | readable-v2 |
| --- | --- | --- |
| Key takeaways | 6.3 quotes/1k chars | 4.6 |
| Executive brief | 4.2 | 3.6 |
| Follow-up questions | 3.4 | 1.2 |
| Investment assessment | 3.2 | 1.9 |
| Q&A log | — | 5.5 (new) |
| Overall | 4.5 | 3.7 |

`readable-v2` did work: density fell in every comparable section and broken numbered lists
went 21 to 0. But the note is 34% longer, so absolute quotes rose 71 to 79, and the rule
"at most one short quoted phrase per point" was ignored in **8 of 9 tagged takeaways**, one
carrying nine quoted fragments. The phrasing was ambiguous and a style preference loses to
the fidelity rules above it in the same prompt.

T89 (`readable-v3`) stops asking:

- Quoting is governed per section. The Q&A log and the per-part management records are
  exempt because verbatim is their job. The four analysis sections carry a stated quota.
- `quote_findings()` checks each draft in code and names the breach; a section over quota is
  redrafted once, instructed to keep every fact, number, qualification and attribution while
  converting the least informative quotes to reported speech.
- Sections already written are passed to later ones so caveats are referred to, not restated.
- The method panel reports which sections were redrafted.

Calibrated against the real note: brief (9 quotes, 3.55/1k) and takeaways (4.6/1k, worst
block 9) breach and would be repaired; assessment (1.88/1k) and questions (1.16/1k) already
comply, so no model call is wasted.

Validation: 575 backend tests including the enforcement and repair-pass tests, 46 frontend
tests, production build, Render revision and Cloudflare marker verified. **Unproven:** no
note has yet been generated under `readable-v3`. Whether the repair pass produces readable
prose without losing content is the next real-source check, and it is a paid run.

### T88 — regenerate an older Improved note in the current format

Commit: `15fa32b` — `Let an older Improved note be regenerated in the current format`

T87 added the Q&A log but left no way to obtain it. Existing notes carry their own prompt
version, and the workspace could only report **Not in this version**. The listing never
returned the pipeline version, the Generate button appeared only when no note existed at
all, and `start()` always passed `resumeId`, which resumes the old row under its own
version.

- The comparisons listing now returns `currentVersion`.
- When every saved note predates it, the workspace offers to generate the current version,
  naming both versions, stating that it re-reads the saved transcript and uses research API
  credits, and that existing notes are kept.
- That run omits `resumeId`, so it creates a note in the current version. Retry still
  resumes in place.
- The saved-version picker marks older notes as "older format".

Nothing is regenerated automatically: upgrading an existing summary is a paid run and stays
user-initiated. New summaries get the current format on their own.

Validation: 567 backend tests, 46 frontend tests, production build, Render revision and
Cloudflare marker verified, and the panel confirmed in the browser against the real
`conservative-v1` CAH note. The generate button was not pressed.

### T87 — Improved notes: readable prose, topic tags, and a Q&A log

Commit: `5cbd098` — `Give Improved notes the readability the Original had`

Prompted by a real side-by-side review of a CAH management meeting. Findings from that
comparison, recorded because they bear on priority 3:

- The **Original** asserted "FY27 guidance of 3.5–4% EPS growth exceeds the long-term 3.5%
  target". The transcript never says EPS and never states a 3.5% long-term target; it
  contains Speaker 3's "3.5 to four is above 3.5" and, separately, Speaker 2's "12 to 14%
  long-term growth algorithm". The Original supplied the unit and dropped the conflicting
  12–14% figure. The Improved version flagged the ambiguity and preserved both figures.
- The **Original** also reported "High credibility (8.5/10)" — an invented numerical score
  the Improved rules already forbid — and inferred psychology, reading the CEO's joke "I
  lose lots of sleep for lots of reasons" as evidence of concern about the CVS renewal.
- The **Improved** version was materially less readable: quote-stuffed (eight quoted
  fragments in one CVS paragraph), repeated the same guidance caveat six times across
  sections, rendered broken numbered lists, and carried a far thinner Q&A log.

Neither version is objectively better overall, and no benchmark exists that could settle
it. T87 changes only the Improved pipeline, leaving the original Summary prompts untouched:

- `summary_comparison.py` RULES now make reported speech the default and reserve quotation
  marks for wording that is itself the evidence, at most one short phrase per point. It
  bans numbered lists, and requires each fact to be stated once rather than restated in
  every later section.
- Key takeaways open with a scannable bracketed topic tag, adopted from the Original.
- A new **Q&A log** section reproduces the substantive exchanges in order, preserving
  numbers, hedges, refusals and non-answers, and must say so rather than invent an exchange
  when the source has no real Q&A structure.
- `VERSION` moves to `readable-v2`. `summary_comparisons` is unique on
  (summary_id, source_hash, version), so existing notes are preserved and readable rather
  than mixed with output from a different prompt. Older complete notes now label a missing
  section **Not in this version** instead of a permanent "Waiting…".

Validation: 565 backend tests including a new cross-file contract test that every generated
section is rendered by the workspace, 46 frontend tests, production build, and the section
list verified in the browser against the real `conservative-v1` CAH note.

**No output-quality claim is being made.** The prompt now forbids the specific defects
observed; whether the result reads better on real sources is priority 2/3 work and needs
the user's own comparison on a new run.

### T86 — Summary Lab gives the page to whichever job is in front

Commit: `942407d` — `Let Summary Lab size itself to the task at hand`

The layout was fixed at `330px / 1fr` whatever the state. Measured at 1600px with no
experiment open: the composer had 330px (26%) while the empty "Independent source review"
panel held 962px (74%) and 1460px of height for 120px of placeholder text.

- With no experiment open the page is a single centred 900px column. The composer spans
  it: five intake buttons in one row, source dropdown 850px instead of 280px, and
  Experiment name beside Optional emphasis rather than stacked. The placeholder panel is
  gone; its orientation copy already exists in the page intro.
- While reading an experiment the two-column split returns, because a long note needs the
  width, and the composer collapses to a **New experiment** button. It went from 654px to
  192px, giving the Experiments list 786px.
- Mobile is unchanged in shape: single column, intake two across, fields stacked.

Verified in the browser against the built bundle by measuring the live DOM in all four
states — idle at 1600px and 375px, reading at both — including that the composer expands
on demand and nothing overflows horizontally.

### T84 / T85 — stop control and duplicate Summary Lab runs

Commits: `4065477` — `Let Summary Lab experiments be stopped, and stop duplicating them`,
`3d42157` — `Stop experiments a restart left running`

One real YouTube submission produced **three identical `korean_bilingual` experiments**
over the same 20k-character transcript. Investigated from the live rows, not reproduced.

- Root cause: the pending transcription job lives in `localStorage`, so every tab and
  every reload resumes monitoring it, and each resumed monitor called `startLab` on
  completion. The manual enqueue path had no duplicate guard at all, unlike the automatic
  one, so each POST bought another Opus multi-pass run.
- A Lab run triggered by a completed transcription now carries that job id, unique
  server-side, so every monitor converges on the first experiment. A deliberate Generate
  click sends no job id and still creates its own experiment, which is what
  `docs/summary-lab.md` specifies.
- New `POST /api/summary-lab/<id>/stop` and a **Stop this experiment** button. Cancellation
  is cooperative at the existing checkpoints: `save()` writes `cancelled` instead of
  `running` when a stop is pending and raises, so a stop cannot be papered over between
  stages, and a queued experiment never reaches the model.
- T85: a row left at `running` by a backend restart has no worker to reach a checkpoint,
  so the stop route tries the experiment's advisory lock. A free lock means nothing is
  working on it and the row is cancelled immediately.

Validation: 555 backend tests, 46 frontend tests, production build, Render revision and
Cloudflare marker verified. The duplicate rows were read from production through the
signed-in app; no experiment was started, cancelled or approved during the investigation.

**Known related design issue, not yet changed.** Summary Lab's YouTube and audio intakes
post to the shared `/api/youtube-summarize` and `/api/auto-process-audio` endpoints, which
always write a full original Summary (five sections plus Korean) to `meeting_summaries`
before Summary Lab starts its own experiment. One Lab submission therefore pays for both
an original Summary and a Lab run, and the original lands in the Summary tab rather than
where the user started. That is the current design — Lab compares against that baseline —
but it surprises the user and deserves an explicit decision.

### T83 — Catalyst synthesis becomes a named destination

Commit: `8ea85ab` — `Promote Catalyst notes to a top-level destination`

Catalyst synthesis was reachable only through Automations → Research agents → an
unrouted `Catalysts` pill, under a page heading that read `TradingAgents`. Three names
for one path, and the workspace had no URL at all.

- **Create → Catalyst notes** is now a top-level destination at `#view=catalysts`. It is
  linkable, bookmarkable and restored by the back button; `agentView` was local state, so
  none of that was previously possible.
- Arriving there shows a `Catalyst notes` heading and no TradingAgents framing or sub-tab
  row. The `Catalysts` pill was removed from the Research agents tab, leaving one path in.
- The agents sub-views (`research`, `new`, `batch`, `dashboard`, `history`) are now scoped
  to `activeTab === 'agents'`, so the last-opened agents panel cannot bleed into the
  routed Catalyst destination.
- Catalyst history is fetched on arrival. It used to depend on the pill's click handler,
  which a URL, a sidebar click or an alert action never invoked.
- Saving a synthesis previously ended in a blocking `alert('Saved to Research tab')` that
  named a destination without going there. It now shows an in-page confirmation naming the
  saved document, with a button that opens Library → Research documents.
- Fixed a pre-existing mobile defect on that screen: at 375px the Auto-Pilot `Auto-fire`
  select overflowed the card and overlapped the paragraph. The row now stacks below `sm`.

Validation completed for T83: 46 frontend tests (two new: `catalysts` in `VIEWS`, its
Create-group membership, its label, and `readRoute`/`routeHash` round-trip with ticker),
549 backend tests, production build, and a browser pass against the built bundle at
desktop and 375px. Verified in the browser: the sidebar entry and breadcrumb, the routed
heading, back-button restore, the Research agents tab still intact with no Catalysts pill
and no sub-view leakage, catalyst history loading on arrival, and the Auto-Pilot row
measured as non-overflowing at both widths. No synthesis was run, no proposal approved,
no note saved and no email sent during verification.

### T82 — correct the automatic dual Summary fan-out

Commit: `1dfaf72` — `Scope SUMMARIES fan-out and harden Summary Lab recovery`

T81 fanned every `/api/auto-process-audio` job into Summary Lab. That endpoint serves
two callers, so three real defects shipped with it. T82 fixes all three.

- **Duplicate paid Summary Lab runs.** Summary Lab's own Audio intake posts to the same
  endpoint and then starts its own experiment from the saved Summary. Since T81 each
  such upload produced two experiments over the identical transcript — two complete
  Opus multi-pass source reviews. Audio uploads now carry an `origin` marker, and only
  `origin=summaries-folder` (sent by `charlie_local_agent.py`) fans out. Unidentified
  callers do not fan out, so every failure mode is a missing experiment rather than
  silent duplicate model spend.
- **Summary Lab now adopts an automatic experiment instead of starting a second one.**
  If a completed transcription already carries `summaryLabId`, the workspace opens that
  experiment and says so. This holds even if a stale frontend bundle or an un-restarted
  agent disagrees about the origin marker.
- **Restart durability regression.** T81 moved `_mirror_transcription_state` after the
  two follow-on queue calls. A backend restart in that window left the mirror at
  `summarizing`, so the folder watcher kept the file in the `SUMMARIES` root and the
  next tick re-uploaded and re-transcribed audio that was already saved. Completion is
  now mirrored before the queues run, and the Summary Lab id is mirrored after.
- **False "Summary Lab did not start" alerts.** The watcher polls every 10 seconds and
  decided Lab state from the same response that first reported completion, before the
  fan-out had returned an id. The status route now reports an explicit
  `summaryLabState` (`pending`, `started`, `failed`, `not_requested`, `unknown`), the
  watcher waits out a bounded 60-second `pending` window, and it stays silent about Lab
  for uploads that never requested a fan-out. The database fallback never reports
  `pending`, because after a restart only a recorded Lab id proves the fan-out started.
- **Summary Lab recovery thread pile-up.** Only two experiments run at once, so a
  recovery thread can block on the semaphore for a long time. The 30-second sweep
  queued another blocked thread for the same experiment on every pass. Recovery now
  tracks in-flight experiment ids and starts at most one thread each.

Validation completed for T82: 549 backend unittests, 45 frontend tests, the research
quality self-test, `py_compile` on `app_v3.py`, `summary_lab.py` and
`charlie_local_agent.py`, and the production build. Render `/health` reported revision
`1dfaf726498be5ddde89bac233d7ea712f2fcbe2`, the Cloudflare worker reported
`2026-09-20T82` and served `service-worker.js` at `20260920-82`, and
`com.charlie.local-agent` restarted to `state = running` with a clean startup log. The
two new structural regression guards were confirmed to fail against the T81 code and to
pass now. No paid audio run and no model-backed workflow was launched for validation, so
the deployed fan-out behavior itself is not yet proven against a real recording.

Not changed in T82: automatic folder fan-out still uses Summary Lab's English mode, and
historical files under `SUMMARIES/Processed` are still not mass-reprocessed.

### T81 — automatic dual Summary processing

Commit: `d8c608d` — `Fan out SUMMARIES audio into Summary Lab`

New audio placed at the root of the iCloud `SUMMARIES` folder now follows the established original Summary workflow and also launches an independent Summary Lab experiment from the complete saved transcript.

- The original Summary remains the primary saved output.
- The older Improved comparison continues unchanged.
- Summary Lab receives a separate durable experiment using its own current prompt and settings.
- Automatic Summary Lab rows display `Auto from SUMMARIES`.
- Automatic experiments are idempotent by saved Summary, source hash, prompt version, and output mode.
- Interrupted automatic Lab jobs resume after a backend restart with bounded recovery.
- Lab completion/failure produces a Charlie alert.
- One branch can fail without preventing the other saved output.
- The transcription job records the linked Summary Lab ID, and the Mac completion notification states whether Lab started.
- Automatic folder jobs currently use Summary Lab's English mode.
- Historical files already under `SUMMARIES/Processed` were intentionally not mass-reprocessed because that would create large unrequested model usage.
- At deployment time the `SUMMARIES` root contained no waiting audio, so the first real new file remains the live end-to-end proof.
- T82 corrected this release's fan-out scope, restart durability, watcher notification and recovery sweep. Read the T82 section above first.

Validation completed for T81: 43 frontend tests, 18 focused backend tests, Python compilation, production build, Render health revision, Cloudflare T81 asset, and launch-agent restart. No paid audio run was launched solely for validation.

### T80 — Summary Lab language and email polish

Commit: `14784c4` — `Add Korean Summary Lab modes and clean email styling`

- Summary Lab YouTube intake mirrors original Summary language choices: English, English plus Korean interpretation, and Korean-only output where appropriate.
- Summary and Summary Lab email rendering was consolidated into a clean professional format.
- Removed the teal/blue title banner and `Generated by TDL Equity Analyzer` footer.
- Email content uses black Calibri 11-point styling with bold section headings.
- Existing Summary content prompts were not intentionally changed by this formatting work.

### Recent Summary Lab usability work

- `516be90`: fixed accordion state so sections expand/collapse reliably.
- `360d0b4`: added document, audio, and YouTube intake; converted raw Markdown-looking output into sanitized, readable HTML; improved section controls and mobile readability.
- `212208f`: added professional document formatting, individual section sharing, and previewed email-all-sections behavior.
- `8ea4e4c`: introduced Summary Lab as an isolated parallel experiment using connected iCloud documents, checkpointed long-source review, and five familiar sections: Executive Brief, Key Takeaways, Meeting Summary, Follow-up Questions, and Overall Assessment.

Summary Lab deliberately remains separate from the original Summary workflow so the user can compare output quality before deciding whether to replace anything.

### T77 — catalyst synthesis upgrade

Commit: `7b67700` — `Ship catalyst investor notes, event synthesis and Q&A sharing in T77`

Catalyst synthesis now distinguishes source shapes:

- Transcript/fireside-chat workflow: concise brief, detailed note, Q&A when a real question-and-answer structure exists, and shareable output.
- The Q&A view can hide/show asker and answerer identities; email/save follows the visible choice.
- Multi-document event workflow: reviews the folder's primary and broker materials, then writes one investor-facing event note in the user's voice. It does not narrate which broker said what and does not add BUY/HOLD/SELL labels.
- The event note centers on: My takeaway, What happened, Why it matters, What remains unproven, and What I'm watching next.
- Generated review assets and local comparison tools exist, but some remain untracked local development material; preserve them unless the user decides to formalize or remove them.

### Other important recent capabilities

- Editable earnings recap decks and guided research starters (`73382a3`).
- Versioned investment-case signals, evidence half-life doctrine, falsification/variant/position-divergence views (`4a7805d`).
- Evidence proposal repair with source re-reading, retained checkpoints, concise proposals, and acceptance gates.
- Command Charlie favorites and guided multi-company meeting preparation.
- Managed AlphaSense refresh policies, source preferences by stock/subsector/analyst, explicit restricted-source handling, durable collection ledgers, and bounded recovery.
- UI release audit across primary routes, themes, and responsive widths, followed by readability and pool-capacity fixes. See `docs/ui-release-audit.md` for proof boundaries.

## Current workflows and boundaries

### Original Summary

The established Summary workflow is still active and remains the comparison baseline. It supports saved documents/audio/YouTube, Brief, Key Takeaways, Meeting Summary, Follow-up Questions, Assessment, transcript access, saving/export, and email. Earlier character clipping was removed. Its prompts were deliberately changed in T94: see that entry. The earlier instruction not to touch them has been superseded by the user's convergence decision.

### Summary Lab

Summary Lab performs a more rigorous, checkpointed full-source review and produces the same recognizable backbone in a separate workspace. It supports saved Summary sources, connected documents, audio, YouTube, output-language modes, collapsible HTML sections, copy/save/email controls, and automatic dual routing from the `SUMMARIES` folder.

Automatic fan-out is scoped to the folder watcher only. Audio uploaded inside Summary Lab
starts exactly one experiment, using the title, emphasis and language the user chose. If a
recording was already fanned out automatically, the workspace opens that experiment rather
than paying for a second review of the same transcript.

Open question: after several real comparisons, decide which Lab prompt/format improvements should migrate into original Summary. Preserve the original until the user explicitly makes that decision.

### Catalyst synthesis

The upgraded code is deployed. The user has reviewed MMM and ELV transcript examples plus a multi-document MRK event example. Further work should focus on output quality, document-shape classification, analyst voice, and repeatable live comparisons rather than merging it with the unrelated Summary workflow.

### Investment thesis lifecycle

Charlie can store cases, compare new evidence, propose source-backed revisions, repair unsupported proposals, discuss interpretations, show case signals/evolution, and retain guarded revision history. The investor remains the approval authority. Evidence checks reduce risk but do not independently certify management claims or complete factual accuracy.

### AlphaSense automation

The scheduled collection worker is active, but a saved policy or queued request is not proof of completed downloads. The last repeated scheduled checks returned no due managed request. Full unattended coverage across all tickers has not been proven end to end.

Collection requires:

- This Mac to be awake and connected.
- A signed-in Chrome AlphaSense session.
- The scheduled browser worker.
- Explicit verification of company, date range, source type, usage restriction, downloaded originals, hashes, destination, and handoff.

Do not invent observed URLs, counts, downloads, or completion. Never pass provider-restricted originals into model workflows.

## Known proof gaps and next priorities

1. **Run a real dual-summary audio comparison.** Add one new representative audio file to the root `SUMMARIES` folder and confirm that original Summary and `Auto from SUMMARIES` Lab outputs both complete, are readable, and can be emailed/saved. This incurs real model usage and should be user-driven, not launched merely for QA. T82 changed the code paths this exercises, so it is still the live end-to-end proof: confirm exactly one Lab experiment per recording, that the Telegram message reports the correct Lab state, and that the file moves to `SUMMARIES/Processed` once.
2. **Evaluate Summary Lab quality across several source types.** Compare earnings calls, investor meetings, noisy audio, long YouTube transcripts, and non-earnings documents. Capture which sections are materially better or worse than original Summary.
3. **Confirm the improved original, then retire Improved.** Decided in T94: the original Summary keeps its fidelity apparatus and has gained Improved's evidence discipline. Generate a summary on a real source, confirm the assessment no longer scores credibility or infers psychology and that nothing else regressed, then remove the `summary_comparison` workflow, its UI and its automatic fan-out. Saved notes stay. Superseded note: The user's stated goal is that Improved eventually replaces the original Summary, so Improved must be written from the original source in every section — as of T93 it is.  After real testing, selectively promote proven Lab prompt/format improvements into original Summary or retain both permanently. T87–T89 moved the Improved pipeline toward the Original's strengths (topic tags, a Q&A log, enforced quoting quotas) while leaving the original Summary prompts untouched. Two defects found in the Original during that work are still unfixed and argue against promoting it as-is: it reported an invented "8.5/10" credibility score, and it asserted an EPS unit for FY27 guidance that the transcript does not support while dropping the 12–14% figure that contradicts it.
4. **Validate catalyst synthesis on more real folders.** Include single transcript, transcript plus presentation, and multi-broker event folders; score concision, factual attribution, analyst voice, unresolved issues, and PM usefulness.
5. **Prove a complete managed AlphaSense assignment.** Demonstrate browser discovery, source restrictions, original download, iCloud handoff, recap, thesis proposal, and recovery for a real user-selected ticker without overstating unattended coverage.
6. **Improve real-source quality benchmarks.** Current automated checks are useful regressions, not expert certification. Add frozen real-source packs and investor-scored outputs without committing licensed source bodies.
7. **Continue UI simplification.** Navigation and complex evidence workflows have improved but remain dense. Any redesign must be inspected at desktop and mobile widths with real long content. T83 did this for Catalyst synthesis. The same pattern is worth auditing elsewhere: sub-views held in unrouted local state have no URL, no back-button behavior and no way for an alert or sidebar entry to link into them. `agentView`'s remaining panels and the Research agents heading are the obvious next candidates.
8. **Broaden recovery cautiously.** Long-running Summary Lab and several research jobs have bounded recovery; audit remaining model-backed jobs for durable identity, checkpointing, ownership fencing, duplicate prevention, and visible failure. T82 fixed the Summary Lab recovery sweep and the audio completion mirror. Two known gaps remain and are unproven in production: a restart between the saved Summary and the fan-out leaves no Lab experiment to recover, because recovery only resumes rows that already exist; and `recover_once` reads `ANTHROPIC_API_KEY` from the environment, so recovery is silently inactive if only a Settings-supplied key is present.

## Safe commands

Never run `pytest`.

```sh
.venv/bin/python -m unittest discover -s tests/unit
npm run test:frontend
.venv/bin/python scripts/evaluate-research-quality.py --self-test
npm run build
```

For focused Summary Lab work, the most recent safe subset was:

```sh
.venv/bin/python -m unittest \
  tests.unit.test_summary_lab \
  tests.unit.test_summary_lab_recovery \
  tests.unit.test_summary_lab_enqueue \
  tests.unit.test_summary_lab_email \
  tests.unit.test_summary_lab_fanout \
  tests.unit.test_summary_bulk \
  tests.unit.test_summary_job_status \
  tests.unit.test_research_email_format
```

Use `py_compile` for every touched Python module. Do not start paid research, send emails, or approve research to prove a code path.

## Deployment and operational checks

Current release markers must stay synchronized:

- `worker.js`: `2026-09-20T97`
- `service-worker.js`: `20260920-97`
- `src/app.jsx`: `2026-09-20T97`

After an application change:

```sh
npm run build
git push origin main
npx wrangler deploy
curl -fsS https://equity-analyzer-backend.onrender.com/health
```

If the local agent changes:

```sh
launchctl kickstart -k gui/$(id -u)/com.charlie.local-agent
launchctl print gui/$(id -u)/com.charlie.local-agent
tail -80 /tmp/charlie-agent.stderr.log
```

Render may return transient 502 responses while rolling forward. Wait for `/health` to report the exact new commit before treating deployment as complete.

## Repository state warning

At this handoff, `main` is committed through `0cb4230`, but the checkout contains unrelated local/runtime state. Preserve it. In particular, do not blanket-stage or delete:

- `.claude/settings.local.json`
- `.omc/**`
- `nohup.out`
- `.DS_Store`
- untracked generated files under `build/`
- `charlie_investment_research_handoff/`
- local catalyst review scripts/assets unless the assigned task explicitly owns them

Always inspect `git status --short`, stage an explicit allowlist, and review `git diff --cached` before committing.

## Suggested first Claude Code instruction

> Continue Charlie from production commit `0cb4230` and release T97. Read `AGENTS.md`, `CLAUDE.md`, and `docs/AI_HANDOFF.md` before acting. Preserve every unrelated dirty or untracked file; do not reset, clean, stash, or broadly stage the repository. First audit the latest dual Summary/Summary Lab implementation and report any correctness gaps without launching paid processing. Then continue the highest-priority assigned item, run only the documented safe tests, commit only intended files, update `docs/AI_HANDOFF.md`, and deploy only when the change is complete and verified.

## Relevant deeper documentation

- `docs/summary-lab.md`
- `docs/catalyst-workflow.md`
- `docs/catalyst-release-T77.md`
- `docs/alphasense-refresh-worker.md`
- `docs/alphasense-collector.md`
- `docs/charlie-roadmap.md`
- `docs/charlie-command-build.md`
- `docs/evidence-workspace.md`
- `docs/earnings-workspace.md`
- `docs/ui-release-audit.md`

Some roadmap/status documents describe earlier releases. Treat this handoff and Git history as the current production baseline, then use older documents for design rationale and proof boundaries.
