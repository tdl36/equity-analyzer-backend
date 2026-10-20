# Research quality benchmark — October 19, 2026

## Observed failure, not acceptance

The investor rejected the October 17 ABT regeneration as weak. It described product
opportunities without establishing differentiated expectations, an earnings/cash
bridge or a justified valuation judgment. Comparing base upside with bear downside
alone did not establish expected return. The call returned 67 fields but mistyped
one identifier; an assistant reconciled the response and saved a pending package.
No investor acceptance occurred. The earlier complete package had 85 editable
fields, exceeding the 80-field rewrite bound.

## Pipeline comparison

Original Research Pipeline (`app_v3.py`, `/analyze` document assembly) submits PDF
bytes as document blocks and asks for a cohesive thesis, pillars and signposts.
Stock Analysis extracts text and historically fits all sources into 160,000
characters using opening/closing excerpts. Its legacy prompts also constrained
claims to 180 characters and one passage. T387 removed those writing restrictions
for new synthesis requests, but existing ABT rewrite sources remained clipped.
ABT's frozen coverage records total 319,792 extracted characters; four of eight
sources lost middle text. These are demonstrated differences, not proof that PDF
input alone guarantees analytical quality or that the original workflow is correct.

## T389 implementation

New synthesis-v2 research preserves complete returned extraction up to 480,000 UTF-8
bytes; over-budget sources/packs stop before model dispatch. Individual-source
extraction limits still apply. This is text fidelity, not PDF/chart certification.
Legacy requests keep their frozen text and policy. New v2 package revisions re-extract
only the same hash-verified originals during preparation/start, reprice and fingerprint
the evidence, and freeze it on the new job. Changed bytes, restrictions or extraction
change prevent authorization from silently applying to different evidence.

Whole-package revisions support 120 fields within the unchanged aggregate character
bound. Model-facing IDs are deterministic short aliases mapped back through the frozen
specification; unknown, missing and repeated IDs still fail. No fuzzy response repair
or additional paid retry was introduced. Writing v2 requires a material earnings/cash
bridge, supported expectations, scenario rationale and an explicit qualified judgment;
it prohibits deriving growth from a P/E alone or expected return from base/bear ranges.
These instructions are requirements to evaluate, not a quality guarantee.

## Acceptance rubric and next real proof

Score each dimension 0 (absent/incorrect), 1 (partial), 2 (supported and useful):
- Factual fidelity: material numbers, dates and claims trace to inspected originals.
- Business economics: explain customer value, recurring economics and profit drivers.
- Underwriting: rank drivers, quantify supported contribution/offsets, expose gaps.
- Expectations: distinguish evidence of market expectations from a single forecast.
- Valuation: justify earnings/multiples, periods, horizon and return calculations.
- Judgment: explain attractiveness or insufficient conviction and strongest countercase.
- Monitoring: identify specific evidence that would change the investment judgment.
- Coherence: brief, summary and thesis express the same assumptions and conclusion.

An invented material fact, unsupported consensus claim, or inconsistent valuation
fails regardless of total score. Target >=13/16 with no zero; investor review remains
separate. Compare anonymized outputs from identical originals and instructions to
avoid mistaking input differences for model quality. Preserve outputs and scores.
Next: prepare a real authorized ABT revision with the restored text, inspect evidence
and rendered output, and compare against this rubric. Do not mark the stock journey
or any of the eight priorities complete from code/tests or a delivered draft alone.
