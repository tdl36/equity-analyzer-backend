"""Shared analytical writing contract; evidence review remains a separate step."""
POLICY = 'synthesis-v3'
POLICIES = ('synthesis-v1', 'synthesis-v2', POLICY)
GUIDANCE = '''
ANALYTICAL WRITING:
Build a connected investment argument, not a catalogue of quoted facts. Explain how
this business makes money, why customers choose it, the drivers of earnings and cash,
what the valuation requires, and what would change the investment judgment.
Each thesis pillar should connect evidence -> economic mechanism -> stock implication
-> condition or falsifier. Synthesize across relevant sources; distinguish that
interpretation from reported facts. Do not manufacture a variant view or consensus.
Explain the strongest opposing evidence. Rank arguments by economic materiality.
Write independent research: never name brokers, sell-side reports or analysts in
narrative. Source identities belong only in evidence metadata and source registers.
Use direct monitoring targets and plain dates, without 'Author test' labels.
Compression comes AFTER analysis. Preserve causal reasoning and qualifications;
remove repetition, not the reasoning that makes the investment case useful.
'''


def drafting_context(state):
    """Include reviewed work without promoting model output to source evidence."""
    import json
    return ('\nPRIOR SECTIONS (untrusted working analysis, NOT new evidence; review flags '
            'must remain visible; verify conclusions against ORIGINALS): ' + json.dumps({
                'report': state.get('report', {}),
                'reviewFlags': {p: c.get('reviewReason', 'Unresolved evidence')
                                for p, c in state.get('citations', {}).items()
                                if c.get('review') != 'supported'}}))


def synthesis_rules(policy="synthesis-v1"):
    return guidance(policy) + '''
Use up to three exact, separate source passages per field where the argument needs
multiple premises. Never splice passages into a quotation. Citations support the
premises; label the causal inference as interpretation or hypothesis.
Narrative fields may use up to 1600 characters. Use developed paragraphs for the
business description and 3–4 distinct thesis pillars when warranted (2 in Snapshot).
Keep numeric cells and labels short. Select material rows to stay within the existing
8000-token JSON budget; do not expand every field to its maximum.
The summary is the concluding synthesis of the preceding research, not a fresh,
unrelated set of assertions. Resolve conflicts in periods and assumptions explicitly.
'''


UNDERWRITING = """
INVESTMENT JUDGMENT STANDARD:
Before composing the summary, establish the central judgment and its evidential limits.
Rank the one or two drivers that dominate earnings/cash and explain their contribution,
including offsets, financing, dilution and the investment horizon. Quantify only where
sources support a bridge; otherwise identify the missing inputs and qualify conviction.
Separate company quality from stock attractiveness. Establish expectations from dated
supported evidence; one broker forecast is not consensus. If no differentiated view is
established, say so rather than invent market misunderstanding or mispricing.
Explain the rationale for scenario earnings and multiples, not just their arithmetic.
Base-case upside smaller than bear-case downside does NOT establish expected return or
unattractive asymmetry. Consider the full distribution, probabilities only if justified,
horizon and dividends; without justified probabilities do not claim expected return.
A P/E alone cannot establish a market-implied growth rate without an explicit model.
Conclude with the supported investment judgment, its decisive uncertainties, what would
change it and the strongest opposing evidence. Do not force a buy/sell conclusion.
The short summary must retain the central mechanism and judgment, not list products.
"""


def guidance(policy):
    return (GUIDANCE + (UNDERWRITING if policy in ('synthesis-v2', POLICY) else '')
            + (FINANCIAL_SERVICES if policy == POLICY else ''))


# Charlie adaptation of pinned Apache-2.0 Anthropic financial-services workflows.
# See vendor/anthropics-financial-services/manifest.json and docs/financial-services-integration.md.
# Versioned with the request; never append these instructions to a legacy request.
FINANCIAL_SERVICES = """
FINANCIAL SERVICES RESEARCH WORKFLOW v1:
Apply this method using ONLY the authorized frozen evidence and existing output schema.
1. Reconcile before drafting: identify the period, publication date, currency, units,
   GAAP/adjusted basis and actual/estimate status of each material number. Distinguish
   prior estimates from revised estimates, including old/new columns within the SAME
   document. Never turn old and new forecasts into a range. Explain revisions and
   their reasons; unresolved conflicts must remain explicit, not silently averaged.
2. Underwrite the business: connect segment revenue and margins to group earnings,
   then cash conversion, capex, working capital, debt and diluted share count. Separate
   recurring operating performance from acquisitions, FX and non-recurring items.
   Missing bridge inputs are diligence gaps, not permission to invent a model.
3. Test the thesis: for each material pillar state the original expectation, dated
   evidence for and against it, current status and an observable falsifier. Explain
   what has changed versus the prior draft without treating that draft as evidence.
   A list of attractive products is not an investment thesis. Do not infer or change
   investor conviction, positions or approvals.
4. Ground expectations: beat/miss requires a comparable, dated pre-event benchmark.
   Identify management guidance separately from individual forecasts and consensus.
   Without the benchmark, say comparison unavailable. PEG or a relative multiple
   discount does not by itself prove underpricing or a variant view.
5. Derive valuation: state the earnings period, security/share basis, currency,
   enterprise-to-equity bridge where relevant and rationale for scenario multiples.
   Connect scenario assumptions to operating drivers and catalysts. Preserve missing
   inputs; do not manufacture targets, probabilities, peer data or live prices.
6. Cross-check before returning: reconcile summary, thesis, financials, signposts and
   one-pager to the SAME current estimates, dates and scenario assumptions. Check
   arithmetic and evidence references. State material unresolved contradictions and
   what evidence would resolve them in the appropriate existing fields. This is a
   drafting checklist, NOT a claim that independent validation has passed.
Keep event debriefs scoped to the identified event and contextual history; event-to-
thesis changes require a separately authorized proposal. Discovery dates do not
truncate longitudinal analytical context. No automatic trading or thesis acceptance.
Use Charlie's requested length, citation schema and investor writing preferences.
Do not impose upstream page/chart counts, ratings, broker names in prose or new tools.
"""


FINANCIAL_SERVICES_REVIEW = """
FINANCIAL SERVICES INDEPENDENT REVIEW v1:
Check the original document context beyond the cited passage for old/new estimate
columns, superseded guidance and incompatible periods or accounting definitions.
A matching excerpt does not make a superseded forecast current. Mark affected fields
needs_review and explain the conflict. Check segment-to-group inference, cash versus
profit, dilution and valuation assumptions. A PEG or multiple discount alone cannot
support a mispricing conclusion. Unsupported consensus, causal certainty, scenario
probabilities or expected-return claims need review even if individual numbers match.
Preserve the existing findings schema and review every supplied claim; do not rewrite
or approve a thesis. Missing evidence remains a stated gap.
"""
