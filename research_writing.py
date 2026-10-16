"""Shared analytical writing contract; evidence review remains a separate step."""
POLICY = 'synthesis-v1'
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


def synthesis_rules():
    return GUIDANCE + '''
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
