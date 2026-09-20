"""Research doctrine shared by every note pipeline.

It lived in app_v3.py, which summary_lab.py cannot import without a cycle, so
Summary Lab carried its own older copy and the two drifted: the Lab still
forbade the credibility scale the Original was told to use, and never received
the date shorthand rule. One definition here means one behaviour everywhere.
"""

DATE_SHORTHAND = """Speakers say effective dates as bare digit runs. In managed care these are month/day rate-cycle or effective dates, not quantities:
- "11" = 1/1 (January 1), "41" = 4/1, "71" = 7/1, "91" = 9/1, "101" = 10/1 (October 1)
- A year may follow: "11 27" = 1/1/27, "101 26" = 10/1/26
- "71 states" means states whose rate cycle begins 7/1, not seventy-one states. "101 implementation" means an October 1 implementation. "the 101 population" is the population affected on 10/1."""

DATE_SHORTHAND_WARNING = """Never reproduce a bare digit run as a count or a percentage when the sentence is about timing, effective dates, rate cycles or affected populations — there are only 50 states, and a rate cycle is a date."""

_BUCKET_CLASSIFICATION = """Bucket A when timing, a quarter, a rate cycle, a tax cycle, an effective date, or another speaker saying the date aloud makes the reading unambiguous; Bucket B otherwise."""

# The Original composes its corrections log in Bucket A / Bucket B terms.
TRANSCRIPT_DATE_RULE = (
    "(d) DATE AND RATE-CYCLE SHORTHAND — a distinct correction class.\n"
    + DATE_SHORTHAND + "\n"
    + _BUCKET_CLASSIFICATION + " " + DATE_SHORTHAND_WARNING
)

# Summary Lab has no bucket framework, so it takes the substance alone.
LAB_DATE_RULE = DATE_SHORTHAND + "\n" + DATE_SHORTHAND_WARNING

RESEARCH_DOCTRINE = """EVIDENCE DISCIPLINE — how to be candid without inventing.
- Commit to a view. If an answer was weak, say it was weak; if someone was impressive, say so. Your own vagueness is not caution, it is a wasted note. What you may not do is invent a fact to support the view.
- Judge the answer, not the interior state. Non-answers, evasions, redirections, rehearsed talking points, internal contradictions and inconsistencies with what the same speaker said earlier are exactly what this section is for — call them out and quote the wording that shows it. What you may not do is claim knowledge of anyone's feelings, anxiety, morale, private belief or motive: a joke, a hedge or a disfluency is not evidence of an interior state.
- Separate what was said from what you conclude. Attribute claims to their speaker; a statement is not verification. Label your own judgment as judgment, name what it rests on, and state its limit.
- Rate the credibility of the key claims on this scale, and name the two or three things driving the rating:
  5 — specific, internally consistent, and backed by figures or mechanisms given in the source
  4 — mostly supported, with minor unsupported assertions
  3 — mixed; material claims rest on assertion alone
  2 — largely assertion, with evasion or inconsistency on material points
  1 — internally contradictory, or contradicted elsewhere in the source
  Use this scale as given. Do not invent decimals, a ten-point range or a different scale: "8.5 out of 10" implies a precision that does not exist and cannot be compared across notes.
- Compare only against a baseline you were actually given. Where a thesis, prior statement or estimate appears in this prompt, compare against it explicitly and say which. Where none is supplied, do not assert novelty, a beat or a miss, a consensus difference or thesis confirmation — you would be inventing the comparison rather than making it."""
