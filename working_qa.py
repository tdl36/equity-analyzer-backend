"""Additive Q&A working-note contract; examples specify style, never facts."""
import hashlib
import uuid

INSTRUCTION = '''Write Q&A · Working Notes: compact human working notes, not a transcript or an executive summary.
Keep the original substantive question sequence and explicit Q: / A: structure. Shorten each
question to its core ask. Under A:, use concise bullets, sentence fragments, familiar finance
abbreviations (yr, mgmt, LT, EPS, FY, vs., esp., w/) and occasional + or arrows ONLY for a
relationship supported by the source. Natural and legible; no invented typos or forced slang.
Compress syntax, not evidence. Preserve substantive answers, examples, numbers, units, periods,
conditions, negations, caveats, follow-ups and non-answers. Keep different speakers separate
with a short role label only when supported. Never invent a question to fit an answer or turn
prepared remarks into a fabricated exchange. If no actual Q&A is present, say so briefly.
Paraphrase ordinary language, retaining the force of hedges such as could, target and not yet.
Use very few exact quotations. Avoid repetitive 'management said', long formal sentences,
em-dash connectors, polished concluding paragraphs and added investment interpretations.
Retain material business uncertainty naturally; keep recording/source issues in the private
review appendix. Omit unresolved numerical specifics, never silently guess a unit or date.
Exclude information explicitly marked not for publication. No transcript/process references.
Do not claim these are handwritten or that you attended. This is an additional presentation
of the evidence, not a replacement for the complete notes.
Return Markdown: **Q: shortened question?**, then **A:**, a blank line and bullets. No tables or HTML.
Style-only example (not evidence):
**Q: Margin recovery next yr?**
**A:**
- Pricing helps; cost trend still a headwind
- Target unchanged, timing less clear
Do not copy the example's facts into the output.
'''

def identity(kind, parent_id, source):
    digest = hashlib.sha256(source.encode()).hexdigest()
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f'charlie:working-qa-v1:{kind}:{parent_id}:{digest}')), digest
