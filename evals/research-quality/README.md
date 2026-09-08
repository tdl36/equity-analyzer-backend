# Research quality regression packs

Run `.venv/bin/python scripts/evaluate-research-quality.py --suite` for all controls. Run with `--pack FILE --candidate SAVED_RECAP.json` to inspect a saved output without new model charges.

The synthetic earnings fixture and two historical FDA excerpt fixtures test selected source matching, period/endpoint terminology, missing information and prohibited overclaims. Real-source excerpts have source URLs, retrieval dates and immutable excerpt hashes. The FDA fixtures cover [Rinvoq induction trial design](https://www.fda.gov/drugs/news-events-human-drugs/fda-approves-first-oral-treatment-moderately-severely-active-crohns-disease) and the [Leqembi primary clinical endpoint](https://www.fda.gov/drugs/drug-trials-snapshots/drug-trials-snapshots-leqembi).

Positive and negative outputs are authored control fixtures, not generated production research. Their `reviewPassed` fields represent fixture labels; they are not evidence an independent reviewer model ran. These tests are not expert graded, comprehensive, or current prescribing guidance. Regex presence does not establish correct reasoning, factual coverage or investment judgment. Quote matching is limited to the selected excerpt. Broader earnings/valuation/clinical packs, adverse-source cases, blind analyst scoring and actual model comparisons remain necessary.
