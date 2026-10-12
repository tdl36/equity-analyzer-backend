# Saved-scenario arithmetic contract

The investment-map renderer checks already-supported scenario claims without
mutating the report, replacing its values, rebasing prices or approving a target.
The check is deliberately narrow: positive per-share diluted EPS multiplied by
a positive P/E, and price return excluding dividends. Cash-flow, enterprise-value,
range and narrative methods remain unverified. Source matching is not a guarantee
of source accuracy or appropriate assumptions.

All four supported string fields must match these explicit forms, with consistent
currency, fiscal year and accounting basis:

- `eps_or_fcf`: `FY2027 adjusted diluted EPS USD 6.00/share`
- `valuation_multiple`: `18.0x FY2027 adjusted diluted EPS; rationale: peer recovery`
- `implied_value`: `USD 108.00/share`
- `upside_downside`: `+8.43% price return vs USD 99.60/share on 2026-10-09`

`GAAP` is also accepted. Dates must be real calendar dates; supported currencies
are explicitly listed in scenario_arithmetic.py. Numbers allow up to 12 integer
and four decimal digits. Rounding uses half-up at each saved output's precision.
Price return is checked against the unrounded EPS-times-multiple result, not an
independently rounded target. No current market snapshot enters the calculation.

Results: `checked` means these two arithmetic comparisons agree, `mismatch`
requires review, and `unverified` means inputs are missing or unsupported. Original
values and citations stay visible in all cases. A missing multiple rationale,
unsupported attribution, malformed period, range or appended narrative cannot
silently become a verified input. This does not validate rationale quality,
currency authenticity, fiscal-calendar mapping or investment materiality.

Existing report schema, research prompts, active jobs and source packs are unchanged.
This is a strict read-only check, not a new assumption editor or automatic repair.
Do not rewrite research just to satisfy this formatting contract. A broader typed
valuation workflow, historical multiples and financial reconciliation remain open.
