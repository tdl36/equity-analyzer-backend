# Company-specific investment-map validation

October 11, 2026. Renderer v5 extends the existing layout to Deere and Stryker.

## Public business context

- [Stryker FY2025 10-K](https://www.sec.gov/Archives/edgar/data/310764/000031076426000010/syk-20251231.htm): two reportable segments, product descriptions and named competitors. The filing uses both Orthopaedics and OrthopaedicsAndSpine members for the same current annual revenue. Canonicalize only the explicitly mapped members; conflicting values remain omitted.
- [Deere FY2025 10-K](https://www.sec.gov/Archives/edgar/data/315189/000110465925122321/de-20251102x10k.htm): four business segments and named equipment competitors. Use the explicit operating-segments consolidation context, rather than mixing it with differently defined revenue contexts. Resolve USD through the filing's unit declaration, not its arbitrary unit ID.

Public filing fragments in `tests/fixtures/visual-segments` retain annual contexts,
units and revenue facts only. Their provenance records the full filing hash and
fragment hash. They are bounded parser regressions, not complete financial audits.
Wrong issuer/period/currency and conflicting-alias controls remain excluded.

## Rendering and retrieval observations

A read-only public retrieval returned five annual records, four Deere segments,
two Stryker segments and three same-date peer quotes for each. No retrieval gaps
were reported in those two runs. Deere operating income remains unavailable under
the existing metric mapping; the visual does not invent an operating margin.
Consolidated FCF is explicitly distinguished from equipment-only cash flow.

Local PDF rendering with the existing WeasyPrint environment produced one A4 page
for each public-data-only Deere/Stryker layout and the existing saved UNH report.
Text inspection found only Calibri variants, black text and zero text spans outside
the page. Images were visually inspected. The two new-company layouts were labeled
LAYOUT QA / PUBLIC DATA ONLY and contain no invented investment thesis, debates,
catalysts or research citations. This is not analyst acceptance of a complete new
research report. Dense, long-source layouts need additional boundary testing.

New equipment and medical-technology art was generated as a conceptual industry
illustration on October 11. It is not an actual company facility, branded product,
or factual source. Labels, figures and citations are rendered separately as text.
The assets are embedded by the backend and committed under assets/research-visuals;
no external image hosting or model call is required to render a saved page.

Existing reports, source originals, approvals and stored fundamentals are not
reprocessed by deployment. New collection uses the corrected parser. No paid
research was launched for these checks.
