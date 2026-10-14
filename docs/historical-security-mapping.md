# Pending historical security mapping contract

The internal `historical_valuation_mapping.py` adapter prepares immutable evidence
claims. It is not connected to an API, retrieval adapter, source store or valuation
calculation. Production historical ingestion remains disabled.

A trusted caller supplies the locked assignment/edition baseline, exact candidate
digest, frozen integer CIK and provider symbol. One assignment-scoped retained
`security_reference` JSON original supplies an ID, SHA-256, bytes, retrieval date
and optional validated acquisition metadata. This is a proposed storage kind;
existing historical storage deliberately does not accept it yet.

A proposal supplies that exact receipt, three distinct top-level field citations
for CIK, symbol and quoted-security currency, plus an as-of date no later than
retrieval. Matching is exact: no ticker aliases, number coercion, implicit USD,
exchange-to-currency inference or substitution of a company reporting currency.
The current parser supports an integer CIK; providers returning string identifiers
need an explicitly tested normalization adapter before use. JSON duplicates,
nonfinite values, malformed bytes and oversized originals are rejected.

A matching field is not proof that the field describes trading currency. The
receipt remains pending with explicit currency-semantics, historical-applicability,
share-class, corporate-action, provenance and investor-review gaps. It cannot
clear valuation readiness or authorize publication. Even identical displayed
values need a new receipt when candidate, baseline, original bytes, retrieval date
or acquisition metadata change. Rehashing an altered readiness flag cannot bypass
replay against the original evidence.

Next implementation: choose and document a provider's actual retained security
reference schema and field semantics; add assignment-locked storage support with
idempotent exact-byte receipts and safe read-only inventory. Only then connect
pending mapping reviews to the historical candidate workspace. A present-day
security profile alone must never establish historical currency/share continuity.
Do not fetch providers, widen original limits, enable ingestion or apply mappings
merely to exercise this internal contract. Investor review remains independent.
