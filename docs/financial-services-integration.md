# Financial-services research integration

Charlie T391 adapts the research methodology from
https://github.com/anthropics/financial-services at commit
`574ed3624aebd0418c7e96cd101262f30210ab26` (Apache-2.0).
The complete checkout was downloaded to
`/Users/tonydlee/Projects/charlie-vendor/financial-services-20261020`.
Selected original instructions, license and SHA-256 manifest are checked into
`vendor/anthropics-financial-services`; these are reference material, not executable
plugins. Charlie's modified adaptation is `research_writing.FINANCIAL_SERVICES`.

## Active integration

New Stock Analysis jobs and new package revisions freeze `synthesis-v3`.
The existing Stock Analysis review pass also checks for superseded estimates,
unsupported valuation inferences and conflicting accounting periods. Package
rewrites remain editorial alternatives without an added review call.
The existing coordinator, model selection, frozen evidence, review stages and
explicit processing approval remain in use. The adaptation is passed to the actual
model prompt through the shared writing contract, including the thesis/stock summary
and one-pager package revision. It adds no model calls or external connectors.
Planning estimates include the added prompt text. Complete source extraction, short
field identities and the quoted 16k rewrite output allowance continue from v2.
Pending v1/v2 requests retain their prior prompts, source handling and fingerprints.

Upstream methods adapted:
- thesis-tracker: expectation, evidence, pillar status and falsifiable monitoring;
- model-update: old/new estimates, reasons, segment and cash-flow reconciliation,
  adjusted versus GAAP, dilution and valuation consequences;
- earnings-analysis best practices: dated expectations, explained variances,
  sourced figures and cross-document consistency;
- initiating-coverage checklist: supported thesis, financial drivers, scenario
  rationale and consistent numbers across outputs.

Charlie deliberately excludes upstream position actions, automatic rating/target
changes, report-length/chart quotas, external searches outside approved evidence,
connector installation and managed-agent orchestration. No AlphaSense API entitlement
is implied. Broker identities stay in evidence metadata. Notes retain user formatting
preferences. Human approval of research and thesis changes remains separate.

## Verification and limits

Offline tests check routing into real generation/revision prompts, v2 stability,
new-policy fingerprint separation, output-budget/identity behavior and vendored hashes.
This is methodology integration, not a deterministic financial model, independent
claim verifier or proof of better investment judgment. Existing ABT output is unchanged;
no paid regeneration is performed by installing the integration. The observed ABT
old/new estimate and PEG failures are addressed explicitly in the drafting method,
but require a separately authorized real evaluation before claiming they are solved.
Future upstream updates must be reviewed, pinned and introduced under a new policy;
never overwrite instructions used by saved requests.
