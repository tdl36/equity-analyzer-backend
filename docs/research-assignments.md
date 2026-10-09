# End-to-end research assignments

Stock Analysis → One research assignment takes the active ticker, an inclusive
1–365 day window, and requested outputs. Defaults are the last 70 days and summary
note, stock summary, thesis draft/update, visual one-pager draft. Output/horizon
preferences can be saved. Starting explicitly authorizes one existing bounded
Stock Analysis run (up to six author and six source-review model calls); all
outputs reuse that report without further model calls.

The durable `mp_jobs` research_assignment owns a deterministic collection command
and Stock Analysis request ID. Mac polling creates a one-off request with frozen
scope without changing existing recurring ticker policies. The scheduled Codex
worker searches signed-in Chrome, reviews sources and exports up to eight relevant
originals. The Mac stages originals in iCloud STOCKS, verifies cloud imports and
reports exact filename/hash receipts before research can start. Existing source
restrictions and user source choices still apply. Research freezes case revision,
original hashes and detailed thesis baseline. Changes in these inputs require
attention rather than silent rebaselining.

The coordinator advances on Mac heartbeats and assignment reads. No open Charlie
page is required while the Mac is awake, online and polling. Browser collection
also requires Codex availability, the scheduled worker, signed-in Chrome and any
necessary direct AlphaSense MFA. A queue is not proof of collection. Coverage is a
selected source pack, not an exhaustive market feed or current-price/consensus
service. OCR/extraction and text bounds are those of the existing Stock Analysis
worker; inadequate source coverage remains visible.

Outputs are saved with the assignment: printable source-linked HTML note/summary,
a compact nine-panel **draft** visual with provenance, the existing full Stock
Analysis report, and a pending detailed-thesis import proposal. The visual does not
claim analyst figure approval and does not bypass Studio's reviewed-figure gate.
The thesis proposal preserves old item IDs, appends supported evidence and clearly
marks carried-forward items as not re-reviewed. It does not remove old claims,
reconcile every old conclusion or approve the detailed thesis/investment case.
Investors review contradictions, targets and significance in the thesis inbox.
Legacy references remain explicitly carried-forward references, not new collection.

Partial deliverables remain accessible when a later stage fails. Stop retains
originals and saved work, cancels research (an in-flight call may finish), and sends
a durable cancellation tombstone to the Mac so delayed creation cannot resurrect
collection. Restart has idempotent dispatch and PostgreSQL worker ownership. An
ambiguous paid call requires explicit usage inspection and retry acknowledgement;
no automatic paid replay. Resume source/auth failures in Collection controls, then
resume the assignment. A failed cloud-to-Mac creation needs its cause fixed and a
new assignment after stopping the failed one. Completed collection receipts are
persisted in the parent so rotating the Mac's latest-50 snapshot cannot lose them.

No real collection or paid research is needed for QA. Test with the isolated
PostgreSQL and temporary SQLite fixtures in test_research_assignments.py, the
existing stock-analysis worker tests, and intercepted synthetic frontend requests.
