# Stock Research Studio in Charlie

Open **Companies → Stock analysis**, choose a company, and open its investment case. The same tab is available within the Research desk’s investment thesis workspace. Company context is preserved in `#view=stockanalysis&ticker=SYK`.

1. Choose Snapshot, Deep research or Update thesis, an investment horizon and optional question.
2. Select up to eight permitted originals already stored under the company. Confirm issuer relevance and source permissions.
3. Generate analysis. Each report freezes the canonical `mp_companies.id`, ticker, original hashes, saved case baseline, research settings and prior completed report. Current coverage is the selected source pack, not an automatic public-web search.
4. Read the research sections and inspect field-level source passages. Missing prices, financial history, consensus and revision data remain unavailable. Same-model source review and quotation matching are not independent factual certification.
5. Use **What changed?** to inspect earlier/current section values and added, removed or changed originals. Changed supported passages are distinguished from changed wording/interpretation; neither automatically establishes an economic change or triggered thesis breaker.
6. Review figures, periods and definitions before **Generate infographic**. The deterministic nine-panel HTML visual preserves source-supported report fields, renders comparable numeric historical series, and is saved in existing Charlie Studio storage. Export HTML for printing/PDF, or export the report and provenance JSON.
7. **Prepare thesis draft for review** copies supported thesis statements into unsaved case fields. The investor reviews and saves explicitly. Existing assumptions and operating models remain in the draft. This is not an approved thesis proposal or a verified evidence-link receipt.

## Architecture and compatibility

The investor-supplied Stock Research Studio's schema and research doctrine are adapted in `stock_analysis_schema.json` and `stock_analysis_prompt.txt`. The standalone Node server, local JSON storage, automatic demo fallback and model defaults are not used.

`stock_analysis.py` validates structured section groups, field-level citations and evidence reviews, checkpoints twelve bounded stages, and compares frozen reports. `company_research.py` provides the existing durable database worker, permission/hash rechecks, per-company unfinished-run exclusion, ownership, stop/resume and submission identity for the separate `stock_analysis_runs` ledger. Existing company research and committee ledgers remain compatible.

`stock_analysis_visual.py` exports escaped HTML and creates idempotent visuals in `studio_outputs`, with the report ID, company ID, renderer fingerprint and analyst figure-review attestation. It uses no image model and cannot invent graphical labels. Negative/mixed-unit/ambiguous/unsupported series are not plotted. Chart extraction is deliberately conservative, not automated financial reconciliation.

The React `StockAnalysis` workspace lives within the existing investment case. It shares Charlie's authentication and API wrapper. All application routes remain under the global authentication gate. No new environment variables are required: existing database/authentication settings, configured research-model credentials and budget policy apply. No model defaults, thinking settings or token caps were increased.

## Recovery

An unchanged submission reuses its request ID. Look at saved runs after a network error before retrying. The database stores each stage before the provider call. Unknown call outcomes require explicit acknowledgement before retry because the call may already have been billed. Resume skips completed stages and rechecks original hashes and permissions. Stop retains saved progress. Backend restarts do not automatically replay paid calls. Complete reports are immutable. Infographic retries reuse the same saved renderer/report fingerprint.

## Verification and limits

Synthetic tests cover schema rejection, unknown citations, quotation matching, interrupted calls, conservative chart parsing, escaping, comparisons and thesis draft safeguards. Disposable PostgreSQL tests exercise the real worker, report/version readback across blueprint restart, duplicate submission, canonical company linkage, source permission changes, comparison selection and idempotent Studio storage. They never connect to the user's database or a model provider. Browser checks cover 1440/390/320px with the real component and synthetic API responses.

No paid real-source report or live image generation is performed solely for QA. User-driven real-source acceptance remains outstanding. Licensed consensus/market data, live public-web retrieval and AI-generated company illustrations are not added in this integration. Visuals use deterministic typography, panels and charts; export is HTML/JSON, with PDF available through browser printing. Reports can describe scenarios but do not independently certify their arithmetic; Charlie's existing Operating scenarios remains the validated calculation workspace.
