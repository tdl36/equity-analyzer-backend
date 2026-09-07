# Automatic research intake

Enabled via Evidence & changes → Automatic source intake. Detects new supported filenames reported in STOCKS ticker roots or uploaded to Charlie. Initial inventories are baselined on enablement; historical files do not create a wave of comparisons. A replacement under an existing filename is not a new event. CATALYSTS-specific intake remains on its existing workflow.

Intake events persist in mp_jobs (`auto_evidence_intake`). Document identities and configuration persist in app_settings under a transaction advisory lock. Repeated agent scans, duplicate uploads and disappearance/reappearance do not create another event. Agent manifests and heartbeats wake the worker. An active Mac/local agent is required for timely iCloud import and queue dispatch.

Events settle for two minutes, then up to ten files for one company are reserved. Companies need a saved thesis and cannot have another unresolved proposal. Default limit: two automatic comparisons per UTC day, configurable from 1–5 through the settings endpoint. Reservations count even if an import fails; the limit cannot be bypassed by retries. Existing monthly budget checks also gate dispatch, but estimated costs are not a hard dollar ceiling.

The worker fetches missing sources using the existing local-agent bridge, then submits the evidence amendment workflow. Imports never launch a comparison if any requested file remains missing. Existing import requests are not overwritten. Proposal application always remains manual. Pausing stops new dispatch and is checked again after import; an already running model call cannot be unbilled. No external notifications are sent.

Import interruption is marked attention after 30 minutes. Attention cases should be inspected before manually comparing the documents; processing is never silently replayed. Proposal execution remains a process-bound thread with durable records, not a fully resumable distributed queue. Credentials are resolved server-side and never stored in the intake ledger.

## AlphaSense rollout status — September 7, 2026

The current universe has 52 tickers. Only DE, ABT and AMT were collected in the August 7–September 6 pilot. There are 49 remaining names, with no collection completion implied:

A, ABBV, AMZN, AVB, BDX, BMY, BSX, CAH, CARR, CI, COR, CP, CRM, CSX, CVS, DGX, DHR, DLR, DOV, ELV, ETN, GD, HCA, HON, HUM, IQV, JNJ, LLY, MDT, MMM, MRK, MSFT, NOW, NSC, PFE, PH, PLD, REGN, RSG, RTX, TERN, TMO, UNH, UNP, VRTX, VST, VTR, WAT, WELL.

The collector validates downloads, deduplicates originals, handles sign-in pauses, routes permitted documents into iCloud and verifies manifest visibility. Browser discovery/export is still supervised. It has no autonomous AlphaSense login/MFA daemon or recurring full-universe collection schedule. Expanding requires batched collection sessions (maximum 12 tickers per run), preserving broker usage restrictions and verifying each handoff. Automatic Charlie intake does not collect documents from AlphaSense itself.
