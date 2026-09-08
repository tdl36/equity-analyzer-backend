import * as React from 'react';
import { eventSources } from './event-sources.mjs';
import { eventResearch } from './earnings-model.mjs';
export function EarningsWorkspace({
  api,
  onRefresh,
  activities,
  onNavigate,
  onCompany,
  renderHtml
}) {
  var [selected, setSelected] = React.useState(null),
    [query, setQuery] = React.useState(''),
    [filter, setFilter] = React.useState('all');
  var [instruction, setInstruction] = React.useState(''),
    [sending, setSending] = React.useState(false),
    [reply, setReply] = React.useState('');
  var [inventory, setInventory] = React.useState(null),
    [inventoryError, setInventoryError] = React.useState('');
  var sendLock = React.useRef(false);
  var events = eventResearch(activities);
  var visible = events.filter(e => `${e.ticker} ${e.input.topic} ${e.analystName}`.toLowerCase().includes(query.toLowerCase()) && (filter === 'all' || e.state === filter));
  var event = visible.find(e => e.id === selected) || visible[0];
  var currentEvent = React.useRef(event?.id);
  currentEvent.current = event?.id;
  var labels = {
    queued: 'Awaiting synthesis',
    running: 'Processing',
    failed: 'Needs attention',
    draft: 'Draft ready'
  };
  React.useEffect(() => {
    setInstruction('');
    setReply('');
  }, [event?.id]);
  React.useEffect(() => {
    var active = true;
    var controller;
    setInventory(null);
    setInventoryError('');
    if (!event) return;
    var refresh = async () => {
      controller?.abort();
      controller = new AbortController();
      var timeout = setTimeout(() => controller.abort(), 15000);
      try {
        var r = await fetch(`${api}/api/agent/local-files/${encodeURIComponent(event.ticker)}`, {
          signal: controller.signal
        });
        if (!r.ok) throw Error('iCloud inventory unavailable');
        var value = await r.json();
        if (active) {
          setInventory(value);
          setInventoryError('');
        }
      } catch (e) {
        if (active) setInventoryError('Could not refresh the iCloud inventory. Any displayed inventory may be stale.');
      } finally {
        clearTimeout(timeout);
      }
    };
    refresh();
    var timer = setInterval(refresh, 30000);
    return () => {
      active = false;
      clearInterval(timer);
      controller?.abort();
    };
  }, [api, event?.id, event?.ticker]);
  var liveSources = eventSources(inventory, event?.input?.topic, event?.sources || []);
  var sendRevision = async () => {
    if (!event || !instruction.trim() || sendLock.current || event.state === 'running') return;
    var submittedEvent = event.id;
    sendLock.current = true;
    setSending(true);
    setReply('Submitting revision…');
    try {
      var res = await fetch(`${api}/api/analyst-activities/${encodeURIComponent(event.id)}/regenerate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          customInstructions: instruction.trim()
        }),
        signal: AbortSignal.timeout(20000)
      });
      var body = await res.json();
      if (!res.ok) throw Error(body.error || 'Revision could not be queued');
      if (currentEvent.current === submittedEvent) {
        setReply('Revision queued with the covering analyst. The previous draft is preserved below. Review the new draft when processing finishes.');
        setInstruction('');
      }
      await onRefresh();
    } catch (e) {
      if (currentEvent.current === submittedEvent) setReply(`Submission not confirmed: ${e.message}. Refresh the event before retrying to avoid duplicate work.`);
      await onRefresh();
    } finally {
      sendLock.current = false;
      setSending(false);
    }
  };
  return /*#__PURE__*/React.createElement("section", {
    className: "earnings-workspace"
  }, /*#__PURE__*/React.createElement("header", {
    className: "workspace-panel"
  }, /*#__PURE__*/React.createElement("p", {
    className: "workspace-eyebrow"
  }, "EARNINGS / EVIDENCE / INVESTMENT IMPACT"), /*#__PURE__*/React.createElement("h2", null, "The event, in context."), /*#__PURE__*/React.createElement("p", {
    className: "desk-explainer"
  }, "Review the source record, inspect the synthesis and compare it with your investment case. Active and recently failed analyst events appear here; approved events remain in the Analyst archive."), /*#__PURE__*/React.createElement("div", {
    className: "earnings-counts"
  }, [['all', 'Events'], ['draft', 'Drafts ready'], ['running', 'Processing'], ['failed', 'Need attention']].map(([id, label]) => /*#__PURE__*/React.createElement("button", {
    key: id,
    "aria-pressed": filter === id,
    onClick: () => setFilter(id)
  }, /*#__PURE__*/React.createElement("strong", null, id === 'all' ? events.length : events.filter(e => e.state === id).length), label)))), /*#__PURE__*/React.createElement("div", {
    className: "earnings-layout"
  }, /*#__PURE__*/React.createElement("aside", {
    className: "workspace-panel"
  }, /*#__PURE__*/React.createElement("label", {
    className: "earnings-search"
  }, "Find an event", /*#__PURE__*/React.createElement("input", {
    value: query,
    onChange: e => setQuery(e.target.value),
    placeholder: "Ticker, event or analyst"
  })), /*#__PURE__*/React.createElement("nav", {
    "aria-label": "Earnings events"
  }, visible.map(e => /*#__PURE__*/React.createElement("button", {
    className: "earnings-event",
    key: e.id,
    "aria-current": event?.id === e.id ? 'true' : undefined,
    onClick: () => setSelected(e.id)
  }, /*#__PURE__*/React.createElement("strong", null, e.ticker, " ", /*#__PURE__*/React.createElement("small", null, labels[e.state])), /*#__PURE__*/React.createElement("span", null, e.input.topic), /*#__PURE__*/React.createElement("small", null, e.analystName || 'Analyst team')))), !visible.length && /*#__PURE__*/React.createElement("p", {
    className: "workspace-empty"
  }, "No events match this view. Create an event from a CATALYSTS folder in Analyst team."), /*#__PURE__*/React.createElement("button", {
    className: "workspace-link-row",
    onClick: () => onNavigate('analysts')
  }, "Open Analyst team \u2192")), event && /*#__PURE__*/React.createElement("article", {
    className: "workspace-panel earnings-detail",
    key: event.id
  }, /*#__PURE__*/React.createElement("p", {
    className: "workspace-eyebrow"
  }, event.ticker, " / ", labels[event.state]), /*#__PURE__*/React.createElement("h2", null, event.input.topic), /*#__PURE__*/React.createElement("ol", {
    className: "earnings-stages",
    "aria-label": "Evidence processing status"
  }, /*#__PURE__*/React.createElement("li", null, "Event detected"), /*#__PURE__*/React.createElement("li", null, event.sources.length ? `${event.sources.length} source names recorded` : 'Source record pending'), /*#__PURE__*/React.createElement("li", null, labels[event.state]), /*#__PURE__*/React.createElement("li", null, "Investment review pending")), event.error && /*#__PURE__*/React.createElement("p", {
    className: "workspace-error",
    role: "alert"
  }, String(event.error)), /*#__PURE__*/React.createElement("section", {
    className: "earnings-impact"
  }, /*#__PURE__*/React.createElement("h3", null, "Instruct ", event.analystName || 'the covering analyst'), /*#__PURE__*/React.createElement("p", null, "Request a revision to this event\u2019s recap\u2014for example, \u201CReconcile the guidance figures and explain the extra-week effect.\u201D This starts a source-based generation using configured API credits; it does not instantly edit or approve saved research."), /*#__PURE__*/React.createElement("div", {
    role: "log",
    "aria-label": "Revision instructions"
  }, (event.output?.revisionInstructions || []).map((m, i) => /*#__PURE__*/React.createElement("p", {
    key: i
  }, /*#__PURE__*/React.createElement("strong", null, "You:"), " ", m.content))), /*#__PURE__*/React.createElement("label", {
    className: "earnings-search"
  }, "Revision instructions", /*#__PURE__*/React.createElement("textarea", {
    rows: 3,
    maxLength: 6000,
    value: instruction,
    onChange: e => setInstruction(e.target.value),
    disabled: sending || event.state === 'running',
    placeholder: "Tell the analyst what to correct, challenge or expand\u2026"
  })), /*#__PURE__*/React.createElement("button", {
    className: "workspace-primary",
    disabled: sending || event.state === 'running' || !instruction.trim(),
    onClick: sendRevision
  }, sending ? 'Submitting…' : event.state === 'running' ? 'Analyst working…' : 'Send revision request'), /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, reply), !!event.output?.priorRuns?.length && /*#__PURE__*/React.createElement("details", null, /*#__PURE__*/React.createElement("summary", null, "Previous drafts \xB7 ", event.output.priorRuns.length), event.output.priorRuns.map((r, i) => /*#__PURE__*/React.createElement("details", {
    key: i
  }, /*#__PURE__*/React.createElement("summary", null, "Draft ", i + 1, " \xB7 ", r.completedAt || 'Date unavailable'), /*#__PURE__*/React.createElement("div", {
    className: "desk-report-prose",
    dangerouslySetInnerHTML: {
      __html: renderHtml(r.synthesisMarkdown || '')
    }
  }))))), event.output?.claimReview && /*#__PURE__*/React.createElement("section", {
    className: "earnings-impact"
  }, /*#__PURE__*/React.createElement("h3", null, "What changed \xB7 evidence review"), /*#__PURE__*/React.createElement("p", null, "Source quotations are checked by text matching. A separate model assesses support. These checks cover selected claims and remain subject to your review."), (event.output.claimReview.changes || []).map((c, i) => /*#__PURE__*/React.createElement("div", {
    key: i
  }, /*#__PURE__*/React.createElement("h4", null, c.area, ": ", c.change), /*#__PURE__*/React.createElement("p", null, c.implication), /*#__PURE__*/React.createElement("small", null, c.baselineAvailable ? 'Comparison baseline supplied' : 'No verified prior baseline', " \xB7 Supporting claims: ", c.claimIds.join(', ') || 'None', " \xB7 Analyst review required"))), !!event.output.claimReview.numericComparisons?.length && /*#__PURE__*/React.createElement("details", {
    open: true
  }, /*#__PURE__*/React.createElement("summary", null, "Numerical reconciliation \xB7 ", event.output.claimReview.numericComparisons.length), event.output.claimReview.numericComparisons.map((n, i) => /*#__PURE__*/React.createElement("article", {
    className: "amendment-card",
    key: i
  }, /*#__PURE__*/React.createElement("h4", null, n.metric, " \xB7 ", n.status === 'arithmetic_checked' ? 'Arithmetic checked' : 'Comparison needs review'), /*#__PURE__*/React.createElement("div", {
    className: "evidence-compare"
  }, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("strong", null, "Actual / updated"), /*#__PURE__*/React.createElement("p", null, n.actual?.value, " ", n.actual?.unit, " \xB7 ", n.actual?.period, " \xB7 ", n.actual?.basis), /*#__PURE__*/React.createElement("details", null, /*#__PURE__*/React.createElement("summary", null, n.actual?.filename || 'Source unavailable'), /*#__PURE__*/React.createElement("blockquote", null, n.actual?.quote))), /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("strong", null, n.benchmarkType?.replaceAll('_', ' ') || 'Benchmark'), /*#__PURE__*/React.createElement("p", null, n.benchmark?.value, " ", n.benchmark?.unit, " \xB7 ", n.benchmark?.period, " \xB7 ", n.benchmark?.basis), /*#__PURE__*/React.createElement("details", null, /*#__PURE__*/React.createElement("summary", null, n.benchmark?.filename || 'Source unavailable'), /*#__PURE__*/React.createElement("blockquote", null, n.benchmark?.quote)))), n.delta !== null && /*#__PURE__*/React.createElement("p", null, "Calculated difference: ", n.delta, " ", n.deltaUnit, n.relativePercent !== null ? ` · ${n.relativePercent}% relative change` : '', n.basisPointDelta !== null ? ` · ${n.basisPointDelta} bps` : ''), /*#__PURE__*/React.createElement("ul", null, (n.issues || []).map((issue, k) => /*#__PURE__*/React.createElement("li", {
    key: k
  }, issue))), /*#__PURE__*/React.createElement("p", null, n.limitation)))), /*#__PURE__*/React.createElement("details", {
    open: true
  }, /*#__PURE__*/React.createElement("summary", null, "Selected claim checks \xB7 ", event.output.claimReview.claims?.length || 0), (event.output.claimReview.claims || []).map(c => /*#__PURE__*/React.createElement("details", {
    key: c.id
  }, /*#__PURE__*/React.createElement("summary", null, "#", c.id, " ", c.passageMatched && c.reviewPassed ? 'Passage matched · model check passed' : 'Needs evidence review', " \u2014 ", c.statement), /*#__PURE__*/React.createElement("p", null, c.kind, " \xB7 ", c.filename || 'Unknown source', c.page ? ` · page ${c.page}` : ''), /*#__PURE__*/React.createElement("blockquote", null, c.quote), c.reviewIssue && /*#__PURE__*/React.createElement("p", null, c.reviewIssue)))), /*#__PURE__*/React.createElement("ul", null, (event.output.claimReview.limitations || []).map((l, i) => /*#__PURE__*/React.createElement("li", {
    key: i
  }, l)))), /*#__PURE__*/React.createElement("section", {
    className: "earnings-impact"
  }, /*#__PURE__*/React.createElement("h3", null, "Event folder \u2192 current draft"), inventoryError && /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, inventoryError), !liveSources.available ? /*#__PURE__*/React.createElement("p", null, "The local agent has not provided an inventory for this view yet. This does not mean the event folder is empty.") : /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("p", null, liveSources.files.length, " source files reported in CATALYSTS / ", event.ticker, " / ", event.input.topic, ". Last report: ", liveSources.updated, ". Inventory presence is not proof of readability."), /*#__PURE__*/React.createElement("p", null, event.draft ? `${liveSources.added.length} filenames not recorded in this draft · ${liveSources.missing.length} recorded filenames no longer in the inventory` : 'Draft source comparison will be available after synthesis.'), /*#__PURE__*/React.createElement("details", null, /*#__PURE__*/React.createElement("summary", null, "Current event files"), /*#__PURE__*/React.createElement("ul", null, liveSources.files.map((f, i) => /*#__PURE__*/React.createElement("li", {
    key: i
  }, f.path || f.filename, " \xB7 ", Math.round((f.size || 0) / 1024), " KB")))), /*#__PURE__*/React.createElement("p", null, "Changes to existing file contents are not detected by this filename comparison. Use the captured source hashes when auditing a specific draft."))), /*#__PURE__*/React.createElement("h3", null, "Source coverage"), /*#__PURE__*/React.createElement("p", {
    className: "desk-explainer"
  }, "Inferred from the filenames recorded with this output. \u201CNot identified\u201D does not prove a document is absent from iCloud. Broker reports and other sources are listed below."), /*#__PURE__*/React.createElement("div", {
    className: "earnings-coverage"
  }, event.coverage.map(c => /*#__PURE__*/React.createElement("div", {
    key: c.id
  }, /*#__PURE__*/React.createElement("strong", null, c.label), /*#__PURE__*/React.createElement("span", null, c.files.length ? `${c.files.length} identified` : 'Not identified')))), /*#__PURE__*/React.createElement("details", null, /*#__PURE__*/React.createElement("summary", null, "Source register \xB7 ", event.sources.length, " named documents"), event.sources.length ? /*#__PURE__*/React.createElement("ul", null, event.sources.map(s => /*#__PURE__*/React.createElement("li", {
    key: s
  }, s))) : /*#__PURE__*/React.createElement("p", null, "No source filenames were attached to this activity.")), event.output?.evidenceSnapshot?.version === 1 && /*#__PURE__*/React.createElement("details", {
    open: true
  }, /*#__PURE__*/React.createElement("summary", null, "Recorded synthesis inputs \xB7 ", event.output.evidenceSnapshot.sources?.length || 0, " documents"), /*#__PURE__*/React.createElement("p", {
    className: "desk-explainer"
  }, "Captured when this draft was generated. This records source delivery, not claim verification."), /*#__PURE__*/React.createElement("ul", null, (event.output.evidenceSnapshot.sources || []).map((s, i) => /*#__PURE__*/React.createElement("li", {
    key: i
  }, /*#__PURE__*/React.createElement("strong", null, s.filename), " \xB7 ", s.inputMode === 'native_pdf' ? 'Native PDF' : s.inputMode === 'extracted_text' ? 'PDF text extraction' : 'Text', s.pages ? ` · ${s.pages} pages` : '', s.characters != null ? ` · ${s.characters.toLocaleString()} characters` : '', /*#__PURE__*/React.createElement("br", null), /*#__PURE__*/React.createElement("small", null, "SHA-256: ", s.sha256))))), event.output?.processingRecovery && /*#__PURE__*/React.createElement("p", {
    className: "desk-explainer"
  }, "Processing record: ", event.output.processingRecovery.totalBatches, " source batches \xB7 ", event.output.processingRecovery.resumedBatches, " reused from a matching recovery checkpoint."), /*#__PURE__*/React.createElement("div", {
    className: "earnings-checks"
  }, /*#__PURE__*/React.createElement("h3", null, "Evidence review"), /*#__PURE__*/React.createElement("ul", null, /*#__PURE__*/React.createElement("li", null, event.sourceMismatch ? `Coverage discrepancy: ${event.expected} documents reported, ${event.sources.length} filenames recorded.` : event.sources.length ? 'Source filenames available for review.' : 'Source completeness cannot be assessed yet.'), /*#__PURE__*/React.createElement("li", null, event.output?.claimReview ? 'A selected-claim source review is attached.' : event.provenance ? 'A legacy model-generated source contribution record is attached.' : 'No source contribution record attached.'), /*#__PURE__*/React.createElement("li", null, event.output?.evidenceSnapshot?.version === 1 ? 'Synthesis input snapshot recorded.' : 'This recap predates input snapshots; source delivery has not been independently recorded.'), /*#__PURE__*/React.createElement("li", null, "Claim accuracy, page-level citations and numerical consistency are not independently verified for this recap."))), event.provenance && !event.output?.claimReview && /*#__PURE__*/React.createElement("details", null, /*#__PURE__*/React.createElement("summary", null, "Source contribution record \xB7 model generated"), /*#__PURE__*/React.createElement("pre", {
    className: "earnings-provenance"
  }, typeof event.provenance === 'string' ? event.provenance : JSON.stringify(event.provenance, null, 2))), /*#__PURE__*/React.createElement("section", {
    className: "earnings-impact"
  }, /*#__PURE__*/React.createElement("h3", null, "Review the investment implications"), /*#__PURE__*/React.createElement("p", null, "Compare the synthesis below with your saved thesis: what changed in earnings power, guidance, valuation, catalysts and downside risk? Separate reported facts from estimates and interpretation."), /*#__PURE__*/React.createElement("button", {
    className: "workspace-link-row",
    onClick: () => onCompany(event.ticker, 'portfolio')
  }, "Compare with ", event.ticker, " thesis \u2192")), /*#__PURE__*/React.createElement("div", {
    className: "workspace-section-heading"
  }, /*#__PURE__*/React.createElement("h3", null, event.state === 'failed' && event.draft ? 'Previous synthesis · latest attempt failed' : 'Event synthesis'), /*#__PURE__*/React.createElement("button", {
    onClick: () => onNavigate('analysts')
  }, "Revise or approve in Analyst team \u2197")), event.draft ? /*#__PURE__*/React.createElement("div", {
    className: "desk-report-prose",
    dangerouslySetInnerHTML: {
      __html: renderHtml(event.output.synthesisMarkdown)
    }
  }) : /*#__PURE__*/React.createElement("p", {
    className: "workspace-empty"
  }, event.state === 'failed' ? 'Synthesis failed. Inspect the error and source folder before retrying in Analyst team.' : event.state === 'running' ? 'Synthesis is processing. This workspace refreshes with the research desk.' : 'No draft is attached yet. Open Analyst team to inspect or start synthesis.'))));
}