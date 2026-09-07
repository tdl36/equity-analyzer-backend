import * as React from 'react';
import { eventResearch } from './earnings-model.mjs';
export function EarningsWorkspace({
  activities,
  onNavigate,
  onCompany,
  renderHtml
}) {
  var [selected, setSelected] = React.useState(null),
    [query, setQuery] = React.useState(''),
    [filter, setFilter] = React.useState('all');
  var events = eventResearch(activities);
  var visible = events.filter(e => `${e.ticker} ${e.input.topic} ${e.analystName}`.toLowerCase().includes(query.toLowerCase()) && (filter === 'all' || e.state === filter));
  var event = visible.find(e => e.id === selected) || visible[0];
  var labels = {
    queued: 'Awaiting synthesis',
    running: 'Processing',
    failed: 'Needs attention',
    draft: 'Draft ready'
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
  }, String(event.error)), /*#__PURE__*/React.createElement("h3", null, "Source coverage"), /*#__PURE__*/React.createElement("p", {
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
  }, /*#__PURE__*/React.createElement("strong", null, s.filename), " \xB7 ", s.inputMode === 'native_pdf' ? 'Native PDF' : s.inputMode === 'extracted_text' ? 'PDF text extraction' : 'Text', s.pages ? ` · ${s.pages} pages` : '', s.characters != null ? ` · ${s.characters.toLocaleString()} characters` : '', /*#__PURE__*/React.createElement("br", null), /*#__PURE__*/React.createElement("small", null, "SHA-256: ", s.sha256))))), /*#__PURE__*/React.createElement("div", {
    className: "earnings-checks"
  }, /*#__PURE__*/React.createElement("h3", null, "Evidence review"), /*#__PURE__*/React.createElement("ul", null, /*#__PURE__*/React.createElement("li", null, event.sourceMismatch ? `Coverage discrepancy: ${event.expected} documents reported, ${event.sources.length} filenames recorded.` : event.sources.length ? 'Source filenames available for review.' : 'Source completeness cannot be assessed yet.'), /*#__PURE__*/React.createElement("li", null, event.provenance ? 'A model-generated source contribution record is attached.' : 'No source contribution record attached.'), /*#__PURE__*/React.createElement("li", null, event.output?.evidenceSnapshot?.version === 1 ? 'Synthesis input snapshot recorded.' : 'This recap predates input snapshots; source delivery has not been independently recorded.'), /*#__PURE__*/React.createElement("li", null, "Claim accuracy, page-level citations and numerical consistency are not independently verified for this recap."))), event.provenance && /*#__PURE__*/React.createElement("details", null, /*#__PURE__*/React.createElement("summary", null, "Source contribution record \xB7 model generated"), /*#__PURE__*/React.createElement("pre", {
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