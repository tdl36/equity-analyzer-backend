import * as React from 'react';
import { compareCases, pillarState, latestWork } from './thesis-evolution-model.mjs';
var stamp = x => new Date(x).toLocaleString();
export function ThesisEvolution({
  api,
  ticker,
  revision
}) {
  var [data, setData] = React.useState(null),
    [asOf, setAsOf] = React.useState(''),
    [chosen, setChosen] = React.useState(null),
    [error, setError] = React.useState(''),
    [busy, setBusy] = React.useState(false),
    [epoch, setEpoch] = React.useState(0);
  React.useEffect(() => {
    var alive = true;
    setBusy(true);
    setError('');
    setData(null);
    setChosen(null);
    var query = '';
    try {
      if (asOf) query = '?asOf=' + encodeURIComponent(new Date(asOf).toISOString());
    } catch {
      setError('Choose a valid date.');
      setBusy(false);
      return;
    }
    fetch(`${api}/api/research/lifecycle/${encodeURIComponent(ticker)}${query}`, {
      signal: AbortSignal.timeout(20000)
    }).then(async r => {
      var d = await r.json();
      if (!r.ok) throw Error(d.error || 'History unavailable');
      return d;
    }).then(d => {
      if (alive) {
        setData(d);
        setChosen(d.cases[0]?.revision ?? null);
      }
    }).catch(e => {
      if (alive) setError(e.message);
    }).finally(() => {
      if (alive) setBusy(false);
    });
    return () => {
      alive = false;
    };
  }, [api, ticker, revision, asOf, epoch]);
  var cases = [...(data?.cases || [])].sort((a, b) => a.revision - b.revision),
    index = cases.findIndex(v => v.revision === chosen),
    selected = cases[index],
    previous = cases[index - 1];
  var pillars = [...new Map(cases.flatMap(v => (v.body.assumptions || []).map(a => [a.id, a.claim]))).entries()];
  var events = [...(data?.work || []).map(r => ({
    ...r,
    kind: r.body.kind
  })), ...(data?.decisions || []).map(r => ({
    ...r,
    kind: 'decision'
  }))].sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
  return /*#__PURE__*/React.createElement("section", {
    className: "thesis-evolution",
    "aria-label": "Thesis evolution"
  }, /*#__PURE__*/React.createElement("p", {
    className: "workspace-eyebrow"
  }, "THE INVESTMENT RECORD"), /*#__PURE__*/React.createElement("h2", null, "How our thinking evolved."), /*#__PURE__*/React.createElement("p", null, "Trace changes in the saved investment case and the decisions recorded alongside it."), /*#__PURE__*/React.createElement("div", {
    className: "evolution-controls"
  }, /*#__PURE__*/React.createElement("label", null, "Known to Charlie by \xB7 your local time", /*#__PURE__*/React.createElement("input", {
    type: "datetime-local",
    value: asOf,
    onChange: e => setAsOf(e.target.value)
  })), /*#__PURE__*/React.createElement("button", {
    onClick: () => {
      setAsOf('');
      setEpoch(x => x + 1);
    }
  }, "Return to latest / refresh")), busy && /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, "Reconstructing saved history\u2026"), error && /*#__PURE__*/React.createElement("p", {
    role: "alert"
  }, error), data && /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("p", {
    className: "desk-explainer"
  }, "Recorded through ", stamp(data.asOf), ". ", data.scope), Object.values(data.limited).some(Boolean) && /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, "This view is limited to 100 records per category. Older history is not shown; the first displayed case is a baseline, not an inferred initial thesis."), !cases.length ? /*#__PURE__*/React.createElement("p", null, "No investment case was saved by this time. Save the initial case to establish your baseline.") : /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("div", {
    className: "evolution-rail",
    "aria-label": "Saved case revisions"
  }, cases.map(v => /*#__PURE__*/React.createElement("button", {
    key: v.revision,
    "aria-pressed": chosen === v.revision,
    onClick: () => setChosen(v.revision)
  }, /*#__PURE__*/React.createElement("strong", null, "Revision ", v.revision), /*#__PURE__*/React.createElement("time", null, stamp(v.created_at)), /*#__PURE__*/React.createElement("span", null, v.body.evidenceLinks?.length || 0, " retained source links")))), /*#__PURE__*/React.createElement("div", {
    className: "evolution-matrix"
  }, /*#__PURE__*/React.createElement("table", null, /*#__PURE__*/React.createElement("caption", null, "Assumption history \xB7 \u201CRevised\u201D describes wording changes, not investment strength"), /*#__PURE__*/React.createElement("thead", null, /*#__PURE__*/React.createElement("tr", null, /*#__PURE__*/React.createElement("th", null, "Investment pillar"), cases.map(v => /*#__PURE__*/React.createElement("th", {
    key: v.revision
  }, /*#__PURE__*/React.createElement("button", {
    onClick: () => setChosen(v.revision)
  }, "R", v.revision))))), /*#__PURE__*/React.createElement("tbody", null, pillars.map(([id, label]) => /*#__PURE__*/React.createElement("tr", {
    key: id
  }, /*#__PURE__*/React.createElement("th", null, label), cases.map((v, i) => /*#__PURE__*/React.createElement("td", {
    key: v.revision,
    "data-state": pillarState(cases[i - 1]?.body, v.body, id, i > 0)
  }, pillarState(cases[i - 1]?.body, v.body, id, i > 0)))))))), selected && /*#__PURE__*/React.createElement("article", {
    className: "workspace-panel"
  }, /*#__PURE__*/React.createElement("p", {
    className: "workspace-eyebrow"
  }, previous ? `R${previous.revision} → R${selected.revision}` : 'FIRST DISPLAYED BASELINE'), /*#__PURE__*/React.createElement("h3", null, ticker, " \xB7 revision ", selected.revision), /*#__PURE__*/React.createElement("p", null, stamp(selected.created_at)), previous ? compareCases(previous.body, selected.body).map((c, i) => /*#__PURE__*/React.createElement("section", {
    className: "evolution-change",
    key: i
  }, /*#__PURE__*/React.createElement("h4", null, c.label), /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("section", null, /*#__PURE__*/React.createElement("small", null, "Before"), /*#__PURE__*/React.createElement("p", null, c.before || 'Not recorded')), /*#__PURE__*/React.createElement("section", null, /*#__PURE__*/React.createElement("small", null, "After"), /*#__PURE__*/React.createElement("p", null, c.after || 'Removed / cleared'))))) : /*#__PURE__*/React.createElement("p", {
    style: {
      whiteSpace: 'pre-wrap'
    }
  }, selected.body.thesis || 'No thesis narrative recorded.'), previous && compareCases(previous.body, selected.body).length === 0 && /*#__PURE__*/React.createElement("p", null, "No tracked thesis wording or scenario changes. This version may preserve metadata or a restoration."), /*#__PURE__*/React.createElement("details", null, /*#__PURE__*/React.createElement("summary", null, "Accepted evidence attached to this version"), !selected.body.evidenceLinks?.length && /*#__PURE__*/React.createElement("p", null, "No accepted source links were recorded."), selected.body.evidenceLinks?.map((link, i) => /*#__PURE__*/React.createElement("section", {
    key: i
  }, /*#__PURE__*/React.createElement("h4", null, link.field, " \xB7 ", link.reason), /*#__PURE__*/React.createElement("p", null, link.before, " \u2192 ", link.after), /*#__PURE__*/React.createElement("p", null, link.provenance), /*#__PURE__*/React.createElement("div", null, (link.evidence || []).map((e, j) => /*#__PURE__*/React.createElement("blockquote", {
    key: j
  }, /*#__PURE__*/React.createElement("strong", null, e.source?.filename || e.sourceId || 'Saved source'), /*#__PURE__*/React.createElement("p", {
    style: {
      whiteSpace: 'pre-wrap'
    }
  }, e.excerpt || 'No excerpt recorded'))))))))), /*#__PURE__*/React.createElement("h3", null, "Underweight decisions at this cutoff"), latestWork(data.work).filter(r => r.body.kind === 'underweight').map(r => /*#__PURE__*/React.createElement("article", {
    className: "workspace-panel",
    key: r.id
  }, /*#__PURE__*/React.createElement("h4", null, r.body.title), /*#__PURE__*/React.createElement("p", null, r.body.mandate, " \xB7 active weight ", r.body.activeWeightPct, "% \xB7 inputs as of ", r.body.asOf), /*#__PURE__*/React.createElement("p", null, r.body.rationale), /*#__PURE__*/React.createElement("p", null, "Decision: ", (r.body.reviewDecision || 'pending').replaceAll('_', ' '), " \xB7 ", r.body.outcome || 'Review remains open'), r.body.reviewConditions?.map(c => /*#__PURE__*/React.createElement("p", {
    key: c.id
  }, /*#__PURE__*/React.createElement("strong", null, c.category, " \xB7 ", c.state.replaceAll('_', ' ')), ": ", c.trigger, /*#__PURE__*/React.createElement("br", null), c.evidence)))), /*#__PURE__*/React.createElement("h3", null, "Decision and work timeline"), /*#__PURE__*/React.createElement("p", {
    className: "desk-explainer"
  }, "These records are shown by save time. Proximity to a thesis revision does not establish that it caused the change."), !events.length && /*#__PURE__*/React.createElement("p", null, "No decisions or work records at this cutoff."), events.map(r => /*#__PURE__*/React.createElement("details", {
    key: r.kind + r.id + r.revision
  }, /*#__PURE__*/React.createElement("summary", null, stamp(r.created_at), " \xB7 ", r.kind.replaceAll('_', ' '), " \xB7 ", r.body.title || r.body.decision), /*#__PURE__*/React.createElement("p", null, r.body.rationale), /*#__PURE__*/React.createElement("p", null, r.body.outcome || r.body.nextAction || r.body.revisitWhen), /*#__PURE__*/React.createElement("p", null, "Recorded revision ", r.revision, r.body.caseRevision ? ` · case baseline R${r.body.caseRevision}` : '')))));
}