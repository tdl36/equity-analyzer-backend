import * as React from 'react';
import { changeBrief } from './research-changes-model.mjs';
var stamp = value => value ? new Date(value).toLocaleString() : 'Time unavailable';
function Evidence({
  items = []
}) {
  return items.map((e, i) => /*#__PURE__*/React.createElement("blockquote", {
    key: i
  }, /*#__PURE__*/React.createElement("strong", null, e.source?.filename || e.filename || e.sourceId || 'Saved source'), /*#__PURE__*/React.createElement("p", {
    style: {
      whiteSpace: 'pre-wrap'
    }
  }, e.excerpt || 'No excerpt saved.')));
}
export function ResearchChanges({
  api,
  ticker,
  revision,
  dirty,
  onEvidence,
  onHistory
}) {
  var [data, setData] = React.useState(null),
    [error, setError] = React.useState(''),
    [selected, setSelected] = React.useState('review'),
    [epoch, setEpoch] = React.useState(0),
    [busy, setBusy] = React.useState(false);
  React.useEffect(() => {
    var live = true;
    var c = new AbortController();
    var timeout = setTimeout(() => c.abort(), 20000);
    setBusy(true);
    setError('');
    setData(null);
    var read = async path => {
      var r = await fetch(api + path, {
        signal: c.signal
      });
      var d = await r.json();
      if (!r.ok) throw Error(d.error || 'Change brief unavailable');
      return d;
    };
    Promise.all([read('/api/research/lifecycle/' + encodeURIComponent(ticker)), read('/api/research/investment-case/' + encodeURIComponent(ticker) + '/source-proposals')]).then(([history, source]) => {
      if (live) setData({
        history,
        source
      });
    }).catch(() => {
      if (live) setError('The change brief could not be loaded completely. Refresh to retrieve saved history and proposals.');
    }).finally(() => {
      clearTimeout(timeout);
      if (live) setBusy(false);
    });
    return () => {
      live = false;
      c.abort();
      clearTimeout(timeout);
    };
  }, [api, ticker, revision, epoch]);
  var brief = changeBrief(data?.history, data?.source?.proposals, selected);
  return /*#__PURE__*/React.createElement("section", {
    "aria-label": "Research change brief",
    className: "thesis-evolution"
  }, /*#__PURE__*/React.createElement("p", {
    className: "workspace-eyebrow"
  }, "SINCE YOUR REVIEW"), /*#__PURE__*/React.createElement("h2", null, "What changed?"), /*#__PURE__*/React.createElement("p", null, "Saved case changes and source-backed proposals, assembled without a new research run. This view does not assess materiality or approve changes."), dirty && /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, "Your unsaved case edits are excluded. This brief compares saved revisions only."), /*#__PURE__*/React.createElement("div", {
    className: "evolution-controls"
  }, /*#__PURE__*/React.createElement("label", null, "Compare against", /*#__PURE__*/React.createElement("select", {
    value: selected,
    onChange: e => setSelected(e.target.value)
  }, /*#__PURE__*/React.createElement("option", {
    value: "review"
  }, "Latest recorded evidence review"), data?.history?.cases?.map(v => /*#__PURE__*/React.createElement("option", {
    key: v.revision,
    value: String(v.revision)
  }, "Saved revision ", v.revision, " \xB7 ", stamp(v.created_at))))), /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: () => setEpoch(x => x + 1)
  }, busy ? 'Refreshing…' : 'Refresh brief')), error && /*#__PURE__*/React.createElement("p", {
    role: "alert"
  }, error), busy && /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, "Loading saved evidence\u2026"), data && /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("p", null, "History captured ", stamp(data.history.asOf), ". Up to 100 history records per category and 30 recent proposals; this is not a complete news or document feed."), Object.values(data.history.limited || {}).some(Boolean) && /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, "Some history is outside this snapshot. Use Evolution to inspect the available record."), !brief.baseline ? /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, selected === 'review' ? 'No usable case baseline exists at the latest recorded evidence review. Choose a saved revision above.' : 'The selected baseline is no longer in this snapshot. Choose another saved revision.') : /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("p", null, "Comparing saved case R", brief.baseline.revision, " with R", brief.latest.revision, ". ", selected === 'review' ? `Review recorded ${stamp(brief.at)}; this was an evidence-proposal review, not approval of all research.` : 'You selected a saved revision; its save date does not establish investor review.'), /*#__PURE__*/React.createElement("h3", null, brief.changes.length, " saved case changes"), !brief.changes.length && /*#__PURE__*/React.createElement("p", null, "No tracked wording or valuation-input changes between these revisions."), brief.changes.map((x, i) => /*#__PURE__*/React.createElement("details", {
    key: i
  }, /*#__PURE__*/React.createElement("summary", null, x.label), /*#__PURE__*/React.createElement("div", {
    className: "evolution-change"
  }, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("section", null, /*#__PURE__*/React.createElement("small", null, "Baseline"), /*#__PURE__*/React.createElement("p", {
    style: {
      whiteSpace: 'pre-wrap'
    }
  }, x.before || 'Not recorded')), /*#__PURE__*/React.createElement("section", null, /*#__PURE__*/React.createElement("small", null, "Current saved case"), /*#__PURE__*/React.createElement("p", {
    style: {
      whiteSpace: 'pre-wrap'
    }
  }, x.after || 'Removed / cleared')))))), /*#__PURE__*/React.createElement("h3", null, brief.accepted.length, " new accepted evidence links"), brief.accepted.map((e, i) => /*#__PURE__*/React.createElement("details", {
    key: i
  }, /*#__PURE__*/React.createElement("summary", null, e.field, " \xB7 ", e.reason || 'Accepted evidence'), /*#__PURE__*/React.createElement("p", null, e.before, " \u2192 ", e.after), /*#__PURE__*/React.createElement("p", null, e.provenance), /*#__PURE__*/React.createElement(Evidence, {
    items: e.evidence
  })))), /*#__PURE__*/React.createElement("h3", null, brief.pending.length, " unresolved proposals in this snapshot"), /*#__PURE__*/React.createElement("p", null, "These remain unapproved, including proposals that predate the baseline. An excerpt match establishes provenance, not investment truth."), !brief.pending.length && /*#__PURE__*/React.createElement("p", null, "No unresolved investment-case proposals in the returned proposal window."), brief.pending.map(p => /*#__PURE__*/React.createElement("details", {
    key: p.id
  }, /*#__PURE__*/React.createElement("summary", null, p.status.replaceAll('_', ' '), " \xB7 updated ", stamp(p.updated_at || p.created_at), p.changedSinceBaseline ? ' · updated since baseline' : ''), p.error && /*#__PURE__*/React.createElement("p", null, p.error), (p.result?.changes || []).map((x, i) => /*#__PURE__*/React.createElement("article", {
    className: "workspace-panel",
    key: x.id || i
  }, /*#__PURE__*/React.createElement("h4", null, x.field || 'Proposed change'), /*#__PURE__*/React.createElement("p", null, /*#__PURE__*/React.createElement("strong", null, "Reason:"), " ", x.reason || 'Not recorded'), /*#__PURE__*/React.createElement("p", null, /*#__PURE__*/React.createElement("strong", null, "Before:"), " ", x.before), /*#__PURE__*/React.createElement("p", null, /*#__PURE__*/React.createElement("strong", null, "Proposed:"), " ", x.after), /*#__PURE__*/React.createElement("p", null, x.passageMatched && x.reviewPassed ? 'Passed excerpt and model checks; still requires your review.' : 'Evidence or model-review checks remain unresolved.'), /*#__PURE__*/React.createElement(Evidence, {
    items: (x.evidence || []).map(e => ({
      ...e,
      source: p.result?.sources?.find(s => s.id === e.sourceId)
    }))
  }))), /*#__PURE__*/React.createElement("button", {
    onClick: onEvidence
  }, "Open evidence & proposals"))), /*#__PURE__*/React.createElement("div", {
    className: "sa-actions"
  }, /*#__PURE__*/React.createElement("button", {
    onClick: onEvidence
  }, "Review evidence & proposals"), /*#__PURE__*/React.createElement("button", {
    onClick: onHistory
  }, "Open full evolution"))));
}