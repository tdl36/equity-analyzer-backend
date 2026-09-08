import * as React from 'react';
import { parseTimestamp } from './workspace-model.mjs';
var stamp = v => parseTimestamp(v)?.toLocaleString() || 'Date unavailable';
var labels = {
  source: 'Source imported',
  note: 'Note version',
  review: 'Review version',
  activity: 'Analyst activity',
  job: 'Research / control request',
  collection: 'Browser collection'
};
export function ResearchHistory({
  api,
  onSection,
  onNavigate
}) {
  var [input, setInput] = React.useState(''),
    [ticker, setTicker] = React.useState(''),
    [kind, setKind] = React.useState('all'),
    [data, setData] = React.useState(null),
    [error, setError] = React.useState(''),
    [refresh, setRefresh] = React.useState(0);
  React.useEffect(() => {
    var active = true;
    var controller = new AbortController();
    var timer = setTimeout(() => controller.abort(), 20000);
    setData(null);
    setError('');
    fetch(`${api}/api/research/history${ticker ? `?ticker=${encodeURIComponent(ticker)}` : ''}`, {
      signal: controller.signal
    }).then(async r => {
      var d = await r.json();
      if (!r.ok) throw Error(d.error || `History unavailable (${r.status})`);
      if (active) setData(d);
    }).catch(e => {
      if (active) setError(e.message);
    }).finally(() => clearTimeout(timer));
    return () => {
      active = false;
      controller.abort();
      clearTimeout(timer);
    };
  }, [api, ticker, refresh]);
  var rows = (data?.records || []).filter(r => kind === 'all' || r.kind === kind);
  var open = r => r.kind === 'collection' || r.kind === 'job' && r.title === 'collection_control' ? onSection('collection') : r.kind === 'activity' ? onNavigate('analysts') : r.kind === 'note' ? onNavigate('pipeline') : onSection('evidence');
  return /*#__PURE__*/React.createElement("section", {
    className: "workspace-panel"
  }, /*#__PURE__*/React.createElement("p", {
    className: "workspace-eyebrow"
  }, "RESEARCH HISTORY"), /*#__PURE__*/React.createElement("h2", null, "From source intake to saved research."), /*#__PURE__*/React.createElement("p", {
    className: "desk-explainer"
  }, "Browse imports, collection requests, analyst activities and saved document versions together. Each row shows a real stored record and its current status."), /*#__PURE__*/React.createElement("form", {
    className: "desk-filter",
    onSubmit: e => {
      e.preventDefault();
      setTicker(input.trim().toUpperCase());
      setRefresh(n => n + 1);
    }
  }, /*#__PURE__*/React.createElement("label", null, "Ticker", /*#__PURE__*/React.createElement("input", {
    value: input,
    onChange: e => setInput(e.target.value),
    maxLength: 20,
    placeholder: "All companies"
  })), /*#__PURE__*/React.createElement("label", null, "Record type", /*#__PURE__*/React.createElement("select", {
    value: kind,
    onChange: e => setKind(e.target.value)
  }, /*#__PURE__*/React.createElement("option", {
    value: "all"
  }, "All records"), Object.entries(labels).map(([k, l]) => /*#__PURE__*/React.createElement("option", {
    key: k,
    value: k
  }, l)))), /*#__PURE__*/React.createElement("button", {
    type: "submit"
  }, "Load history")), error ? /*#__PURE__*/React.createElement("p", {
    className: "workspace-error",
    role: "alert"
  }, error, " ", /*#__PURE__*/React.createElement("button", {
    onClick: () => setRefresh(n => n + 1)
  }, "Retry")) : !data ? /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, "Loading saved history\u2026") : /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("p", {
    className: "desk-explainer"
  }, data.scope), /*#__PURE__*/React.createElement("p", null, rows.length, " shown \xB7 Mac collection snapshot: ", stamp(data.collectionReportedAt)), !rows.length && /*#__PURE__*/React.createElement("p", null, "No records match this view. Broaden the record filter or select a ticker to search its latest history."), /*#__PURE__*/React.createElement("ol", {
    className: "research-history-list"
  }, rows.map(r => /*#__PURE__*/React.createElement("li", {
    key: r.id
  }, /*#__PURE__*/React.createElement("article", null, /*#__PURE__*/React.createElement("div", {
    className: "workspace-section-heading"
  }, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("small", null, labels[r.kind], " \xB7 ", r.ticker || 'Coverage group'), /*#__PURE__*/React.createElement("h3", null, r.title.replaceAll('_', ' '))), /*#__PURE__*/React.createElement("span", {
    className: "desk-status"
  }, r.status.replaceAll('_', ' '))), /*#__PURE__*/React.createElement("p", null, "Created ", stamp(r.createdAt), r.updatedAt && r.updatedAt !== r.createdAt ? ` · Updated ${stamp(r.updatedAt)}` : ''), (r.parentId || r.proposalId) && /*#__PURE__*/React.createElement("details", null, /*#__PURE__*/React.createElement("summary", null, "Recorded version links"), r.parentId && /*#__PURE__*/React.createElement("p", null, "Parent / target document: ", r.parentId), r.proposalId && /*#__PURE__*/React.createElement("p", null, "Revision proposal: ", r.proposalId)), /*#__PURE__*/React.createElement("details", null, /*#__PURE__*/React.createElement("summary", null, "Record identifier"), /*#__PURE__*/React.createElement("code", null, r.recordId)), /*#__PURE__*/React.createElement("button", {
    onClick: () => open(r)
  }, "Open ", r.kind === 'collection' ? 'collection controls' : r.kind === 'activity' ? 'Analyst team' : r.kind === 'note' ? 'Research Pipeline' : 'related workspace', " \u2192")))))));
}