import * as React from 'react';
import { collectionRequests, collectionOpen, collectionTime, collectionNext } from './collection-request-model.mjs';
export function CollectionRequests({
  requests = [],
  query,
  onQuery,
  busy,
  onAction,
  available,
  focusId = '',
  onClearFocus
}) {
  var [finished, setFinished] = React.useState(false);
  var rows = focusId ? requests.filter(r => r.id === focusId && (!query || r.ticker === query)) : collectionRequests(requests, query, finished);
  return /*#__PURE__*/React.createElement("section", {
    className: "workspace-panel",
    "aria-label": "Collection requests"
  }, /*#__PURE__*/React.createElement("div", {
    className: "workspace-section-heading"
  }, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("p", {
    className: "workspace-eyebrow"
  }, "COLLECTION / FOLLOW-THROUGH"), /*#__PURE__*/React.createElement("h3", null, query ? `${query.toUpperCase()} collection` : 'Collection requests')), /*#__PURE__*/React.createElement("button", {
    disabled: !query,
    onClick: () => {
      onClearFocus?.();
      onQuery('');
    }
  }, "Show all companies")), /*#__PURE__*/React.createElement("p", {
    hidden: !focusId,
    role: "status"
  }, "Opened request ", focusId.slice(0, 8), ". ", /*#__PURE__*/React.createElement("button", {
    onClick: onClearFocus
  }, "Show all company requests")), /*#__PURE__*/React.createElement("div", {
    className: "desk-filter"
  }, /*#__PURE__*/React.createElement("label", null, "Find collection company", /*#__PURE__*/React.createElement("input", {
    value: query,
    maxLength: 20,
    onChange: e => {
      onClearFocus?.();
      onQuery(e.target.value.toUpperCase());
    },
    placeholder: "Ticker"
  })), /*#__PURE__*/React.createElement("label", null, /*#__PURE__*/React.createElement("input", {
    type: "checkbox",
    checked: finished,
    onChange: e => setFinished(e.target.checked)
  }), "Include completed and cancelled")), /*#__PURE__*/React.createElement("p", {
    className: "desk-explainer"
  }, rows.length, " matching requests in the latest ", requests.length, " reported by this Mac (up to 50). Search and document counts describe collection progress, not report completion."), !available ? /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, "Waiting for a Mac collection snapshot.") : !rows.length ? /*#__PURE__*/React.createElement("p", null, focusId ? 'The linked request is outside this snapshot or belongs to another company. No replacement request is selected.' : `No ${finished ? 'matching' : 'unfinished'} collection requests in this snapshot.`) : rows.map(r => /*#__PURE__*/React.createElement("article", {
    className: "workspace-panel",
    key: r.id
  }, /*#__PURE__*/React.createElement("div", {
    className: "workspace-section-heading"
  }, /*#__PURE__*/React.createElement("h4", null, r.ticker, " \xB7 ", String(r.status || 'unknown').replaceAll('_', ' ')), /*#__PURE__*/React.createElement("small", null, "Submitted ", collectionTime(r.created)?.toLocaleString() || 'time unavailable')), r.issue && /*#__PURE__*/React.createElement("p", {
    className: "workspace-error",
    style: {
      whiteSpace: 'pre-wrap',
      overflowWrap: 'anywhere'
    }
  }, r.issue), /*#__PURE__*/React.createElement("p", null, /*#__PURE__*/React.createElement("strong", null, "Next:"), " ", collectionNext(r)), r.result?.newDocuments != null && /*#__PURE__*/React.createElement("p", null, r.result.newDocuments, " new eligible originals reported."), /*#__PURE__*/React.createElement("details", null, /*#__PURE__*/React.createElement("summary", null, "Saved searches and originals"), r.sourceProgress?.tasks?.length ? /*#__PURE__*/React.createElement("ul", null, r.sourceProgress.tasks.map((t, i) => /*#__PURE__*/React.createElement("li", {
    key: i
  }, t.kind, " \xB7 ", t.status, t.expected != null ? ` · ${t.expected} expected` : ''))) : /*#__PURE__*/React.createElement("p", null, "No search-progress receipt available."), Object.entries(r.sourceProgress?.documents || {}).map(([state, count]) => /*#__PURE__*/React.createElement("p", {
    key: state
  }, String(state).replaceAll('_', ' '), ": ", count)), /*#__PURE__*/React.createElement("p", {
    className: "desk-explainer"
  }, "Request ", r.id, ". Staged or handed-off counts alone do not prove cloud import or downstream research completion.")), /*#__PURE__*/React.createElement("div", {
    className: "desk-filter"
  }, ['attention', 'needs_auth'].includes(r.status) && /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: () => onAction('retry', {
      ticker: r.ticker,
      refreshRequestId: r.id
    })
  }, r.status === 'needs_auth' ? 'Signed in on Mac — resume' : 'Retry after resolving'), collectionOpen(r) && /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: () => onAction('cancel', {
      ticker: r.ticker,
      refreshRequestId: r.id
    })
  }, "Cancel this refresh")))), /*#__PURE__*/React.createElement("p", {
    className: "desk-explainer"
  }, "Cancellation takes effect when acknowledged on Mac; an active download may finish. Sign-in and MFA happen directly in AlphaSense."));
}